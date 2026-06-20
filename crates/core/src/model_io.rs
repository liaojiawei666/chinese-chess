//! 样本分片写盘 + 模型指针读取（对应 shared/shard-format.md 与 shared/model-format.md）。
//!
//! - 分片：safetensors 布局（state uint8 + 稀疏 π 的 CSR + z），原子写（.tmp → fsync → rename）。
//! - 背压：`LocalSampleStore::pending_count` 数未消费的 `.st`，selfplay 据此节流。
//! - 模型：读 `latest.json` 取 version / 对应 .pt 路径，供热加载。

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use crate::engine::{BOARD_HEIGHT, BOARD_WIDTH, INPUT_CHANNELS};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::serialize;
use serde::Deserialize;

/// 一条训练样本（datagen 侧表示）：state 已量化成 uint8，π 已是稀疏 (idx,val)。
pub struct Sample {
    /// 长度 INPUT_CHANNELS*10*9 的量化局面张量（round(v*255)），C 序展平。
    pub state: Vec<u8>,
    /// 稀疏 π 的 canonical action_id。
    pub pi_idx: Vec<i32>,
    /// 稀疏 π 的概率值（与 pi_idx 同序）。
    pub pi_val: Vec<f32>,
    /// 走棋方视角的终局价值 +1/0/-1。
    pub z: f32,
}

const STATE_PER_SAMPLE: usize = INPUT_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH;

/// 把若干样本序列化成一个 safetensors 分片字节流（布局见 shared/shard-format.md）。
pub fn serialize_shard(samples: &[Sample]) -> Result<Vec<u8>> {
    let n = samples.len();

    let mut state = Vec::with_capacity(n * STATE_PER_SAMPLE);
    let mut pi_ptr: Vec<i32> = Vec::with_capacity(n + 1);
    let mut pi_idx: Vec<i32> = Vec::new();
    let mut pi_val: Vec<f32> = Vec::new();
    let mut z: Vec<f32> = Vec::with_capacity(n);

    let mut offset: i32 = 0;
    pi_ptr.push(0);
    for s in samples {
        anyhow::ensure!(
            s.state.len() == STATE_PER_SAMPLE,
            "state 长度 {} != {STATE_PER_SAMPLE}",
            s.state.len()
        );
        anyhow::ensure!(
            s.pi_idx.len() == s.pi_val.len(),
            "pi_idx/pi_val 长度不一致"
        );
        state.extend_from_slice(&s.state);
        pi_idx.extend_from_slice(&s.pi_idx);
        pi_val.extend_from_slice(&s.pi_val);
        offset += s.pi_idx.len() as i32;
        pi_ptr.push(offset);
        z.push(s.z);
    }

    let pi_ptr_bytes = i32_le_bytes(&pi_ptr);
    let pi_idx_bytes = i32_le_bytes(&pi_idx);
    let pi_val_bytes = f32_le_bytes(&pi_val);
    let z_bytes = f32_le_bytes(&z);

    let nnz = pi_idx.len();
    let views = vec![
        (
            "state".to_string(),
            TensorView::new(Dtype::U8, vec![n, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH], &state)?,
        ),
        (
            "pi_ptr".to_string(),
            TensorView::new(Dtype::I32, vec![n + 1], &pi_ptr_bytes)?,
        ),
        (
            "pi_idx".to_string(),
            TensorView::new(Dtype::I32, vec![nnz], &pi_idx_bytes)?,
        ),
        (
            "pi_val".to_string(),
            TensorView::new(Dtype::F32, vec![nnz], &pi_val_bytes)?,
        ),
        ("z".to_string(), TensorView::new(Dtype::F32, vec![n], &z_bytes)?),
    ];

    let bytes = serialize(views, &None).context("safetensors 序列化失败")?;
    Ok(bytes)
}

fn i32_le_bytes(v: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn f32_le_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// 分片文件名：shard_{model_version:06}_w{worker:02}_{seq:06}.st。
pub fn shard_name(model_version: i64, worker: usize, seq: u64) -> String {
    format!("shard_{:06}_w{:02}_{:06}.st", model_version, worker, seq)
}

/// 样本分片落盘接口（屏蔽本地盘 / 未来 OSS）。
pub trait SampleStore: Send + Sync {
    /// 原子写一个分片。
    fn put_shard(&self, name: &str, bytes: &[u8]) -> Result<()>;
    /// 未消费分片数（背压用）。
    fn pending_count(&self) -> Result<usize>;
}

/// 本地目录实现：先写 `*.st.tmp`，fsync 后 rename 成 `*.st`。
pub struct LocalSampleStore {
    dir: PathBuf,
}

impl LocalSampleStore {
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)
            .with_context(|| format!("创建样本目录失败：{}", dir.display()))?;
        Ok(LocalSampleStore { dir })
    }
}

impl SampleStore for LocalSampleStore {
    fn put_shard(&self, name: &str, bytes: &[u8]) -> Result<()> {
        let final_path = self.dir.join(name);
        let tmp_path = self.dir.join(format!("{name}.tmp"));
        {
            let mut f = File::create(&tmp_path)
                .with_context(|| format!("创建分片临时文件失败：{}", tmp_path.display()))?;
            f.write_all(bytes)?;
            f.sync_all()?;
        }
        atomic_rename(&tmp_path, &final_path)
            .with_context(|| format!("rename 分片失败：{}", final_path.display()))?;
        Ok(())
    }

    fn pending_count(&self) -> Result<usize> {
        let mut count = 0;
        for entry in fs::read_dir(&self.dir)
            .with_context(|| format!("读取样本目录失败：{}", self.dir.display()))?
        {
            let entry = entry?;
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".st") {
                    count += 1;
                }
            }
        }
        Ok(count)
    }
}

/// 原子 rename：Unix 上 rename(2) 天然原子；Windows NTFS rename 在目标存在时可能报错，
/// 改用 MoveFileExW(MOVEFILE_REPLACE_EXISTING) 保证覆盖语义。
#[cfg(not(target_os = "windows"))]
fn atomic_rename(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    fs::rename(from, to)
}

#[cfg(target_os = "windows")]
fn atomic_rename(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    extern "system" {
        fn MoveFileExW(src: *const u16, dst: *const u16, flags: u32) -> i32;
    }
    const MOVEFILE_REPLACE_EXISTING: u32 = 1;
    let wide = |p: &std::path::Path| -> Vec<u16> {
        p.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
    };
    let ret = unsafe { MoveFileExW(wide(from).as_ptr(), wide(to).as_ptr(), MOVEFILE_REPLACE_EXISTING) };
    if ret == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// latest.json 内容。
#[derive(Debug, Clone, Deserialize)]
pub struct ModelPointer {
    pub version: i64,
    pub path: String,
    #[serde(default)]
    pub ts: String,
}

/// 模型读取接口：只读 latest.json 版本（便宜）/ 取最新模型 .pt 的绝对路径。
pub trait ModelStore: Send + Sync {
    fn get_version(&self) -> Result<Option<i64>>;
    fn get_latest_path(&self) -> Result<Option<(i64, PathBuf)>>;
}

/// 本地目录实现：读 `<dir>/latest.json`。
pub struct LocalModelStore {
    dir: PathBuf,
}

impl LocalModelStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        LocalModelStore { dir: dir.into() }
    }

    fn read_pointer(&self) -> Result<Option<ModelPointer>> {
        let path = self.dir.join("latest.json");
        if !path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path)
            .with_context(|| format!("读取 latest.json 失败：{}", path.display()))?;
        let pointer: ModelPointer =
            serde_json::from_str(&text).context("解析 latest.json 失败")?;
        Ok(Some(pointer))
    }

    /// 列出目录下所有 `model_{step:06}.pt` 的版本号（升序）。目录不存在则返回空。
    pub fn list_versions(&self) -> Result<Vec<i64>> {
        let mut versions = Vec::new();
        if !self.dir.exists() {
            return Ok(versions);
        }
        for entry in fs::read_dir(&self.dir)
            .with_context(|| format!("读取模型目录失败：{}", self.dir.display()))?
        {
            let entry = entry?;
            if let Some(name) = entry.file_name().to_str() {
                if let Some(v) = parse_model_version(name) {
                    versions.push(v);
                }
            }
        }
        versions.sort_unstable();
        Ok(versions)
    }

    /// 版本号 → 对应 `model_{version:06}.pt` 的路径（不校验存在性）。
    pub fn path_for(&self, version: i64) -> PathBuf {
        self.dir.join(format!("model_{version:06}.pt"))
    }
}

/// 解析 `model_{digits}.pt` 文件名为版本号；不匹配则 None。
fn parse_model_version(name: &str) -> Option<i64> {
    let digits = name.strip_prefix("model_")?.strip_suffix(".pt")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

impl ModelStore for LocalModelStore {
    fn get_version(&self) -> Result<Option<i64>> {
        Ok(self.read_pointer()?.map(|p| p.version))
    }

    fn get_latest_path(&self) -> Result<Option<(i64, PathBuf)>> {
        match self.read_pointer()? {
            Some(p) => Ok(Some((p.version, self.dir.join(p.path)))),
            None => Ok(None),
        }
    }
}

/// 把 encoder 产出的 f32 局面张量量化成 uint8（round(v*255)，clamp 到 0..=255）。
pub fn quantize_state(state_f32: &[f32]) -> Vec<u8> {
    state_f32
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::SafeTensors;

    #[test]
    fn shard_roundtrip() {
        let s0 = Sample {
            state: vec![0u8; STATE_PER_SAMPLE],
            pi_idx: vec![3, 7, 11],
            pi_val: vec![0.5, 0.3, 0.2],
            z: 1.0,
        };
        let mut state1 = vec![0u8; STATE_PER_SAMPLE];
        state1[0] = 255;
        let s1 = Sample {
            state: state1,
            pi_idx: vec![42],
            pi_val: vec![1.0],
            z: -1.0,
        };
        let bytes = serialize_shard(&[s0, s1]).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();

        let pi_ptr = st.tensor("pi_ptr").unwrap();
        assert_eq!(pi_ptr.shape(), &[3]);
        let z = st.tensor("z").unwrap();
        assert_eq!(z.shape(), &[2]);
        let state = st.tensor("state").unwrap();
        assert_eq!(state.shape(), &[2, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH]);
        let pi_idx = st.tensor("pi_idx").unwrap();
        assert_eq!(pi_idx.shape(), &[4]);
    }

    #[test]
    fn quantize_binary_and_ratio() {
        assert_eq!(quantize_state(&[1.0, 0.0]), vec![255, 0]);
        assert_eq!(quantize_state(&[0.5]), vec![128]);
    }

    #[test]
    fn local_store_atomic_and_count() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalSampleStore::new(dir.path()).unwrap();
        assert_eq!(store.pending_count().unwrap(), 0);
        store.put_shard("shard_000000_w00_000000.st", b"hello").unwrap();
        assert_eq!(store.pending_count().unwrap(), 1);
        // 临时文件不应被计入
        assert!(!dir.path().join("shard_000000_w00_000000.st.tmp").exists());
    }

    #[test]
    fn model_store_reads_pointer() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());
        assert_eq!(store.get_version().unwrap(), None);
        fs::write(
            dir.path().join("latest.json"),
            r#"{"version": 40, "path": "model_000040.pt", "ts": "x"}"#,
        )
        .unwrap();
        assert_eq!(store.get_version().unwrap(), Some(40));
        let (v, p) = store.get_latest_path().unwrap().unwrap();
        assert_eq!(v, 40);
        assert!(p.ends_with("model_000040.pt"));
    }

    #[test]
    fn model_store_lists_versions_and_resolves_path() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalModelStore::new(dir.path());
        assert_eq!(store.list_versions().unwrap(), Vec::<i64>::new());

        for v in [40i64, 100, 20] {
            fs::write(store.path_for(v), b"x").unwrap();
        }
        // 干扰文件不应入选。
        fs::write(dir.path().join("latest.json"), b"{}").unwrap();
        fs::write(dir.path().join("model_bad.pt"), b"x").unwrap();
        fs::write(dir.path().join("model_000050.pt.tmp"), b"x").unwrap();

        assert_eq!(store.list_versions().unwrap(), vec![20, 40, 100]);
        assert!(store.path_for(100).ends_with("model_000100.pt"));
    }
}
