use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use engine::encode::EncodedState;

/// Shard 文件魔数："CXSG" (Chinese Xiangqi Self-play Game)
const MAGIC: u32 = 0x4358_5347;

/// 单个训练样本。
pub struct TrainingSample {
    pub state: EncodedState,
    /// 稀疏策略目标：(action_id, probability) 对
    pub policy: Vec<(u16, f32)>,
    /// 终局结果（当前行棋方视角）: +1 胜, -1 负, 0 和
    pub value: f32,
}

/// 一局自我对弈中间状态（value 尚未回填）。
pub struct PendingSample {
    pub state: EncodedState,
    pub policy: Vec<(u16, f32)>,
    pub side_to_move: engine::Color,
}

/// 回填 value：对局结束后根据胜负结果，对每个样本设置 value。
pub fn finalize_samples(
    pending: Vec<PendingSample>,
    winner: Option<engine::Color>,
) -> Vec<TrainingSample> {
    pending
        .into_iter()
        .map(|s| {
            let value = match winner {
                Some(w) if w == s.side_to_move => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            };
            TrainingSample {
                state: s.state,
                policy: s.policy,
                value,
            }
        })
        .collect()
}

/// 将一批训练样本写入 shard 文件（见 docs/pipeline-protocol.md §3）。
pub fn write_shard(path: &Path, samples: &[TrainingSample]) -> Result<()> {
    let mut buf = Vec::with_capacity(samples.len() * 10_000);

    buf.extend_from_slice(&MAGIC.to_le_bytes());
    buf.extend_from_slice(&(samples.len() as u32).to_le_bytes());

    for sample in samples {
        buf.extend_from_slice(&sample.state.pieces);
        buf.push(sample.state.no_capture_plies);

        let count = sample.policy.len() as u16;
        buf.extend_from_slice(&count.to_le_bytes());
        for &(id, _) in &sample.policy {
            buf.extend_from_slice(&id.to_le_bytes());
        }
        for &(_, prob) in &sample.policy {
            buf.extend_from_slice(&prob.to_le_bytes());
        }

        buf.extend_from_slice(&sample.value.to_le_bytes());
    }

    // 原子写：先写临时文件再 rename，避免 trainer 读到写了一半的 shard。
    // 临时名带 .tmp 后缀，不匹配 trainer 的 `^shard_\d{6}\.bin$` 正则，不会被扫到。
    let tmp_path = path.with_file_name(format!(
        "{}.tmp",
        path.file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "shard.bin".to_string())
    ));
    {
        let mut file = std::fs::File::create(&tmp_path)
            .with_context(|| format!("failed to create shard tmp: {}", tmp_path.display()))?;
        file.write_all(&buf)?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp_path, path)
        .with_context(|| format!("failed to finalize shard: {}", path.display()))?;
    Ok(())
}

/// Shard 管理器：收集训练样本，满一个 shard 就写出。
pub struct ShardWriter {
    samples_dir: String,
    shard_size: usize,
    buffer: Vec<TrainingSample>,
    shard_index: u32,
    total_samples: u64,
}

impl ShardWriter {
    pub fn new(samples_dir: String, shard_size: usize) -> Self {
        std::fs::create_dir_all(&samples_dir).ok();
        Self {
            samples_dir,
            shard_size,
            buffer: Vec::with_capacity(shard_size),
            shard_index: 0,
            total_samples: 0,
        }
    }

    pub fn add_game_samples(&mut self, samples: Vec<TrainingSample>) -> Result<()> {
        self.buffer.extend(samples);

        while self.buffer.len() >= self.shard_size {
            let rest = self.buffer.split_off(self.shard_size);
            self.flush_shard()?;
            self.buffer = rest;
        }

        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        if !self.buffer.is_empty() {
            self.flush_shard()?;
        }
        Ok(())
    }

    fn flush_shard(&mut self) -> Result<()> {
        let path = Path::new(&self.samples_dir).join(format!("shard_{:06}.bin", self.shard_index));
        let count = self.buffer.len();
        write_shard(&path, &self.buffer)?;
        self.total_samples += count as u64;
        self.shard_index += 1;
        self.buffer.clear();
        tracing::info!(
            "wrote shard {} ({count} samples, total: {})",
            path.display(),
            self.total_samples,
        );
        Ok(())
    }

    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::encode::PIECES_SIZE;

    fn make_sample(value: f32, no_capture_plies: u8) -> TrainingSample {
        TrainingSample {
            state: EncodedState {
                pieces: [0u8; PIECES_SIZE],
                no_capture_plies,
            },
            policy: vec![(10, 0.6), (42, 0.3), (99, 0.1)],
            value,
        }
    }

    #[test]
    fn test_shard_write_read() {
        let dir = std::env::temp_dir().join("cxsg_test_write_read");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let samples = vec![
            make_sample(1.0, 0),
            make_sample(-1.0, 50),
            make_sample(0.0, 100),
        ];
        let path = dir.join("test.bin");
        write_shard(&path, &samples).unwrap();

        let bytes = std::fs::read(&path).unwrap();

        // Verify magic
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(magic, MAGIC, "CXSG magic number");

        // Verify sample count
        let count = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(count, 3);

        // Verify we can walk the binary format
        let mut offset = 8;
        for (i, s) in samples.iter().enumerate() {
            // state_pieces
            assert_eq!(&bytes[offset..offset + PIECES_SIZE], &s.state.pieces[..]);
            offset += PIECES_SIZE;

            // no_capture_plies
            assert_eq!(bytes[offset], s.state.no_capture_plies, "sample {i}");
            offset += 1;

            // policy count
            let pc = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            assert_eq!(pc as usize, s.policy.len());
            offset += 2;

            // policy ids
            for (j, &(id, _)) in s.policy.iter().enumerate() {
                let read_id = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                assert_eq!(read_id, id, "sample {i} policy id {j}");
                offset += 2;
            }

            // policy probs
            for (j, &(_, prob)) in s.policy.iter().enumerate() {
                let read_prob = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                assert!(
                    (read_prob - prob).abs() < 1e-6,
                    "sample {i} policy prob {j}"
                );
                offset += 4;
            }

            // value
            let read_val = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            assert!((read_val - s.value).abs() < 1e-6, "sample {i} value");
            offset += 4;
        }

        assert_eq!(offset, bytes.len(), "all bytes consumed");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_shard_writer_flush() {
        let dir = std::env::temp_dir().join("cxsg_test_writer");
        let _ = std::fs::remove_dir_all(&dir);

        let mut writer = ShardWriter::new(dir.to_string_lossy().into_owned(), 2);

        let samples = vec![make_sample(1.0, 0), make_sample(-1.0, 10), make_sample(0.0, 20)];
        writer.add_game_samples(samples).unwrap();

        // With shard_size=2, one shard should have been written (2 samples) + 1 in buffer
        assert_eq!(writer.total_samples(), 2);
        assert!(dir.join("shard_000000.bin").exists());

        writer.flush().unwrap();
        assert_eq!(writer.total_samples(), 3);
        assert!(dir.join("shard_000001.bin").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_finalize_samples_red_wins() {
        let pending = vec![
            PendingSample {
                state: EncodedState { pieces: [0; PIECES_SIZE], no_capture_plies: 0 },
                policy: vec![(0, 1.0)],
                side_to_move: engine::Color::Red,
            },
            PendingSample {
                state: EncodedState { pieces: [0; PIECES_SIZE], no_capture_plies: 0 },
                policy: vec![(0, 1.0)],
                side_to_move: engine::Color::Black,
            },
        ];

        let samples = finalize_samples(pending, Some(engine::Color::Red));
        assert_eq!(samples[0].value, 1.0, "red sample should be +1 on red win");
        assert_eq!(samples[1].value, -1.0, "black sample should be -1 on red win");
    }

    #[test]
    fn test_finalize_samples_draw() {
        let pending = vec![
            PendingSample {
                state: EncodedState { pieces: [0; PIECES_SIZE], no_capture_plies: 0 },
                policy: vec![(0, 1.0)],
                side_to_move: engine::Color::Red,
            },
            PendingSample {
                state: EncodedState { pieces: [0; PIECES_SIZE], no_capture_plies: 0 },
                policy: vec![(0, 1.0)],
                side_to_move: engine::Color::Black,
            },
        ];

        let samples = finalize_samples(pending, None);
        assert_eq!(samples[0].value, 0.0, "draw → value 0");
        assert_eq!(samples[1].value, 0.0, "draw → value 0");
    }
}
