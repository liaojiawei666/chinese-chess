//! tch-rs 加载 TorchScript model.pt 并前向（移植 evaluator.py 的网络前向部分）。
//! 仅在启用 `torch` 特性时编译，需要 libtorch（见 README）。

use engine::{GameState, Move, BOARD_HEIGHT, BOARD_WIDTH, INPUT_CHANNELS};
use tch::{CModule, Device, IValue, Tensor};

use crate::{priors_from_logits, LeafEvaluator};

/// 加载好的策略-价值网络（TorchScript）。前向签名：
/// 输入 (N, 99, 10, 9) f32 → 输出 (policy_logits [N,8100], value [N,1])。
pub struct TorchModel {
    module: CModule,
    device: Device,
}

/// Windows 上 torch-sys 虽链接了 torch_cuda，但 MSVC 会丢弃未引用的 import，
/// 导致 torch_cuda.dll 运行时不加载、CUDA dispatch key 未注册（前向会报
/// “Could not run 'aten::*' with arguments from the 'CUDA' backend”）。
/// 这里在用 CUDA 设备加载模型前显式 LoadLibrary 一次，触发其静态初始化注册 CUDA 后端。
/// 依赖的 cudart/cublas/cudnn 与之同在 torch/lib（运行期该目录已在 PATH 上），可正常解析。
#[cfg(target_os = "windows")]
fn ensure_cuda_loaded() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        extern "system" {
            fn LoadLibraryW(name: *const u16) -> *mut std::ffi::c_void;
        }
        let wide: Vec<u16> =
            "torch_cuda.dll".encode_utf16().chain(std::iter::once(0)).collect();
        let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
        if handle.is_null() {
            eprintln!(
                "警告：加载 torch_cuda.dll 失败，请确认 venv 的 torch/lib 在 PATH 上；CUDA 推理将不可用"
            );
        }
    });
}

#[cfg(not(target_os = "windows"))]
fn ensure_cuda_loaded() {}

impl TorchModel {
    pub fn load(path: &str, device: Device) -> Result<Self, tch::TchError> {
        if matches!(device, Device::Cuda(_)) {
            ensure_cuda_loaded();
        }
        let mut module = CModule::load_on_device(path, device)?;
        module.set_eval();
        Ok(TorchModel { module, device })
    }

    /// 用设备字符串（"cpu" | "cuda" | "mps"）加载，免得调用方（selfplay bin）
    /// 直接依赖 tch 才能构造 `Device`。
    pub fn load_str(path: &str, device: &str) -> Result<Self, tch::TchError> {
        let dev = match device {
            "cuda" => Device::Cuda(0),
            "mps" => Device::Mps,
            _ => Device::Cpu,
        };
        Self::load(path, dev)
    }

    /// 单局面前向：输入展平的 (99*10*9) f32，返回 (policy_logits[8100], value)。
    pub fn forward(&self, input: &[f32]) -> (Vec<f32>, f32) {
        let tensor = Tensor::from_slice(input)
            .reshape([
                1,
                INPUT_CHANNELS as i64,
                BOARD_HEIGHT as i64,
                BOARD_WIDTH as i64,
            ])
            .to_device(self.device);

        let output = self
            .module
            .forward_is(&[IValue::Tensor(tensor)])
            .expect("model forward failed");

        let (logits, value) = match output {
            IValue::Tuple(mut items) => {
                assert_eq!(items.len(), 2, "expected (policy_logits, value) tuple");
                let value = items.pop().unwrap();
                let logits = items.pop().unwrap();
                (tensor_of(logits), tensor_of(value))
            }
            other => panic!("unexpected model output: {other:?}"),
        };

        let logits_vec: Vec<f32> = Vec::<f32>::try_from(logits.reshape(-1).to_kind(tch::Kind::Float))
            .expect("policy logits -> Vec<f32>");
        let value_scalar = value.double_value(&[0, 0]) as f32;
        (logits_vec, value_scalar)
    }

    /// 批量前向：把 N 个展平局面堆成 (N,99,10,9) 一次前向，返回每个的
    /// (policy_logits[8100], value)。供推理 actor 跨 worker 聚合后凑批调用。
    pub fn forward_batch(&self, inputs: &[&[f32]]) -> Vec<(Vec<f32>, f32)> {
        let n = inputs.len();
        if n == 0 {
            return Vec::new();
        }
        let per = INPUT_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH;
        let mut buf = Vec::with_capacity(n * per);
        for inp in inputs {
            debug_assert_eq!(inp.len(), per, "每个局面应为 {per} 维");
            buf.extend_from_slice(inp);
        }

        let tensor = Tensor::from_slice(&buf)
            .reshape([
                n as i64,
                INPUT_CHANNELS as i64,
                BOARD_HEIGHT as i64,
                BOARD_WIDTH as i64,
            ])
            .to_device(self.device);

        let output = self
            .module
            .forward_is(&[IValue::Tensor(tensor)])
            .expect("model forward failed");

        let (logits, value) = match output {
            IValue::Tuple(mut items) => {
                assert_eq!(items.len(), 2, "expected (policy_logits, value) tuple");
                let value = items.pop().unwrap();
                let logits = items.pop().unwrap();
                (tensor_of(logits), tensor_of(value))
            }
            other => panic!("unexpected model output: {other:?}"),
        };

        let logits_all: Vec<f32> =
            Vec::<f32>::try_from(logits.reshape(-1).to_kind(tch::Kind::Float))
                .expect("policy logits -> Vec<f32>");
        let action = logits_all.len() / n;

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let row = logits_all[i * action..(i + 1) * action].to_vec();
            let v = value.double_value(&[i as i64, 0]) as f32;
            out.push((row, v));
        }
        out
    }
}

fn tensor_of(value: IValue) -> Tensor {
    match value {
        IValue::Tensor(t) => t,
        other => panic!("expected tensor, got {other:?}"),
    }
}

impl LeafEvaluator for TorchModel {
    fn evaluate(&self, state: &GameState) -> (Vec<(Move, f32)>, f32) {
        let input = encoder::encode(state);
        let (logits, value) = self.forward(&input);
        (priors_from_logits(&logits, state), value)
    }
}
