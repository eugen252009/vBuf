//! CUDA backend adapter.
//!
//! CUDA runtime, cuBLAS, and kernel details are confined to this module and
//! the linked CUDA implementation. The graph executor below consumes only the
//! portable graph and opaque `DeviceTensor` values.

use crate::device::{
    DeviceBackend, DeviceCapabilities, DeviceDType, DeviceId, DeviceKind, DeviceStorage,
    DeviceTensor,
};
use crate::generic::GenericTopKSelection;
use crate::graph::{
    ActivationKind, AttentionMaskKind, ExecutionGraph, InputRef, OperationKind, TensorId, ValueId,
};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

#[repr(C)]
struct CudaDeviceInfo {
    name: [c_char; 256],
    major: c_int,
    minor: c_int,
    total_bytes: u64,
    free_bytes: u64,
}

#[repr(C)]
struct CudaContext {
    _private: [u8; 0],
}

#[repr(C)]
struct CudaTensor {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn vbuf_cuda_device_count() -> c_int;
    fn vbuf_cuda_device_info(index: c_int, info: *mut CudaDeviceInfo) -> c_int;
    fn vbuf_cuda_last_error() -> *const c_char;
    fn vbuf_cuda_synchronize(context: *mut CudaContext) -> c_int;
    fn vbuf_cuda_memory_info(
        context: *mut CudaContext,
        free_bytes: *mut u64,
        total_bytes: *mut u64,
    ) -> c_int;
    fn vbuf_cuda_context_create(index: c_int, result: *mut *mut CudaContext) -> c_int;
    fn vbuf_cuda_context_destroy(context: *mut CudaContext) -> c_int;
    fn vbuf_cuda_tensor_destroy(tensor: *mut CudaTensor) -> c_int;
    fn vbuf_cuda_tensor_upload(
        context: *mut CudaContext,
        values: *const f32,
        count: u64,
        rank: c_int,
        dimensions: *const u64,
        result: *mut *mut CudaTensor,
    ) -> c_int;
    fn vbuf_cuda_tensor_download(tensor: *mut CudaTensor, values: *mut f32, count: u64) -> c_int;
    fn vbuf_cuda_rms_norm(
        input: *mut CudaTensor,
        weight: *mut CudaTensor,
        epsilon: f32,
        result: *mut *mut CudaTensor,
    ) -> c_int;
    fn vbuf_cuda_matmul(
        input: *mut CudaTensor,
        weight: *mut CudaTensor,
        result: *mut *mut CudaTensor,
    ) -> c_int;
    fn vbuf_cuda_add(
        left: *mut CudaTensor,
        right: *mut CudaTensor,
        result: *mut *mut CudaTensor,
    ) -> c_int;
    fn vbuf_cuda_mul(
        left: *mut CudaTensor,
        right: *mut CudaTensor,
        result: *mut *mut CudaTensor,
    ) -> c_int;
    fn vbuf_cuda_bias_add(
        input: *mut CudaTensor,
        bias: *mut CudaTensor,
        result: *mut *mut CudaTensor,
    ) -> c_int;
    fn vbuf_cuda_activation(
        input: *mut CudaTensor,
        kind: c_int,
        result: *mut *mut CudaTensor,
    ) -> c_int;
    fn vbuf_cuda_reshape(
        input: *mut CudaTensor,
        rank: c_int,
        dimensions: *const u64,
        result: *mut *mut CudaTensor,
    ) -> c_int;
    fn vbuf_cuda_rotary(
        input: *mut CudaTensor,
        heads: u64,
        head_dim: u64,
        rotary_dim: u64,
        theta: f32,
        position_start: u64,
        result: *mut *mut CudaTensor,
    ) -> c_int;
    fn vbuf_cuda_attention(
        query: *mut CudaTensor,
        key: *mut CudaTensor,
        value: *mut CudaTensor,
        q_heads: u64,
        kv_heads: u64,
        head_dim: u64,
        scale: f32,
        result: *mut *mut CudaTensor,
    ) -> c_int;
    fn vbuf_cuda_expert_dispatch(
        input: *mut CudaTensor,
        gate: *mut CudaTensor,
        up: *mut CudaTensor,
        down: *mut CudaTensor,
        ids: *const u32,
        weights: *const f32,
        token_count: u64,
        top_k: u64,
        expert: u32,
        result: *mut *mut CudaTensor,
    ) -> c_int;
}

fn last_error() -> String {
    // SAFETY: The backend owns a thread-local diagnostic string.
    unsafe {
        let pointer = vbuf_cuda_last_error();
        if pointer.is_null() {
            "CUDA backend operation failed".into()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    }
}

fn check(status: c_int) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(last_error())
    }
}

#[derive(Debug, Default)]
struct AllocationStats {
    live_tensors: u64,
    live_bytes: u64,
    peak_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct CudaTelemetry {
    pub total_ops: u64,
    pub cuda_ops: u64,
    pub host_control_ops: u64,
    pub cpu_fallback_ops: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub control_to_device_bytes: u64,
    pub control_to_host_bytes: u64,
    pub persistent_bytes_read: u64,
    pub selected_expert_ids: Vec<u32>,
    pub device_weight_peak_bytes: u64,
}

impl Default for CudaTelemetry {
    fn default() -> Self {
        Self {
            total_ops: 0,
            cuda_ops: 0,
            host_control_ops: 0,
            cpu_fallback_ops: 0,
            host_to_device_bytes: 0,
            device_to_host_bytes: 0,
            control_to_device_bytes: 0,
            control_to_host_bytes: 0,
            persistent_bytes_read: 0,
            selected_expert_ids: Vec::new(),
            device_weight_peak_bytes: 0,
        }
    }
}

#[derive(Debug)]
pub struct CudaBackend {
    context: NonNull<CudaContext>,
    capabilities: DeviceCapabilities,
    stats: Arc<Mutex<AllocationStats>>,
    telemetry: Mutex<CudaTelemetry>,
}

unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // SAFETY: context is created by the constructor and lives until Drop.
        unsafe {
            let _ = vbuf_cuda_context_destroy(self.context.as_ptr());
        }
    }
}

impl CudaBackend {
    pub fn device_count() -> u32 {
        // SAFETY: CUDA runtime query has no Rust-managed pointers.
        unsafe { vbuf_cuda_device_count().max(0) as u32 }
    }

    pub fn device_info(index: u32) -> Result<DeviceCapabilities, String> {
        let mut info = CudaDeviceInfo {
            name: [0; 256],
            major: 0,
            minor: 0,
            total_bytes: 0,
            free_bytes: 0,
        };
        // SAFETY: info is a valid writable output buffer.
        unsafe {
            check(vbuf_cuda_device_info(index as c_int, &mut info))?;
        }
        let end = info
            .name
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(info.name.len());
        let name = info.name[..end]
            .iter()
            .map(|byte| *byte as u8)
            .collect::<Vec<_>>();
        Ok(DeviceCapabilities {
            id: DeviceId(index),
            kind: DeviceKind::Accelerator,
            name: String::from_utf8_lossy(&name).into_owned(),
            compute_capability: Some((info.major as u32, info.minor as u32)),
            total_bytes: info.total_bytes,
            free_bytes: info.free_bytes,
        })
    }

    pub fn new(index: u32) -> Result<Self, String> {
        let capabilities = Self::device_info(index)?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: raw is a valid output pointer.
        unsafe {
            check(vbuf_cuda_context_create(index as c_int, &mut raw))?;
        }
        let context = NonNull::new(raw).ok_or("CUDA context was null")?;
        Ok(Self {
            context,
            capabilities,
            stats: Arc::new(Mutex::new(AllocationStats::default())),
            telemetry: Mutex::new(CudaTelemetry::default()),
        })
    }

    pub fn telemetry(&self) -> CudaTelemetry {
        self.telemetry.lock().expect("CUDA telemetry lock").clone()
    }

    pub fn live_bytes(&self) -> u64 {
        self.stats.lock().expect("CUDA stats lock").live_bytes
    }
    pub fn peak_bytes(&self) -> u64 {
        self.stats.lock().expect("CUDA stats lock").peak_bytes
    }
    pub fn live_tensors(&self) -> u64 {
        self.stats.lock().expect("CUDA stats lock").live_tensors
    }

    pub fn synchronize(&self) -> Result<(), String> {
        // SAFETY: context is owned by this backend.
        unsafe { check(vbuf_cuda_synchronize(self.context.as_ptr())) }
    }

    pub fn memory_info(&self) -> Result<(u64, u64), String> {
        let mut free = 0;
        let mut total = 0;
        // SAFETY: both values are valid writable output buffers.
        unsafe {
            check(vbuf_cuda_memory_info(
                self.context.as_ptr(),
                &mut free,
                &mut total,
            ))?;
        }
        Ok((free, total))
    }

    pub fn record_weight_bytes(&self, bytes: u64) {
        let mut telemetry = self.telemetry.lock().expect("CUDA telemetry lock");
        telemetry.device_weight_peak_bytes = telemetry.device_weight_peak_bytes.max(bytes);
    }

    fn record_cuda(&self) {
        let mut telemetry = self.telemetry.lock().expect("CUDA telemetry lock");
        telemetry.total_ops += 1;
        telemetry.cuda_ops += 1;
    }

    fn record_transfer(&self, host_to_device: bool, bytes: u64) {
        let mut telemetry = self.telemetry.lock().expect("CUDA telemetry lock");
        if host_to_device {
            telemetry.host_to_device_bytes += bytes;
        } else {
            telemetry.device_to_host_bytes += bytes;
        }
    }

    fn wrap(&self, raw: *mut CudaTensor, dimensions: Vec<u64>) -> Result<DeviceTensor, String> {
        let raw = NonNull::new(raw).ok_or("CUDA operation returned a null tensor")?;
        let elements = dimensions.iter().try_fold(1u64, |value, dimension| {
            value
                .checked_mul(*dimension)
                .ok_or("CUDA tensor shape overflows")
        })?;
        let bytes = elements
            .checked_mul(4)
            .ok_or("CUDA tensor bytes overflow")?;
        {
            let mut stats = self.stats.lock().expect("CUDA stats lock");
            stats.live_tensors += 1;
            stats.live_bytes += bytes;
            stats.peak_bytes = stats.peak_bytes.max(stats.live_bytes);
        }
        let storage = CudaStorageWithBytes {
            raw,
            stats: Arc::clone(&self.stats),
            bytes,
        };
        DeviceTensor::new(
            Arc::new(storage),
            self.capabilities.id,
            DeviceDType::F32,
            dimensions,
            bytes,
        )
    }

    fn unary_result(
        &self,
        status: c_int,
        raw: *mut CudaTensor,
        dimensions: Vec<u64>,
    ) -> Result<DeviceTensor, String> {
        check(status)?;
        self.wrap(raw, dimensions)
    }

    pub fn matmul(
        &self,
        input: &DeviceTensor,
        weight: &DeviceTensor,
    ) -> Result<DeviceTensor, String> {
        ensure_same(input, weight)?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: opaque handles originate from this adapter.
        let status = unsafe {
            vbuf_cuda_matmul(
                input.raw_handle().cast(),
                weight.raw_handle().cast(),
                &mut raw,
            )
        };
        let mut dimensions = input.dimensions().to_vec();
        dimensions.pop();
        dimensions.push(
            *weight
                .dimensions()
                .first()
                .ok_or("CUDA weight rank is invalid")?,
        );
        self.record_cuda();
        self.unary_result(status, raw, dimensions)
    }

    pub fn rms_norm(
        &self,
        input: &DeviceTensor,
        weight: &DeviceTensor,
        epsilon: f32,
    ) -> Result<DeviceTensor, String> {
        ensure_same(input, weight)?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            vbuf_cuda_rms_norm(
                input.raw_handle().cast(),
                weight.raw_handle().cast(),
                epsilon,
                &mut raw,
            )
        };
        self.record_cuda();
        self.unary_result(status, raw, input.dimensions().to_vec())
    }

    pub fn add(&self, left: &DeviceTensor, right: &DeviceTensor) -> Result<DeviceTensor, String> {
        self.binary(left, right, true)
    }
    pub fn mul(&self, left: &DeviceTensor, right: &DeviceTensor) -> Result<DeviceTensor, String> {
        self.binary(left, right, false)
    }

    fn binary(
        &self,
        left: &DeviceTensor,
        right: &DeviceTensor,
        add: bool,
    ) -> Result<DeviceTensor, String> {
        ensure_same(left, right)?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            if add {
                vbuf_cuda_add(
                    left.raw_handle().cast(),
                    right.raw_handle().cast(),
                    &mut raw,
                )
            } else {
                vbuf_cuda_mul(
                    left.raw_handle().cast(),
                    right.raw_handle().cast(),
                    &mut raw,
                )
            }
        };
        self.record_cuda();
        self.unary_result(status, raw, left.dimensions().to_vec())
    }

    pub fn bias_add(
        &self,
        input: &DeviceTensor,
        bias: &DeviceTensor,
    ) -> Result<DeviceTensor, String> {
        ensure_same(input, bias)?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            vbuf_cuda_bias_add(
                input.raw_handle().cast(),
                bias.raw_handle().cast(),
                &mut raw,
            )
        };
        self.record_cuda();
        self.unary_result(status, raw, input.dimensions().to_vec())
    }

    pub fn activation(
        &self,
        input: &DeviceTensor,
        kind: ActivationKind,
    ) -> Result<DeviceTensor, String> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            vbuf_cuda_activation(
                input.raw_handle().cast(),
                if kind == ActivationKind::Silu { 0 } else { 1 },
                &mut raw,
            )
        };
        self.record_cuda();
        self.unary_result(status, raw, input.dimensions().to_vec())
    }

    pub fn reshape(
        &self,
        input: &DeviceTensor,
        dimensions: &[u64],
    ) -> Result<DeviceTensor, String> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            vbuf_cuda_reshape(
                input.raw_handle().cast(),
                dimensions.len() as c_int,
                dimensions.as_ptr(),
                &mut raw,
            )
        };
        self.record_cuda();
        self.unary_result(status, raw, dimensions.to_vec())
    }

    pub fn rotary(
        &self,
        input: &DeviceTensor,
        attrs: crate::graph::RotaryAttributes,
    ) -> Result<DeviceTensor, String> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            vbuf_cuda_rotary(
                input.raw_handle().cast(),
                attrs.head_count,
                attrs.head_dim,
                attrs.rotary_dim,
                attrs.theta,
                attrs.position_start,
                &mut raw,
            )
        };
        self.record_cuda();
        self.unary_result(status, raw, input.dimensions().to_vec())
    }

    pub fn attention(
        &self,
        query: &DeviceTensor,
        key: &DeviceTensor,
        value: &DeviceTensor,
        attrs: crate::graph::AttentionAttributes,
    ) -> Result<DeviceTensor, String> {
        ensure_same(query, key)?;
        ensure_same(query, value)?;
        if attrs.mask != AttentionMaskKind::Causal || attrs.current_kv_length != attrs.query_length
        {
            return Err(
                "CUDA qualification backend supports only causal empty-state attention".into(),
            );
        }
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            vbuf_cuda_attention(
                query.raw_handle().cast(),
                key.raw_handle().cast(),
                value.raw_handle().cast(),
                attrs.query_head_count,
                attrs.kv_head_count,
                attrs.head_dim,
                attrs.scale,
                &mut raw,
            )
        };
        self.record_cuda();
        self.unary_result(status, raw, query.dimensions().to_vec())
    }

    pub fn expert_dispatch(
        &self,
        input: &DeviceTensor,
        gate: &DeviceTensor,
        up: &DeviceTensor,
        down: &DeviceTensor,
        selection: &GenericTopKSelection,
        expert: u32,
    ) -> Result<DeviceTensor, String> {
        ensure_same(input, gate)?;
        ensure_same(input, up)?;
        ensure_same(input, down)?;
        let tokens = input.dimensions().get(0).copied().unwrap_or(0)
            * input.dimensions().get(1).copied().unwrap_or(0);
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            vbuf_cuda_expert_dispatch(
                input.raw_handle().cast(),
                gate.raw_handle().cast(),
                up.raw_handle().cast(),
                down.raw_handle().cast(),
                selection.ids.as_ptr(),
                selection.weights.as_ptr(),
                tokens,
                selection.top_k as u64,
                expert,
                &mut raw,
            )
        };
        {
            let mut telemetry = self.telemetry.lock().expect("CUDA telemetry lock");
            telemetry.control_to_device_bytes += (selection.ids.len() * std::mem::size_of::<u32>()
                + selection.weights.len() * std::mem::size_of::<f32>())
                as u64;
        }
        self.record_cuda();
        self.unary_result(status, raw, input.dimensions().to_vec())
    }

    pub fn download_tracked(&self, tensor: &DeviceTensor) -> Result<Vec<f32>, String> {
        let count = tensor.elements()?;
        let mut output = vec![0.0; count];
        let status = unsafe {
            vbuf_cuda_tensor_download(
                tensor.raw_handle().cast(),
                output.as_mut_ptr(),
                count as u64,
            )
        };
        check(status)?;
        self.record_transfer(false, (count * 4) as u64);
        Ok(output)
    }
}

impl DeviceBackend for CudaBackend {
    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    fn allocate_f32(&self, dimensions: &[u64]) -> Result<DeviceTensor, String> {
        let values = vec![
            0.0f32;
            dimensions.iter().try_fold(1usize, |value, dimension| value
                .checked_mul(usize::try_from(*dimension).map_err(|_| "dimension too large")?)
                .ok_or_else(|| "allocation is too large".to_string()))?
        ];
        self.upload_f32(&values, dimensions)
    }

    fn upload_f32(&self, values: &[f32], dimensions: &[u64]) -> Result<DeviceTensor, String> {
        let count = dimensions.iter().try_fold(1u64, |value, dimension| {
            value
                .checked_mul(*dimension)
                .ok_or("tensor shape overflows")
        })?;
        if count as usize != values.len() {
            return Err("CUDA upload shape does not match values".into());
        }
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            vbuf_cuda_tensor_upload(
                self.context.as_ptr(),
                values.as_ptr(),
                count,
                dimensions.len() as c_int,
                dimensions.as_ptr(),
                &mut raw,
            )
        };
        check(status)?;
        self.record_transfer(true, count * 4);
        self.wrap(raw, dimensions.to_vec())
    }

    fn download_f32(&self, tensor: &DeviceTensor) -> Result<Vec<f32>, String> {
        self.download_tracked(tensor)
    }
}

#[derive(Debug)]
struct CudaStorageWithBytes {
    raw: NonNull<CudaTensor>,
    stats: Arc<Mutex<AllocationStats>>,
    bytes: u64,
}
unsafe impl Send for CudaStorageWithBytes {}
unsafe impl Sync for CudaStorageWithBytes {}
impl Drop for CudaStorageWithBytes {
    fn drop(&mut self) {
        unsafe {
            let _ = vbuf_cuda_tensor_destroy(self.raw.as_ptr());
        }
        if let Ok(mut stats) = self.stats.lock() {
            stats.live_tensors = stats.live_tensors.saturating_sub(1);
            stats.live_bytes = stats.live_bytes.saturating_sub(self.bytes);
        }
    }
}
impl DeviceStorage for CudaStorageWithBytes {
    fn raw_handle(&self) -> *mut c_void {
        self.raw.as_ptr().cast()
    }
    fn backend_name(&self) -> &'static str {
        "cuda"
    }
}

fn ensure_same(left: &DeviceTensor, right: &DeviceTensor) -> Result<(), String> {
    if left.device() != right.device()
        || left.dtype() != DeviceDType::F32
        || right.dtype() != DeviceDType::F32
    {
        return Err("CUDA operation combines incompatible device tensors".into());
    }
    if left.backend_name() != "cuda" || right.backend_name() != "cuda" {
        return Err("CUDA operation received a foreign backend tensor".into());
    }
    Ok(())
}

/// Executes a lowered graph with CUDA for all mathematically substantial work.
/// Top-K is deliberately host control: only its small scores/weights cross the
/// boundary after router MatMul has completed on CUDA.
pub fn execute_cuda_graph<T>(
    backend: &CudaBackend,
    graph: &ExecutionGraph,
    input: DeviceTensor,
    mut provider: T,
    selection: Option<&GenericTopKSelection>,
    attention_state_length: &mut u64,
) -> Result<CudaExecutionResult, String>
where
    T: FnMut(TensorId) -> Result<DeviceTensor, String>,
{
    let mut values = std::collections::HashMap::from([(graph.input, input)]);
    let mut result = CudaExecutionResult::default();
    for operation in &graph.operations {
        let value = |index: usize| -> Result<DeviceTensor, String> {
            match operation.inputs.get(index) {
                Some(InputRef::Value(id)) => values
                    .get(id)
                    .cloned()
                    .ok_or_else(|| format!("CUDA value {} is unavailable", id.0)),
                _ => Err("CUDA operation expected a value input".into()),
            }
        };
        let output = match operation.kind {
            OperationKind::RmsNorm => backend.rms_norm(
                &value(0)?,
                &provider(tensor_id(&operation.inputs, 1))?,
                operation
                    .attributes
                    .epsilon
                    .ok_or("RMSNorm epsilon missing")?,
            )?,
            OperationKind::MatMul | OperationKind::QuantizedMatMul => {
                backend.matmul(&value(0)?, &provider(tensor_id(&operation.inputs, 1))?)?
            }
            OperationKind::BiasAdd => {
                backend.bias_add(&value(0)?, &provider(tensor_id(&operation.inputs, 1))?)?
            }
            OperationKind::Activation => backend.activation(
                &value(0)?,
                operation
                    .attributes
                    .activation
                    .ok_or("activation kind missing")?,
            )?,
            OperationKind::ReshapeHeads => backend.reshape(
                &value(0)?,
                &reshape_dimensions(
                    &value(0)?,
                    operation
                        .attributes
                        .head_reshape
                        .ok_or("head reshape attributes missing")?,
                )?,
            )?,
            OperationKind::Rotary => backend.rotary(
                &value(0)?,
                operation
                    .attributes
                    .rotary
                    .ok_or("rotary attributes missing")?,
            )?,
            OperationKind::ResidualAdd | OperationKind::WeightedAdd => {
                backend.add(&value(0)?, &value(1)?)?
            }
            OperationKind::ElementwiseMul => backend.mul(&value(0)?, &value(1)?)?,
            OperationKind::ZeroLike => {
                let input = value(0)?;
                backend.allocate_f32(input.dimensions())?
            }
            OperationKind::Attention => {
                let attrs = operation
                    .attributes
                    .attention
                    .ok_or("attention attributes missing")?;
                if *attention_state_length != 0 {
                    return Err("CUDA attention state reuse is not in Step32K-A scope".into());
                }
                let output = backend.attention(&value(0)?, &value(1)?, &value(2)?, attrs)?;
                *attention_state_length = attrs.current_kv_length;
                output
            }
            OperationKind::TopKRouter => {
                let scores = value(0)?;
                let corrected = value(1)?;
                let scores_values = backend.download_tracked(&scores)?;
                let corrected_values = backend.download_tracked(&corrected)?;
                {
                    let mut telemetry = backend.telemetry.lock().expect("CUDA telemetry lock");
                    telemetry.host_control_ops += 1;
                    telemetry.control_to_host_bytes +=
                        ((scores_values.len() + corrected_values.len()) * 4) as u64;
                }
                result.selection = Some(select_top_k(
                    &scores,
                    &scores_values,
                    &corrected_values,
                    operation.attributes.top_k.ok_or("top-k missing")?,
                )?);
                scores
            }
            OperationKind::ExpertDispatch => {
                let selection = result
                    .selection
                    .as_ref()
                    .or(selection)
                    .ok_or("expert selection is missing")?;
                backend.expert_dispatch(
                    &value(0)?,
                    &provider(tensor_id(&operation.inputs, 1))?,
                    &provider(tensor_id(&operation.inputs, 2))?,
                    &provider(tensor_id(&operation.inputs, 3))?,
                    selection,
                    operation
                        .attributes
                        .expert_dispatch
                        .ok_or("expert identity missing")?
                        .expert_id,
                )?
            }
            _ => {
                return Err(format!(
                    "CUDA operation {:?} is unsupported",
                    operation.kind
                ));
            }
        };
        values.insert(operation.output, output);
        result
            .operation_ids
            .push((operation.id.clone(), operation.kind));
    }
    result.values = values;
    Ok(result)
}

#[derive(Clone, Debug, Default)]
pub struct CudaExecutionResult {
    pub values: std::collections::HashMap<ValueId, DeviceTensor>,
    pub selection: Option<GenericTopKSelection>,
    pub operation_ids: Vec<(String, OperationKind)>,
}

fn tensor_id(inputs: &[InputRef], index: usize) -> TensorId {
    match inputs.get(index) {
        Some(InputRef::Tensor(id)) => *id,
        _ => TensorId(u32::MAX),
    }
}

fn reshape_dimensions(
    input: &DeviceTensor,
    attrs: crate::graph::HeadReshapeAttributes,
) -> Result<Vec<u64>, String> {
    let expected = attrs
        .head_count
        .checked_mul(attrs.head_dim)
        .ok_or("head reshape overflows")?;
    if attrs.flatten {
        if input.dimensions().len() != 4
            || input.dimensions()[2] != attrs.head_count
            || input.dimensions()[3] != attrs.head_dim
        {
            return Err("CUDA head flatten geometry is invalid".into());
        }
        Ok(vec![input.dimensions()[0], input.dimensions()[1], expected])
    } else {
        if input.dimensions().len() != 3 || input.dimensions()[2] != expected {
            return Err("CUDA head reshape geometry is invalid".into());
        }
        Ok(vec![
            input.dimensions()[0],
            input.dimensions()[1],
            attrs.head_count,
            attrs.head_dim,
        ])
    }
}

fn select_top_k(
    scores: &DeviceTensor,
    scores_values: &[f32],
    corrected_values: &[f32],
    top_k: u32,
) -> Result<GenericTopKSelection, String> {
    let expert_count = *scores
        .dimensions()
        .last()
        .ok_or("router score rank is invalid")? as usize;
    if scores_values.len() != corrected_values.len()
        || expert_count == 0
        || top_k as usize > expert_count
    {
        return Err("CUDA router geometry is invalid".into());
    }
    let token_count = scores_values.len() / expert_count;
    let mut ids = Vec::with_capacity(token_count * top_k as usize);
    let mut weights = Vec::with_capacity(ids.capacity());
    for token in 0..token_count {
        let start = token * expert_count;
        let mut candidates: Vec<_> = (0..expert_count)
            .map(|index| (corrected_values[start + index], index))
            .collect();
        candidates.sort_by(|left, right| right.0.total_cmp(&left.0).then(left.1.cmp(&right.1)));
        let selected = &candidates[..top_k as usize];
        let denominator: f32 = selected
            .iter()
            .map(|(_, index)| scores_values[start + *index])
            .sum();
        for (_, index) in selected {
            ids.push(*index as u32);
            weights.push(scores_values[start + *index] / (denominator + 1e-20));
        }
    }
    Ok(GenericTopKSelection {
        ids,
        weights,
        token_count: token_count as u64,
        top_k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_uses_lower_index_ties() {
        let values = [1.0, 2.0, 2.0, 0.0];
        let corrected = [1.0, 2.0, 2.0, 0.0];
        let storage = crate::device::DeviceTensor::new(
            Arc::new(TestStorage),
            DeviceId(0),
            DeviceDType::F32,
            vec![1, 4],
            16,
        )
        .unwrap();
        let selection = select_top_k(&storage, &values, &corrected, 2).unwrap();
        assert_eq!(selection.ids, [1, 2]);
    }

    #[derive(Debug)]
    struct TestStorage;
    impl DeviceStorage for TestStorage {
        fn raw_handle(&self) -> *mut c_void {
            std::ptr::null_mut()
        }
        fn backend_name(&self) -> &'static str {
            "test"
        }
    }
}
