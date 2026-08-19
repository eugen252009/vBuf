//! Versioned, read-only C ABI for lowered portable execution graphs.
//!
//! This ABI exports semantic graph structure only. Payload resolution and its
//! lease stay outside this handle and are supplied by the vBuf-ML runtime.

use crate::graph::{
    ExecutionGraph, InputRef, MatMulWeightOperand, OperationKind, TopKOrder, TopKTieBreak,
};
use crate::lowering::{
    OperationAttributes, PortableInput, PortableOperation, PortableOperationKind, PortableProgram,
    PortableRegion, SemanticTensorKey, TensorBinding, lower_region,
};
use std::ffi::c_void;
use std::slice;

pub const VBUF_PORTABLE_EXEC_ABI_V1: u32 = 1;

pub const VBUF_FFI_OK: u32 = 0;
pub const VBUF_FFI_INVALID_ARGUMENT: u32 = 1;
pub const VBUF_FFI_INVALID_UTF8: u32 = 2;
pub const VBUF_FFI_INVALID_GRAPH: u32 = 3;
pub const VBUF_FFI_UNSUPPORTED: u32 = 4;
pub const VBUF_FFI_MISSING_BINDING: u32 = 5;
pub const VBUF_FFI_INVALID_ATTRIBUTES: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VbufFfiBytes {
    pub ptr: *const u8,
    pub len: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VbufPortableBindingDesc {
    pub semantic: VbufFfiBytes,
    pub tensor_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VbufPortableInputDesc {
    /// 0 = value, 1 = semantic tensor binding.
    pub kind: u8,
    pub _reserved: [u8; 3],
    pub id: u32,
    pub semantic: VbufFfiBytes,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VbufPortableOperationDesc {
    /// 1 = RmsNorm, 2 = MatMul, 3 = Activation, 4 = TopK.
    pub kind: u32,
    pub id: VbufFfiBytes,
    pub inputs: *const VbufPortableInputDesc,
    pub input_count: u32,
    pub output: u32,
    pub has_epsilon: u8,
    /// 0 = absent, 1 = lhs, 2 = rhs.
    pub matmul_weight_operand: u8,
    /// 0 = absent, 1 = not transposed, 2 = transposed.
    pub matmul_transpose_weight: u8,
    /// 0 = absent, 1 = descending.
    pub top_k_order: u8,
    /// 0 = absent, 1 = lower index.
    pub top_k_tie_break: u8,
    pub _reserved: [u8; 3],
    pub epsilon: f32,
    pub has_top_k: u8,
    pub _reserved_top_k: [u8; 3],
    pub top_k: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VbufPortableProgramDesc {
    pub bindings: *const VbufPortableBindingDesc,
    pub binding_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VbufPortableRegionDesc {
    pub input: u32,
    pub output: u32,
    pub operations: *const VbufPortableOperationDesc,
    pub operation_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VbufFfiError {
    pub code: u32,
    pub message: VbufFfiBytes,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VbufGraphTensorDesc {
    pub tensor_id: u32,
    pub semantic: VbufFfiBytes,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VbufGraphInputDesc {
    /// 0 = value, 1 = tensor.
    pub kind: u8,
    pub _reserved: [u8; 3],
    pub id: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VbufGraphOperationDesc {
    pub kind: u32,
    pub id: VbufFfiBytes,
    pub inputs: *const VbufGraphInputDesc,
    pub input_count: u32,
    pub output: u32,
    pub has_epsilon: u8,
    pub matmul_weight_operand: u8,
    pub matmul_transpose_weight: u8,
    pub top_k_order: u8,
    pub top_k_tie_break: u8,
    pub epsilon: f32,
    pub has_top_k: u8,
    pub _reserved: [u8; 3],
    pub top_k: u32,
}

pub struct VbufRuntimeGraphHandle {
    graph: ExecutionGraph,
    inputs: Vec<Vec<VbufGraphInputDesc>>,
}

fn bytes(value: VbufFfiBytes) -> Result<&'static [u8], u32> {
    if value.len == 0 {
        return Ok(&[]);
    }
    if value.ptr.is_null() || value.len > usize::MAX as u64 {
        return Err(VBUF_FFI_INVALID_ARGUMENT);
    }
    // SAFETY: The caller owns these immutable buffers for the duration of the ABI call.
    Ok(unsafe { slice::from_raw_parts(value.ptr, value.len as usize) })
}

fn text(value: VbufFfiBytes) -> Result<String, u32> {
    std::str::from_utf8(bytes(value)?)
        .map(str::to_owned)
        .map_err(|_| VBUF_FFI_INVALID_UTF8)
}

fn input(value: &VbufPortableInputDesc) -> Result<PortableInput, u32> {
    match value.kind {
        0 => Ok(PortableInput::Value(crate::ValueId(value.id))),
        1 => Ok(PortableInput::Tensor(SemanticTensorKey(text(
            value.semantic,
        )?))),
        _ => Err(VBUF_FFI_INVALID_GRAPH),
    }
}

fn operation(value: &VbufPortableOperationDesc) -> Result<PortableOperation, u32> {
    let kind = match value.kind {
        1 => PortableOperationKind::RmsNorm,
        2 => PortableOperationKind::MatMul,
        3 => PortableOperationKind::Activation,
        4 => PortableOperationKind::TopK,
        _ => return Err(VBUF_FFI_UNSUPPORTED),
    };
    if value.input_count != 0 && value.inputs.is_null() {
        return Err(VBUF_FFI_INVALID_ARGUMENT);
    }
    let inputs = if value.input_count == 0 {
        Vec::new()
    } else {
        // SAFETY: validated pointer/count are owned by the caller for this call.
        unsafe { slice::from_raw_parts(value.inputs, value.input_count as usize) }
            .iter()
            .map(input)
            .collect::<Result<Vec<_>, _>>()?
    };
    let matmul_weight_operand = match value.matmul_weight_operand {
        0 => None,
        1 => Some(MatMulWeightOperand::Lhs),
        2 => Some(MatMulWeightOperand::Rhs),
        _ => return Err(VBUF_FFI_INVALID_GRAPH),
    };
    let top_k_order = match value.top_k_order {
        0 => None,
        1 => Some(TopKOrder::Descending),
        _ => return Err(VBUF_FFI_INVALID_GRAPH),
    };
    let top_k_tie_break = match value.top_k_tie_break {
        0 => None,
        1 => Some(TopKTieBreak::LowerIndex),
        _ => return Err(VBUF_FFI_INVALID_GRAPH),
    };
    Ok(PortableOperation {
        id: text(value.id)?,
        kind,
        inputs,
        output: crate::ValueId(value.output),
        attributes: OperationAttributes {
            epsilon: (value.has_epsilon != 0).then_some(value.epsilon),
            top_k: (value.has_top_k != 0).then_some(value.top_k),
            matmul_weight_operand,
            matmul_transpose_weight: match value.matmul_transpose_weight {
                0 => None,
                1 => Some(false),
                2 => Some(true),
                _ => return Err(VBUF_FFI_INVALID_GRAPH),
            },
            top_k_order,
            top_k_tie_break,
            ..Default::default()
        },
    })
}

fn lower(
    program: &VbufPortableProgramDesc,
    region: &VbufPortableRegionDesc,
) -> Result<VbufRuntimeGraphHandle, u32> {
    if program.binding_count != 0 && program.bindings.is_null() {
        return Err(VBUF_FFI_INVALID_ARGUMENT);
    }
    if region.operation_count != 0 && region.operations.is_null() {
        return Err(VBUF_FFI_INVALID_ARGUMENT);
    }
    let bindings = if program.binding_count == 0 {
        Vec::new()
    } else {
        // SAFETY: validated pointer/count are owned by the caller for this call.
        unsafe { slice::from_raw_parts(program.bindings, program.binding_count as usize) }
            .iter()
            .map(|binding| {
                Ok(TensorBinding {
                    semantic: SemanticTensorKey(text(binding.semantic)?),
                    tensor: crate::TensorId(binding.tensor_id),
                })
            })
            .collect::<Result<Vec<_>, u32>>()?
    };
    let operations = if region.operation_count == 0 {
        Vec::new()
    } else {
        // SAFETY: validated pointer/count are owned by the caller for this call.
        unsafe { slice::from_raw_parts(region.operations, region.operation_count as usize) }
            .iter()
            .map(operation)
            .collect::<Result<Vec<_>, _>>()?
    };
    let graph = lower_region(
        &PortableProgram {
            tensor_bindings: bindings,
            state_refs: Vec::new(),
        },
        &PortableRegion {
            id: "ffi.region".to_owned(),
            input: crate::ValueId(region.input),
            output: crate::ValueId(region.output),
            operations,
        },
    )
    .map_err(|error| match error {
        crate::lowering::LoweringError::MissingBinding(_) => VBUF_FFI_MISSING_BINDING,
        crate::lowering::LoweringError::InvalidAttribute(_) => VBUF_FFI_INVALID_ATTRIBUTES,
        crate::lowering::LoweringError::UnsupportedOperation(_) => VBUF_FFI_UNSUPPORTED,
        crate::lowering::LoweringError::DuplicateBinding(_) => VBUF_FFI_INVALID_GRAPH,
    })?;
    let inputs = graph
        .operations
        .iter()
        .map(|operation| {
            operation
                .inputs
                .iter()
                .map(|input| match input {
                    InputRef::Value(value) => VbufGraphInputDesc {
                        kind: 0,
                        _reserved: [0; 3],
                        id: value.0,
                    },
                    InputRef::Tensor(tensor) => VbufGraphInputDesc {
                        kind: 1,
                        _reserved: [0; 3],
                        id: tensor.0,
                    },
                })
                .collect()
        })
        .collect();
    Ok(VbufRuntimeGraphHandle { graph, inputs })
}

fn set_error(error: *mut VbufFfiError, code: u32, message: &'static [u8]) {
    if !error.is_null() {
        unsafe {
            *error = VbufFfiError {
                code,
                message: VbufFfiBytes {
                    ptr: message.as_ptr(),
                    len: message.len() as u64,
                },
            };
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_runtime_graph_lower_v1(
    program: *const VbufPortableProgramDesc,
    region: *const VbufPortableRegionDesc,
    output: *mut *mut VbufRuntimeGraphHandle,
    error: *mut VbufFfiError,
) -> u32 {
    if !output.is_null() {
        unsafe {
            *output = std::ptr::null_mut();
        }
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if program.is_null() || region.is_null() || output.is_null() {
            return Err(VBUF_FFI_INVALID_ARGUMENT);
        }
        // SAFETY: pointers are checked above and valid for the duration of this call.
        let handle = lower(unsafe { &*program }, unsafe { &*region })?;
        unsafe {
            *output = Box::into_raw(Box::new(handle));
        }
        Ok(())
    }))
    .unwrap_or(Err(VBUF_FFI_INVALID_GRAPH));
    match result {
        Ok(()) => {
            set_error(error, VBUF_FFI_OK, b"ok");
            VBUF_FFI_OK
        }
        Err(code) => {
            let message = match code {
                VBUF_FFI_INVALID_ARGUMENT => b"invalid argument".as_slice(),
                VBUF_FFI_INVALID_UTF8 => b"invalid utf8".as_slice(),
                VBUF_FFI_UNSUPPORTED => b"unsupported operation".as_slice(),
                VBUF_FFI_MISSING_BINDING => b"missing tensor binding".as_slice(),
                VBUF_FFI_INVALID_ATTRIBUTES => b"invalid operation attributes".as_slice(),
                _ => b"invalid graph".as_slice(),
            };
            set_error(error, code, message);
            code
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_runtime_graph_close(handle: *mut VbufRuntimeGraphHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_runtime_graph_abi_version() -> u32 {
    VBUF_PORTABLE_EXEC_ABI_V1
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_runtime_graph_operation_count(
    handle: *const VbufRuntimeGraphHandle,
) -> u32 {
    if handle.is_null() {
        0
    } else {
        unsafe { (*handle).graph.operations.len() as u32 }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_runtime_graph_tensor_count(
    handle: *const VbufRuntimeGraphHandle,
) -> u32 {
    if handle.is_null() {
        0
    } else {
        unsafe { (*handle).graph.tensors.len() as u32 }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_runtime_graph_input_value(
    handle: *const VbufRuntimeGraphHandle,
    output: *mut u32,
) -> u32 {
    if handle.is_null() || output.is_null() {
        return VBUF_FFI_INVALID_ARGUMENT;
    }
    unsafe {
        *output = (*handle).graph.input.0;
    }
    VBUF_FFI_OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_runtime_graph_output_value(
    handle: *const VbufRuntimeGraphHandle,
    output: *mut u32,
) -> u32 {
    if handle.is_null() || output.is_null() {
        return VBUF_FFI_INVALID_ARGUMENT;
    }
    unsafe {
        *output = (*handle).graph.output.0;
    }
    VBUF_FFI_OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_runtime_graph_tensor_desc(
    handle: *const VbufRuntimeGraphHandle,
    index: u32,
    output: *mut VbufGraphTensorDesc,
) -> u32 {
    if handle.is_null() || output.is_null() {
        return VBUF_FFI_INVALID_ARGUMENT;
    }
    let Some(tensor) = (unsafe { &*handle }).graph.tensors.get(index as usize) else {
        return VBUF_FFI_INVALID_ARGUMENT;
    };
    unsafe {
        *output = VbufGraphTensorDesc {
            tensor_id: tensor.id.0,
            semantic: VbufFfiBytes {
                ptr: tensor.name.as_ptr(),
                len: tensor.name.len() as u64,
            },
        };
    }
    VBUF_FFI_OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_runtime_graph_operation_desc(
    handle: *const VbufRuntimeGraphHandle,
    index: u32,
    output: *mut VbufGraphOperationDesc,
) -> u32 {
    if handle.is_null() || output.is_null() {
        return VBUF_FFI_INVALID_ARGUMENT;
    }
    let handle = unsafe { &*handle };
    let Some(operation) = handle.graph.operations.get(index as usize) else {
        return VBUF_FFI_INVALID_ARGUMENT;
    };
    let kind = match operation.kind {
        OperationKind::RmsNorm => 1,
        OperationKind::MatMul => 2,
        OperationKind::TopKRouter => 4,
        _ => 0,
    };
    if kind == 0 {
        return VBUF_FFI_UNSUPPORTED;
    }
    let attrs = operation.attributes;
    unsafe {
        *output = VbufGraphOperationDesc {
            kind,
            id: VbufFfiBytes {
                ptr: operation.id.as_ptr(),
                len: operation.id.len() as u64,
            },
            inputs: handle.inputs[index as usize].as_ptr(),
            input_count: handle.inputs[index as usize].len() as u32,
            output: operation.output.0,
            has_epsilon: attrs.epsilon.is_some() as u8,
            matmul_weight_operand: match attrs.matmul_weight_operand {
                Some(MatMulWeightOperand::Lhs) => 1,
                Some(MatMulWeightOperand::Rhs) => 2,
                None => 0,
            },
            matmul_transpose_weight: match attrs.matmul_transpose_weight {
                Some(false) => 1,
                Some(true) => 2,
                None => 0,
            },
            top_k_order: match attrs.top_k_order {
                Some(TopKOrder::Descending) => 1,
                None => 0,
            },
            top_k_tie_break: match attrs.top_k_tie_break {
                Some(TopKTieBreak::LowerIndex) => 1,
                None => 0,
            },
            epsilon: attrs.epsilon.unwrap_or(0.0),
            has_top_k: attrs.top_k.is_some() as u8,
            _reserved: [0; 3],
            top_k: attrs.top_k.unwrap_or(0),
        };
    }
    VBUF_FFI_OK
}

#[allow(dead_code)]
fn _opaque_marker(_: *const c_void) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn abi_layout_is_explicitly_checked() {
        assert_eq!(size_of::<VbufPortableOperationDesc>(), 64);
        assert_eq!(size_of::<VbufGraphOperationDesc>(), 64);
        assert_eq!(size_of::<VbufPortableInputDesc>(), 24);
    }

    #[test]
    fn abi_preserves_selected_operation_attributes() {
        let norm_name = b"norm";
        let router_name = b"router";
        let norm_input = VbufPortableInputDesc {
            kind: 0,
            _reserved: [0; 3],
            id: 0,
            semantic: VbufFfiBytes {
                ptr: std::ptr::null(),
                len: 0,
            },
        };
        let norm_weight = VbufPortableInputDesc {
            kind: 1,
            _reserved: [0; 3],
            id: 0,
            semantic: VbufFfiBytes {
                ptr: norm_name.as_ptr(),
                len: norm_name.len() as u64,
            },
        };
        let router_input = VbufPortableInputDesc {
            kind: 0,
            _reserved: [0; 3],
            id: 1,
            semantic: VbufFfiBytes {
                ptr: std::ptr::null(),
                len: 0,
            },
        };
        let router_weight = VbufPortableInputDesc {
            kind: 1,
            _reserved: [0; 3],
            id: 0,
            semantic: VbufFfiBytes {
                ptr: router_name.as_ptr(),
                len: router_name.len() as u64,
            },
        };
        let topk_input = VbufPortableInputDesc {
            kind: 0,
            _reserved: [0; 3],
            id: 2,
            semantic: VbufFfiBytes {
                ptr: std::ptr::null(),
                len: 0,
            },
        };
        let norm_inputs = [norm_input, norm_weight];
        let router_inputs = [router_input, router_weight];
        let norm_op = VbufPortableOperationDesc {
            kind: 1,
            id: VbufFfiBytes {
                ptr: b"norm".as_ptr(),
                len: 4,
            },
            inputs: norm_inputs.as_ptr(),
            input_count: 2,
            output: 1,
            has_epsilon: 1,
            matmul_weight_operand: 0,
            matmul_transpose_weight: 0,
            top_k_order: 0,
            top_k_tie_break: 0,
            _reserved: [0; 3],
            epsilon: 1e-6,
            has_top_k: 0,
            _reserved_top_k: [0; 3],
            top_k: 0,
        };
        let matmul_op = VbufPortableOperationDesc {
            kind: 2,
            id: VbufFfiBytes {
                ptr: b"router".as_ptr(),
                len: 6,
            },
            inputs: router_inputs.as_ptr(),
            input_count: 2,
            output: 2,
            has_epsilon: 0,
            matmul_weight_operand: 2,
            matmul_transpose_weight: 1,
            top_k_order: 0,
            top_k_tie_break: 0,
            _reserved: [0; 3],
            epsilon: 0.0,
            has_top_k: 0,
            _reserved_top_k: [0; 3],
            top_k: 0,
        };
        let topk_op = VbufPortableOperationDesc {
            kind: 4,
            id: VbufFfiBytes {
                ptr: b"select".as_ptr(),
                len: 6,
            },
            inputs: &topk_input,
            input_count: 1,
            output: 3,
            has_epsilon: 0,
            matmul_weight_operand: 0,
            matmul_transpose_weight: 0,
            top_k_order: 1,
            top_k_tie_break: 1,
            _reserved: [0; 3],
            epsilon: 0.0,
            has_top_k: 1,
            _reserved_top_k: [0; 3],
            top_k: 6,
        };
        let bindings = [
            VbufPortableBindingDesc {
                semantic: VbufFfiBytes {
                    ptr: norm_name.as_ptr(),
                    len: norm_name.len() as u64,
                },
                tensor_id: 10,
            },
            VbufPortableBindingDesc {
                semantic: VbufFfiBytes {
                    ptr: router_name.as_ptr(),
                    len: router_name.len() as u64,
                },
                tensor_id: 11,
            },
        ];
        let mut operations = [norm_op, matmul_op, topk_op];
        let program = VbufPortableProgramDesc {
            bindings: bindings.as_ptr(),
            binding_count: 2,
        };
        let region = VbufPortableRegionDesc {
            input: 0,
            output: 3,
            operations: operations.as_ptr(),
            operation_count: 3,
        };
        let mut handle = std::ptr::null_mut();
        let mut error = VbufFfiError {
            code: 99,
            message: VbufFfiBytes {
                ptr: std::ptr::null(),
                len: 0,
            },
        };
        assert_eq!(
            unsafe { vbuf_runtime_graph_lower_v1(&program, &region, &mut handle, &mut error) },
            VBUF_FFI_OK
        );
        assert_eq!(
            unsafe { (&(*handle).graph.operations)[0].attributes.epsilon },
            Some(1e-6)
        );
        assert_eq!(
            unsafe {
                (&(*handle).graph.operations)[1]
                    .attributes
                    .matmul_weight_operand
            },
            Some(MatMulWeightOperand::Rhs)
        );
        assert_eq!(
            unsafe { (&(*handle).graph.operations)[2].attributes.top_k },
            Some(6)
        );
        unsafe {
            vbuf_runtime_graph_close(handle);
        }

        operations[2].has_top_k = 0;
        let invalid_attributes_region = VbufPortableRegionDesc {
            input: 0,
            output: 3,
            operations: operations.as_ptr(),
            operation_count: 3,
        };
        let mut invalid_attributes_handle = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                vbuf_runtime_graph_lower_v1(
                    &program,
                    &invalid_attributes_region,
                    &mut invalid_attributes_handle,
                    &mut error,
                )
            },
            VBUF_FFI_INVALID_ATTRIBUTES
        );
        assert!(invalid_attributes_handle.is_null());

        let missing_program = VbufPortableProgramDesc {
            bindings: bindings.as_ptr(),
            binding_count: 1,
        };
        let mut missing_handle = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                vbuf_runtime_graph_lower_v1(
                    &missing_program,
                    &region,
                    &mut missing_handle,
                    &mut error,
                )
            },
            VBUF_FFI_MISSING_BINDING
        );
        assert!(missing_handle.is_null());

        let unknown = VbufPortableOperationDesc {
            kind: 99,
            id: VbufFfiBytes {
                ptr: b"unknown".as_ptr(),
                len: 7,
            },
            inputs: std::ptr::null(),
            input_count: 0,
            output: 3,
            has_epsilon: 0,
            matmul_weight_operand: 0,
            matmul_transpose_weight: 0,
            top_k_order: 0,
            top_k_tie_break: 0,
            _reserved: [0; 3],
            epsilon: 0.0,
            has_top_k: 0,
            _reserved_top_k: [0; 3],
            top_k: 0,
        };
        let unknown_region = VbufPortableRegionDesc {
            input: 0,
            output: 3,
            operations: &unknown,
            operation_count: 1,
        };
        let mut unknown_handle = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                vbuf_runtime_graph_lower_v1(
                    &program,
                    &unknown_region,
                    &mut unknown_handle,
                    &mut error,
                )
            },
            VBUF_FFI_UNSUPPORTED
        );
        assert!(unknown_handle.is_null());
    }
}
