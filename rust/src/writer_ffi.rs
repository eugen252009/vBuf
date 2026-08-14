use crate::{v06, writer};
use std::ffi::{CStr, c_char};
use std::fs::File;

/// Opaque C handle for the canonical v0.6 writer.
pub struct V06CWriter {
    writer: Option<writer::VBufV06Writer<File>>,
}

/// # Safety
/// `path` must be a valid C string and `error_out`, when non-null, writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_v06_writer_create(
    path: *const c_char,
    base_shift: u8,
    indefinite: bool,
    error_out: *mut u32,
) -> *mut V06CWriter {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if !error_out.is_null() {
            *error_out = 0;
        }
        if path.is_null() {
            return std::ptr::null_mut();
        }
        let path = match CStr::from_ptr(path).to_str() {
            Ok(path) => path,
            Err(_) => return std::ptr::null_mut(),
        };
        let file = match std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)
        {
            Ok(file) => file,
            Err(_) => return std::ptr::null_mut(),
        };
        let writer = if indefinite {
            writer::VBufV06Writer::new_indefinite(file, base_shift)
        } else {
            writer::VBufV06Writer::new_known_size(file, base_shift)
        };
        match writer {
            Ok(writer) => Box::into_raw(Box::new(V06CWriter {
                writer: Some(writer),
            })),
            Err(_) => std::ptr::null_mut(),
        }
    }));
    match result {
        Ok(handle) if !handle.is_null() => handle,
        _ => {
            unsafe {
                if !error_out.is_null() {
                    *error_out = 1;
                }
            }
            std::ptr::null_mut()
        }
    }
}

unsafe fn ffi_values<T: Copy>(data: *const std::ffi::c_void, count: usize) -> Result<Vec<T>, ()> {
    if count > 0 && data.is_null() {
        return Err(());
    }
    let width = std::mem::size_of::<T>();
    if width == 0
        || count
            .checked_mul(width)
            .filter(|byte_len| *byte_len <= isize::MAX as usize)
            .is_none()
    {
        return Err(());
    }
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| ())?;
    for index in 0..count {
        // SAFETY: the FFI caller guarantees `count` readable T-sized values;
        // read_unaligned avoids imposing an undocumented C pointer alignment.
        values.push(unsafe { std::ptr::read_unaligned((data as *const T).add(index)) });
    }
    Ok(values)
}

/// Writes one canonical portable primitive block.
///
/// # Safety
/// `handle` must be live. `data` must contain `count` readable values of the
/// selected C type code. The handle must not be used after finish.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_v06_writer_write(
    handle: *mut V06CWriter,
    key_id: u16,
    physical: u8,
    continuation: bool,
    payload_shift: u8,
    value_type: u8,
    data: *const std::ffi::c_void,
    count: libc::size_t,
) -> u32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let Some(handle) = handle.as_mut() else {
            return Err(());
        };
        let Some(writer) = handle.writer.as_mut() else {
            return Err(());
        };
        let physical = match physical {
            0 => v06::V06Physical::Scalar,
            1 => v06::V06Physical::Array,
            _ => return Err(()),
        };
        let options = writer::BlockOptions {
            key_id,
            physical,
            continuation,
            payload_shift,
        };
        let result = match value_type {
            0 => writer.write_u8(options, &ffi_values::<u8>(data, count)?),
            1 => writer.write_u16(options, &ffi_values::<u16>(data, count)?),
            2 => writer.write_u32(options, &ffi_values::<u32>(data, count)?),
            3 => writer.write_u64(options, &ffi_values::<u64>(data, count)?),
            4 => writer.write_i8(options, &ffi_values::<i8>(data, count)?),
            5 => writer.write_i16(options, &ffi_values::<i16>(data, count)?),
            6 => writer.write_i32(options, &ffi_values::<i32>(data, count)?),
            7 => writer.write_i64(options, &ffi_values::<i64>(data, count)?),
            8 => writer.write_f32(options, &ffi_values::<f32>(data, count)?),
            9 => writer.write_f64(options, &ffi_values::<f64>(data, count)?),
            10 => writer.write_opaque(options, &ffi_values::<u8>(data, count)?),
            _ => return Err(()),
        };
        result.map_err(|_| ())
    }));
    match result {
        Ok(Ok(())) => 0,
        _ => 1,
    }
}

/// Finishes and destroys a canonical v0.6 writer handle.
///
/// # Safety
/// `handle` must be a unique live handle returned by create and is consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbuf_v06_writer_finish(handle: *mut V06CWriter) -> u32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        if handle.is_null() {
            return Err(());
        }
        let mut handle = Box::from_raw(handle);
        let writer = handle.writer.take().ok_or(())?;
        writer.finish().map(|_| ()).map_err(|_| ())
    }));
    match result {
        Ok(Ok(())) => 0,
        _ => 1,
    }
}
