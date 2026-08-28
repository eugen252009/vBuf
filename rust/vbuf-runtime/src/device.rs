//! Backend-neutral device execution contracts.
//!
//! This module contains only logical device identity, tensor metadata, and an
//! opaque backend-owned storage handle. Concrete runtime/library types belong
//! in the backend adapter which implements the storage trait.

use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeviceId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceKind {
    Host,
    Accelerator,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceDType {
    F32,
    F16,
    BF16,
    F8E4M3,
}

impl DeviceDType {
    pub const fn byte_width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F8E4M3 => 1,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceCapabilities {
    pub id: DeviceId,
    pub kind: DeviceKind,
    pub name: String,
    pub compute_capability: Option<(u32, u32)>,
    pub total_bytes: u64,
    pub free_bytes: u64,
}

pub trait DeviceStorage: Send + Sync + fmt::Debug {
    /// Returns an opaque backend handle. Portable code must never interpret it.
    fn raw_handle(&self) -> *mut c_void;
    fn backend_name(&self) -> &'static str;
}

#[derive(Clone, Debug)]
pub struct DeviceTensor {
    storage: Arc<dyn DeviceStorage>,
    device: DeviceId,
    dtype: DeviceDType,
    dimensions: Vec<u64>,
    bytes: u64,
}

impl DeviceTensor {
    pub fn new(
        storage: Arc<dyn DeviceStorage>,
        device: DeviceId,
        dtype: DeviceDType,
        dimensions: Vec<u64>,
        bytes: u64,
    ) -> Result<Self, String> {
        if dimensions.is_empty() || dimensions.iter().any(|dimension| *dimension == 0) {
            return Err("device tensor dimensions are invalid".into());
        }
        let elements = dimensions.iter().try_fold(1u64, |value, dimension| {
            value
                .checked_mul(*dimension)
                .ok_or("device tensor element count overflows")
        })?;
        let expected = elements
            .checked_mul(dtype.byte_width() as u64)
            .ok_or("device tensor byte count overflows")?;
        if expected != bytes {
            return Err("device tensor byte count does not match shape".into());
        }
        Ok(Self {
            storage,
            device,
            dtype,
            dimensions,
            bytes,
        })
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn dtype(&self) -> DeviceDType {
        self.dtype
    }

    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn elements(&self) -> Result<usize, String> {
        self.dimensions.iter().try_fold(1usize, |value, dimension| {
            value
                .checked_mul(
                    usize::try_from(*dimension).map_err(|_| "device dimension is too large")?,
                )
                .ok_or_else(|| "device tensor element count is too large".to_string())
        })
    }

    #[allow(dead_code)]
    pub(crate) fn raw_handle(&self) -> *mut c_void {
        self.storage.raw_handle()
    }

    #[allow(dead_code)]
    pub(crate) fn backend_name(&self) -> &'static str {
        self.storage.backend_name()
    }
}

pub trait DeviceBackend {
    fn capabilities(&self) -> &DeviceCapabilities;
    fn allocate_f32(&self, dimensions: &[u64]) -> Result<DeviceTensor, String>;
    fn upload_f32(&self, values: &[f32], dimensions: &[u64]) -> Result<DeviceTensor, String>;
    fn download_f32(&self, tensor: &DeviceTensor) -> Result<Vec<f32>, String>;
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn device_tensor_contract_validates_shape_and_bytes() {
        let storage: Arc<dyn DeviceStorage> = Arc::new(TestStorage);
        let tensor =
            DeviceTensor::new(storage, DeviceId(0), DeviceDType::F32, vec![2, 3], 24).unwrap();
        assert_eq!(tensor.elements().unwrap(), 6);
        assert_eq!(tensor.bytes(), 24);
        assert_eq!(tensor.backend_name(), "test");
    }

    #[test]
    fn device_tensor_contract_rejects_mismatched_bytes() {
        let storage: Arc<dyn DeviceStorage> = Arc::new(TestStorage);
        assert!(DeviceTensor::new(storage, DeviceId(0), DeviceDType::F32, vec![2, 3], 12).is_err());
    }

    #[derive(Debug)]
    struct FailingBackend;

    impl DeviceBackend for FailingBackend {
        fn capabilities(&self) -> &DeviceCapabilities {
            static CAPABILITIES: DeviceCapabilities = DeviceCapabilities {
                id: DeviceId(9),
                kind: DeviceKind::Accelerator,
                name: String::new(),
                compute_capability: None,
                total_bytes: 0,
                free_bytes: 0,
            };
            &CAPABILITIES
        }

        fn allocate_f32(&self, _dimensions: &[u64]) -> Result<DeviceTensor, String> {
            Err("mock device out of memory".into())
        }

        fn upload_f32(&self, _values: &[f32], _dimensions: &[u64]) -> Result<DeviceTensor, String> {
            Err("mock device out of memory".into())
        }

        fn download_f32(&self, _tensor: &DeviceTensor) -> Result<Vec<f32>, String> {
            Err("mock device has no tensor".into())
        }
    }

    #[test]
    fn mocked_device_oom_fails_without_creating_ownership() {
        let backend = FailingBackend;
        assert!(backend.allocate_f32(&[1024, 1024]).is_err());
    }
}
