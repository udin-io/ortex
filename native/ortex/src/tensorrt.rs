//! TensorRT's engine cache, and the versions an engine is only valid for.
//!
//! ONNX Runtime's TensorRT provider builds an engine for the card it runs on,
//! which takes minutes, and with `trt_engine_cache_enable` it writes the engine
//! to `trt_engine_cache_path` and reads it back on the next session. It reuses
//! a cached engine only by path and prefix: it does not notice a new TensorRT,
//! driver or ONNX Runtime by itself. So a caller keys the cache directory on
//! those versions, and `runtime_version/0` and `tensorrt_version/0` report the
//! ones this process loaded.

/// The TensorRT options a load asks for. `None` leaves an option at ONNX
/// Runtime's default (no engine cache, no timing cache).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TensorRt {
    /// Where TensorRT writes and reads its engines.
    pub cache_dir: Option<String>,
    /// Where TensorRT keeps the timing cache that speeds up later builds.
    pub timing_cache_dir: Option<String>,
    /// Build the engine in half precision.
    pub fp16: bool,
    /// The smallest, usual and largest input shapes the engine takes, each in
    /// ONNX Runtime's `input:1x3x256x192` form, inputs comma separated. ONNX
    /// Runtime builds one engine for the whole range, and needs the same three
    /// again on each load of a cached engine, or it rebuilds the engine.
    pub profile: Option<Profile>,
}

/// An explicit TensorRT optimization profile: `trt_profile_min_shapes`,
/// `trt_profile_opt_shapes` and `trt_profile_max_shapes`. ONNX Runtime uses
/// them only when all three are given, so they travel together.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Profile {
    pub min: String,
    pub opt: String,
    pub max: String,
}

impl TensorRt {
    /// The provider options ONNX Runtime's TensorRT provider takes, by the
    /// names `ort`'s `with_engine_cache`, `with_engine_cache_path`,
    /// `with_timing_cache`, `with_timing_cache_path` and `with_fp16` set.
    ///
    /// Refuses a path that holds a NUL byte: `ort` turns each value into a C
    /// string and panics on one.
    pub fn options(&self) -> Result<Vec<(&'static str, String)>, String> {
        let mut options = Vec::new();

        if let Some(dir) = &self.cache_dir {
            options.push(("trt_engine_cache_enable", "1".to_string()));
            options.push(("trt_engine_cache_path", path(dir, "cache_dir")?));
        }

        if let Some(dir) = &self.timing_cache_dir {
            options.push(("trt_timing_cache_enable", "1".to_string()));
            options.push(("trt_timing_cache_path", path(dir, "timing_cache_dir")?));
        }

        options.push((
            "trt_fp16_enable",
            if self.fp16 { "1" } else { "0" }.to_string(),
        ));

        if let Some(profile) = &self.profile {
            options.push(("trt_profile_min_shapes", shapes(&profile.min, "min")?));
            options.push(("trt_profile_opt_shapes", shapes(&profile.opt, "opt")?));
            options.push(("trt_profile_max_shapes", shapes(&profile.max, "max")?));
        }

        Ok(options)
    }
}

fn shapes(value: &str, name: &str) -> Result<String, String> {
    if value.is_empty() || value.contains('\0') {
        Err(format!(
            "the TensorRT profile's {name} shapes must be non-empty with no NUL byte"
        ))
    } else {
        Ok(value.to_string())
    }
}

fn path(dir: &str, name: &str) -> Result<String, String> {
    if dir.is_empty() || dir.contains('\0') {
        Err(format!(
            "the TensorRT option {name} must be a non-empty path with no NUL byte"
        ))
    } else {
        Ok(dir.to_string())
    }
}

/// Whether a session for these providers is built while every other GPU run
/// waits (`crate::gate::alone`).
///
/// A CUDA session's build allocates and copies on the GPU, the work that breaks
/// another session's first-run algorithm search (Eagle issue 1376), and it
/// takes a second or two. A TensorRT session's build can take minutes, since
/// that is when TensorRT builds its engine; holding every GPU session that long
/// stalls every camera. So a TensorRT build runs beside the other sessions, and
/// the caller builds one TensorRT engine at a time instead. Its runs still go
/// through the session's first-run gate.
pub fn builds_alone(eps: &[&str]) -> bool {
    use crate::constants::{CUDA, TENSORRT};
    eps.contains(&CUDA) && !eps.contains(&TENSORRT)
}

/// Whether a session for these providers takes the first-run gate on its runs.
pub fn gated(eps: &[&str]) -> bool {
    use crate::constants::{CUDA, TENSORRT};
    eps.contains(&CUDA) || eps.contains(&TENSORRT)
}

/// The version of the ONNX Runtime library this process loaded, as its own
/// `GetVersionString` reports it ("1.19.2").
pub fn runtime_version() -> Result<String, String> {
    let base = unsafe { ort::sys::OrtGetApiBase() };
    if base.is_null() {
        return Err("OrtGetApiBase returned null".to_string());
    }
    let get = unsafe { (*base).GetVersionString }.ok_or("GetVersionString is null")?;
    let version = unsafe { std::ffi::CStr::from_ptr(get()) };
    Ok(version.to_string_lossy().into_owned())
}

/// The version of the TensorRT library (`libnvinfer.so.10`) this process uses,
/// from its own `getInferLibVersion`, or `None` when it cannot be loaded.
///
/// ONNX Runtime's TensorRT provider links `libnvinfer.so.10` by that name. The
/// dynamic loader keeps one copy per name, so the library opened here is the
/// one the provider runs on, whichever of the two opens it first.
#[cfg(all(target_os = "linux", feature = "tensorrt"))]
pub fn tensorrt_version() -> Option<String> {
    use std::os::raw::{c_char, c_int, c_void};

    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    const RTLD_LAZY: c_int = 0x1;
    const RTLD_NOLOAD: c_int = 0x4;
    let name = b"libnvinfer.so.10\0".as_ptr() as *const c_char;

    // The handle is never closed: the library stays loaded for the provider.
    let handle = unsafe {
        let loaded = dlopen(name, RTLD_LAZY | RTLD_NOLOAD);
        if loaded.is_null() {
            dlopen(name, RTLD_LAZY)
        } else {
            loaded
        }
    };
    if handle.is_null() {
        return None;
    }

    let symbol = unsafe { dlsym(handle, b"getInferLibVersion\0".as_ptr() as *const c_char) };
    if symbol.is_null() {
        return None;
    }

    let get: extern "C" fn() -> i32 = unsafe { std::mem::transmute(symbol) };
    Some(decode_version(get()))
}

/// No TensorRT build on this platform, or no `tensorrt` feature.
#[cfg(not(all(target_os = "linux", feature = "tensorrt")))]
pub fn tensorrt_version() -> Option<String> {
    None
}

/// TensorRT 10 encodes its version as `(major * 100 + minor) * 100 + patch`
/// (`NV_TENSORRT_VERSION_INT`, `NvInferRuntimeBase.h`).
#[allow(dead_code)]
pub fn decode_version(version: i32) -> String {
    format!(
        "{}.{}.{}",
        version / 10_000,
        (version / 100) % 100,
        version % 100
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{CPU, CUDA, TENSORRT};

    fn pairs(options: Vec<(&'static str, String)>) -> Vec<(&'static str, &'static str)> {
        options
            .into_iter()
            .map(|(k, v)| (k, &*Box::leak(v.into_boxed_str())))
            .collect()
    }

    #[test]
    fn every_option_reaches_the_provider() {
        let trt = TensorRt {
            cache_dir: Some("/cache/engines".into()),
            timing_cache_dir: Some("/cache/timing".into()),
            fp16: true,
            profile: None,
        };

        assert_eq!(
            pairs(trt.options().unwrap()),
            vec![
                ("trt_engine_cache_enable", "1"),
                ("trt_engine_cache_path", "/cache/engines"),
                ("trt_timing_cache_enable", "1"),
                ("trt_timing_cache_path", "/cache/timing"),
                ("trt_fp16_enable", "1"),
            ]
        );
    }

    #[test]
    fn a_profile_reaches_the_provider_as_its_three_shapes() {
        let trt = TensorRt {
            profile: Some(Profile {
                min: "input:1x3x256x192".into(),
                opt: "input:16x3x256x192".into(),
                max: "input:16x3x256x192".into(),
            }),
            ..Default::default()
        };

        assert_eq!(
            pairs(trt.options().unwrap()),
            vec![
                ("trt_fp16_enable", "0"),
                ("trt_profile_min_shapes", "input:1x3x256x192"),
                ("trt_profile_opt_shapes", "input:16x3x256x192"),
                ("trt_profile_max_shapes", "input:16x3x256x192"),
            ]
        );
    }

    #[test]
    fn an_empty_or_nul_profile_shape_is_refused() {
        for (min, opt, max, name) in [
            ("", "a:1", "a:1", "min"),
            ("a:1", "a\0:1", "a:1", "opt"),
            ("a:1", "a:1", "", "max"),
        ] {
            let trt = TensorRt {
                profile: Some(Profile {
                    min: min.into(),
                    opt: opt.into(),
                    max: max.into(),
                }),
                ..Default::default()
            };

            assert!(trt.options().unwrap_err().contains(name));
        }
    }

    #[test]
    fn no_cache_dir_leaves_the_engine_cache_off() {
        let trt = TensorRt {
            fp16: false,
            ..Default::default()
        };

        assert_eq!(
            pairs(trt.options().unwrap()),
            vec![("trt_fp16_enable", "0")]
        );
    }

    #[test]
    fn a_path_with_a_nul_byte_is_refused() {
        let trt = TensorRt {
            cache_dir: Some("/cache\0/x".into()),
            ..Default::default()
        };

        assert!(trt.options().unwrap_err().contains("cache_dir"));
    }

    #[test]
    fn an_empty_path_is_refused() {
        let trt = TensorRt {
            timing_cache_dir: Some("".into()),
            ..Default::default()
        };

        assert!(trt.options().unwrap_err().contains("timing_cache_dir"));
    }

    #[test]
    fn a_cuda_build_waits_alone_and_a_tensorrt_build_does_not() {
        assert!(builds_alone(&[CUDA, CPU]));
        assert!(!builds_alone(&[TENSORRT, CUDA, CPU]));
        assert!(!builds_alone(&[CPU]));
    }

    #[test]
    fn a_tensorrt_session_keeps_the_first_run_gate() {
        assert!(gated(&[TENSORRT, CUDA, CPU]));
        assert!(gated(&[TENSORRT]));
        assert!(gated(&[CUDA, CPU]));
        assert!(!gated(&[CPU]));
    }

    #[test]
    fn tensorrt_10_versions_decode() {
        assert_eq!(decode_version(100200), "10.2.0");
        assert_eq!(decode_version(100903), "10.9.3");
        assert_eq!(decode_version(101304), "10.13.4");
    }

    #[test]
    fn the_runtime_reports_its_version() {
        let version = runtime_version().unwrap();
        assert!(version.starts_with("1."), "got {}", version);
    }
}
