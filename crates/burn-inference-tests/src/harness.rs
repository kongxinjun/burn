//! The device under test and the runner shared by the test and tool binaries.

use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};

use burn::tensor::Device;

/// Whether the device under test is another backend than the Flex reference.
pub const ACCELERATED: bool = cfg!(any(
    feature = "cpu",
    feature = "cuda",
    feature = "metal",
    feature = "wgpu"
));

/// Keeps CubeCL's environment, its autotune results and compiled kernels, in `directory` rather
/// than in the cache every process on the machine shares, so a run depends only on what the
/// directory holds. Records of how kernels were built are left out. Call it before the device is
/// first used.
pub fn isolate_environment(directory: &std::path::Path) {
    #[cfg(any(feature = "cpu", feature = "cuda", feature = "metal", feature = "wgpu"))]
    {
        use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

        let config = serde_json::json!({
            "environment": {
                "path": { "directory": directory },
                "records": { "level": "off" },
            },
        });
        let config: CubeClRuntimeConfig =
            serde_json::from_value(config).expect("The CubeCL configuration should deserialize");
        CubeClRuntimeConfig::set(config);
    }
    #[cfg(not(any(feature = "cpu", feature = "cuda", feature = "metal", feature = "wgpu")))]
    let _ = directory;
}

/// The device selected by the enabled backend feature.
pub fn device() -> Device {
    #[cfg(feature = "metal")]
    return Device::metal();
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    return Device::cuda(0);
    #[cfg(all(feature = "wgpu", not(any(feature = "metal", feature = "cuda"))))]
    return Device::wgpu(Default::default());
    #[cfg(all(
        feature = "cpu",
        not(any(feature = "metal", feature = "cuda", feature = "wgpu"))
    ))]
    return Device::cpu();
    #[cfg(not(any(feature = "cpu", feature = "cuda", feature = "metal", feature = "wgpu")))]
    Device::flex()
}

/// The name of the backend under test, with `+fusion` when fusion is enabled.
pub fn backend_name() -> String {
    let backend = if cfg!(feature = "metal") {
        "metal"
    } else if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "wgpu") {
        "wgpu"
    } else if cfg!(feature = "cpu") {
        "cpu"
    } else {
        "flex"
    };
    match cfg!(feature = "fusion") {
        true => format!("{backend}+fusion"),
        false => backend.to_string(),
    }
}

/// A named test case.
pub type TestCase = (&'static str, fn());

/// Runs the tests selected by the first command line filter and exits with their status.
///
/// `LAB_SKIP` holds comma-separated name fragments of tests to skip, for known limitations of an
/// environment.
pub fn run(mut tests: Vec<TestCase>) -> ! {
    let filter = std::env::args().skip(1).find(|arg| !arg.starts_with('-'));
    let skip = std::env::var("LAB_SKIP").unwrap_or_default();
    let skip = skip
        .split(',')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    tests.retain(|(name, _)| {
        let skipped = skip.iter().any(|part| name.contains(part));
        if skipped {
            println!("test {name} ... skipped");
        }
        !skipped && filter.as_deref().is_none_or(|filter| name.contains(filter))
    });

    let mut failed = Vec::new();
    for (name, test) in &tests {
        let passed = catch_unwind(AssertUnwindSafe(test)).is_ok();
        println!("test {name} ... {}", if passed { "ok" } else { "FAILED" });
        if !passed {
            failed.push(*name);
        }
    }

    println!(
        "\ntest result: {}. {} passed; {} failed",
        if failed.is_empty() { "ok" } else { "FAILED" },
        tests.len() - failed.len(),
        failed.len()
    );
    exit(i32::from(!failed.is_empty()));
}

/// Ends the process with `code` once the device has finished its work.
pub fn exit(code: i32) -> ! {
    device().sync().expect("The device should finish its work");
    std::io::stdout().flush().expect("Stdout should flush");

    // CubeCL's exit handler reads a thread-local that macOS has already destroyed and aborts.
    // `LAB_NORMAL_EXIT` exits normally, to check whether that is still the case.
    #[cfg(target_os = "macos")]
    if std::env::var_os("LAB_NORMAL_EXIT").is_none() {
        unsafe extern "C" {
            fn _exit(code: core::ffi::c_int) -> !;
        }
        // SAFETY: `_exit` takes a status and never returns; the output was flushed above.
        unsafe { _exit(code) }
    }
    std::process::exit(code)
}
