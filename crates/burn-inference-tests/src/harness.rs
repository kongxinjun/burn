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
pub fn run(mut tests: Vec<TestCase>) -> ! {
    let filter = std::env::args().skip(1).find(|arg| !arg.starts_with('-'));
    tests.retain(|(name, _)| filter.as_deref().is_none_or(|filter| name.contains(filter)));

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
