fn main() {
    // Only compile Objective-C extension code when targeting macOS
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rerun-if-changed=macos-extension/");

        cc::Build::new()
            .files(&[
                "macos-extension/ProviderSource.m",
                "macos-extension/DeviceSource.m",
                "macos-extension/StreamSource.m",
                "macos-extension/SinkStreamSource.m",
                "macos-extension/ExtensionInstaller.m",
            ])
            .flag("-fobjc-arc")
            .flag("-fmodules")
            .include("macos-extension")
            .compile("vcam_extension");

        // Link required frameworks
        println!("cargo:rustc-link-lib=framework=CoreMediaIO");
        println!("cargo:rustc-link-lib=framework=CoreMedia");
        println!("cargo:rustc-link-lib=framework=CoreVideo");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=SystemExtensions");
    }
}
