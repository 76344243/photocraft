//! On macOS, ONNX Runtime's static library (linked through `ort`) uses clang's `@available`
//! checks, whose helper `___isPlatformVersionAtLeast` lives in clang's compiler-rt; rustc links
//! with `-nodefaultlibs`, so name that library here.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let Ok(out) = Command::new("clang").arg("--print-resource-dir").output() else { return };
    let dir = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim()).join("lib/darwin");
    if dir.join("libclang_rt.osx.a").exists() {
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    } else {
        println!("cargo:warning=libclang_rt.osx.a not found under {}; linking may miss ___isPlatformVersionAtLeast", dir.display());
    }
}
