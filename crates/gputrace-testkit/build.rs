use std::env;
use std::path::PathBuf;

fn main() {
    // `cargo test` never builds cdylib artifacts and there is no stable
    // `CARGO_CDYLIB_FILE_*`, so the harness has to build and locate the shim
    // and the stub itself. OUT_DIR is `<target>/<profile>/build/<pkg>-<hash>/out`.
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let profile_dir = out.ancestors().nth(3).expect("profile dir");
    println!(
        "cargo::rustc-env=GPUTRACE_PROFILE_DIR={}",
        profile_dir.display()
    );

    let profile = profile_dir
        .file_name()
        .expect("profile name")
        .to_string_lossy();
    println!("cargo::rustc-env=GPUTRACE_CARGO_PROFILE={profile}");

    let host = env::var("HOST").expect("HOST");
    let target = env::var("TARGET").expect("TARGET");
    let explicit_target = if target == host {
        String::new()
    } else {
        target
    };
    println!("cargo::rustc-env=GPUTRACE_CARGO_TARGET={explicit_target}");

    println!("cargo::rerun-if-changed=build.rs");
}
