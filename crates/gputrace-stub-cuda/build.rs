fn main() {
    // Real libcuda.so.1 is DT_SYMBOLIC with zero JUMP_SLOT relocations, so its
    // internal self-calls never reach an LD_PRELOADed interposer. A stock Rust
    // cdylib binds its own calls through the PLT and would be re-intercepted,
    // making tests observe duplicate calls that a real driver never makes.
    println!("cargo::rustc-cdylib-link-arg=-Wl,-Bsymbolic");
    println!("cargo::rerun-if-changed=build.rs");
}
