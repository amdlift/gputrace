//! Dev-time codegen for the interposer's generated surface (ADR-0005).
//!
//! Runs bindgen over the CUDA driver header and vendors the result into
//! `crates/gputrace-shim/src/generated/`, so a downstream build needs
//! neither CUDA headers nor bindgen. Regenerating is a deliberate
//! maintainer action:
//!
//! ```text
//! nix develop --command cargo run --manifest-path crates/xtask/Cargo.toml -- gen-trampolines
//! ```
//!
//! bindgen only parses `cuda.h` inside the devShell, which is where both
//! `libclang` and the header come from.

mod surface;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use surface::{Surface, Vocabulary};
use syn::ForeignItemFn;

const HEADER_ARG: &str = "-D__CUDA_API_VERSION_INTERNAL=1";
const GENERATED_BY: &str =
    "cargo run --manifest-path crates/xtask/Cargo.toml -- gen-trampolines";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("xtask: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<String, String> {
    match args.first().map(String::as_str) {
        Some("gen-trampolines") => gen_trampolines(&args[1..]),
        Some(other) => Err(format!("unknown command `{other}`; try `gen-trampolines`")),
        None => Err("usage: xtask gen-trampolines [--header <cuda.h>] [--out <dir>]".to_owned()),
    }
}

fn gen_trampolines(args: &[String]) -> Result<String, String> {
    let header = match flag(args, "--header") {
        Some(header) => PathBuf::from(header),
        None => devshell_header()?,
    };
    let out = match flag(args, "--out") {
        Some(out) => PathBuf::from(out),
        None => repo_root().join("crates/gputrace-shim/src/generated"),
    };

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_arg(HEADER_ARG)
        .allowlist_function("^cu.*")
        .allowlist_type("^CU.*")
        .allowlist_var("^CU.*")
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        })
        .derive_default(false)
        .generate_comments(false)
        .layout_tests(false)
        .generate()
        .map_err(|e| format!("bindgen could not parse {}: {e}", header.display()))?
        .to_string();

    let parsed = syn::parse_file(&bindings).map_err(|e| format!("bindgen output: {e}"))?;
    let surface = Surface::of(parsed)?;
    let vocabulary = Vocabulary::of(&surface.types);

    write(&out.join("cuda_types.rs"), &cuda_types(&surface))?;
    write(
        &out.join("record_args.rs"),
        &record_args(&vocabulary, &surface.entry_points),
    )?;
    write(&out.join("trampolines.rs"), &trampolines(&surface))?;
    write(
        &out.join("PROVENANCE.txt"),
        &provenance(&header, &surface, &bindings),
    )?;

    Ok(format!(
        "{} entry points, {} vendored type items -> {}",
        surface.entry_points.len(),
        surface.types.len(),
        out.display()
    ))
}

fn cuda_types(surface: &Surface) -> String {
    let file = syn::File {
        shebang: None,
        attrs: Vec::new(),
        items: surface.types.clone(),
    };
    format!(
        "{}#![allow(dead_code)]\n#![allow(non_camel_case_types)]\n#![allow(non_snake_case)]\n\
         #![allow(non_upper_case_globals)]\n#![allow(unused_imports)]\n\n//! The CUDA ABI vocabulary the generated\n\
         //! trampolines are written in, vendored so a downstream build needs no\n\
         //! `cuda.h` (ADR-0005).\n\n{}",
        banner(),
        prettyplease::unparse(&file)
    )
}

fn record_args(vocabulary: &Vocabulary, entry_points: &[ForeignItemFn]) -> String {
    let file = syn::parse_file(&vocabulary.record_arg_impls(entry_points).to_string())
        .expect("generated impls parse");
    format!(
        "{}#![allow(non_camel_case_types)]\n\n//! How the argument types that are neither a\n\
         //! scalar nor a pointer record. A C enum records as its underlying integer and a\n\
         //! callback as the address it would be called at; a struct passed by value has no\n\
         //! metadata-level rendering, so it records as [`ArgValue::Null`] until a deep\n\
         //! handler captures it.\n\n\
         use std::ffi::c_void;\n\n\
         use gputrace_core::ArgValue;\n\n\
         use super::cuda_types::*;\n\
         use crate::shim::RecordArg;\n\n{}",
        banner(),
        prettyplease::unparse(&file)
    )
}

fn trampolines(surface: &Surface) -> String {
    let mut out = format!(
        "{}#![allow(non_snake_case)]\n#![allow(clippy::too_many_arguments)]\n\n\
         //! A pass-through shim for every entry point `cuda.h` declares, minus the\n\
         //! ones with a hand-written shim (see [`super::hand_written`]).\n\n\
         use super::cuda_types::*;\n\
         use super::hand_written::pass_through;\n",
        banner()
    );
    for entry in &surface.entry_points {
        out.push_str("\npass_through! {\n");
        for line in declaration(entry).lines() {
            out.push_str("    ");
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("}\n");
    }
    out
}

/// One entry point as the `pass_through!` grammar spells it, formatted by
/// the same printer as the rest of the vendored output.
fn declaration(entry: &ForeignItemFn) -> String {
    let signature = &entry.sig;
    let block: syn::File = syn::parse_quote! {
        unsafe extern "C" {
            pub #signature;
        }
    };
    let printed = prettyplease::unparse(&block);
    let lines: Vec<&str> = printed.lines().collect();
    lines[1..lines.len() - 1]
        .iter()
        .map(|line| line.strip_prefix("    ").unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n")
        .replacen("pub fn ", "fn ", 1)
}

fn provenance(header: &Path, surface: &Surface, bindings: &str) -> String {
    let bindgen = bindings
        .lines()
        .next()
        .filter(|line| line.contains("bindgen"))
        .unwrap_or("/* rust-bindgen */")
        .trim_matches(|c: char| c == '/' || c == '*' || c.is_whitespace())
        .to_owned();
    let names: HashSet<String> = surface
        .entry_points
        .iter()
        .map(|entry| entry.sig.ident.to_string())
        .collect();
    format!(
        "Generated by `{GENERATED_BY}`.\n\n\
         header        {}\n\
         clang args    {HEADER_ARG}\n\
         CUDA_VERSION  {}\n\
         bindgen       {bindgen}\n\
         entry points  {}\n\
         type items    {}\n\n\
         The vendored types are pinned to this CUDA version's ABI. A struct that\n\
         grows a field in a later version changes the ABI of every entry point\n\
         taking it, so regenerate when retargeting (ADR-0005).\n",
        header.display(),
        surface.cuda_version().unwrap_or_else(|| "unknown".to_owned()),
        names.len(),
        surface.types.len(),
    )
}

fn banner() -> String {
    format!("// @generated by `{GENERATED_BY}`. Do not edit.\n")
}

fn write(path: &Path, contents: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    std::fs::write(path, contents).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    if path.extension().is_some_and(|extension| extension == "rs") {
        rustfmt(path)?;
    }
    Ok(())
}

/// Format with the same rustfmt `cargo fmt` runs, so a regeneration lands
/// exactly what is committed and the only diff is a real change.
fn rustfmt(path: &Path) -> Result<(), String> {
    let formatted = std::process::Command::new("rustfmt")
        .args(["--edition", "2024"])
        .arg(path)
        .status()
        .map_err(|e| format!("cannot run rustfmt: {e}"))?;
    formatted
        .success()
        .then_some(())
        .ok_or_else(|| format!("rustfmt rejected {}", path.display()))
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let at = args.iter().position(|arg| arg == name)?;
    args.get(at + 1).map(String::as_str)
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("xtask lives two directories below the repo root")
        .to_path_buf()
}

/// The header the devShell puts on the include path. Pinning the exact store
/// path in `PROVENANCE.txt` is what makes a regeneration reproducible.
fn devshell_header() -> Result<PathBuf, String> {
    let flags = std::env::var("NIX_CFLAGS_COMPILE").unwrap_or_default();
    flags
        .split_whitespace()
        .find(|flag| flag.contains("cuda_cudart") && flag.ends_with("include"))
        .map(|include| PathBuf::from(include).join("cuda.h"))
        .ok_or_else(|| {
            "no CUDA header on the include path: run under `nix develop`, or pass \
             --header <path/to/cuda.h>"
                .to_owned()
        })
}
