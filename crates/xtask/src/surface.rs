//! What bindgen found in `cuda.h`, split into the two things the interposer
//! needs from it: the ABI vocabulary (types), and the entry points to stand
//! a shim in front of.

use std::collections::{BTreeMap, HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{File, ForeignItem, ForeignItemFn, Item, ItemStruct, PathArguments, Type, UseTree};

pub struct Surface {
    pub types: Vec<Item>,
    pub entry_points: Vec<ForeignItemFn>,
}

impl Surface {
    pub fn of(file: File) -> Result<Self, String> {
        let mut types = Vec::new();
        let mut entry_points = Vec::new();
        for item in file.items {
            let Item::ForeignMod(block) = item else {
                types.push(item);
                continue;
            };
            for foreign in block.items {
                match foreign {
                    ForeignItem::Fn(entry) if entry.sig.ident.to_string().starts_with("cu") => {
                        entry_points.push(entry);
                    }
                    other => {
                        return Err(format!(
                            "unexpected foreign item in the bindgen output: {}",
                            other.to_token_stream()
                        ));
                    }
                }
            }
        }
        entry_points.sort_by_key(|entry| entry.sig.ident.to_string());
        Ok(Self {
            types,
            entry_points,
        })
    }

    pub fn cuda_version(&self) -> Option<String> {
        self.types.iter().find_map(|item| match item {
            Item::Const(konst) if konst.ident == "CUDA_VERSION" => {
                Some(konst.expr.to_token_stream().to_string())
            }
            _ => None,
        })
    }
}

/// How one argument of a shimmed entry point is recorded. [`Recording::Builtin`]
/// types are the ones `shim::RecordArg` already covers; the rest need an impl
/// generated alongside the trampolines.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Recording {
    Builtin,
    Unsigned,
    Signed,
    Callback,
    Opaque,
}

/// The type names bindgen emitted, so an argument's spelling can be followed
/// to the thing it is really made of.
pub struct Vocabulary {
    aliases: HashMap<String, Type>,
    renames: HashMap<String, String>,
    structs: HashMap<String, ItemStruct>,
    unions: HashSet<String>,
}

impl Vocabulary {
    pub fn of(types: &[Item]) -> Self {
        let mut vocabulary = Self {
            aliases: HashMap::new(),
            renames: HashMap::new(),
            structs: HashMap::new(),
            unions: HashSet::new(),
        };
        for item in types {
            match item {
                Item::Type(alias) => {
                    vocabulary
                        .aliases
                        .insert(alias.ident.to_string(), (*alias.ty).clone());
                }
                Item::Struct(structure) => {
                    vocabulary
                        .structs
                        .insert(structure.ident.to_string(), structure.clone());
                }
                Item::Union(union) => {
                    vocabulary.unions.insert(union.ident.to_string());
                }
                Item::Use(import) => vocabulary.collect_rename(&import.tree),
                _ => {}
            }
        }
        vocabulary
    }

    fn collect_rename(&mut self, tree: &UseTree) {
        match tree {
            UseTree::Path(path) => self.collect_rename(&path.tree),
            UseTree::Group(group) => group.items.iter().for_each(|it| self.collect_rename(it)),
            UseTree::Rename(rename) => {
                self.renames
                    .insert(rename.rename.to_string(), rename.ident.to_string());
            }
            _ => {}
        }
    }

    /// The type an impl must be written for, and how it records, or `None`
    /// when `shim::RecordArg` already covers the type.
    pub fn recording(&self, ty: &Type) -> Option<(Type, Recording)> {
        match self.classify(ty, 0) {
            (_, Recording::Builtin) => None,
            (canonical, recording) => Some((canonical, recording)),
        }
    }

    fn classify(&self, ty: &Type, depth: usize) -> (Type, Recording) {
        if depth > 16 {
            return (ty.clone(), Recording::Opaque);
        }
        let Type::Path(path) = ty else {
            return match ty {
                Type::Ptr(_) => (ty.clone(), Recording::Builtin),
                _ => (ty.clone(), Recording::Opaque),
            };
        };
        if path.qself.is_some() {
            return (ty.clone(), Recording::Opaque);
        }
        let Some(segment) = path.path.segments.last() else {
            return (ty.clone(), Recording::Opaque);
        };
        let name = segment.ident.to_string();
        if name == "Option" && matches!(segment.arguments, PathArguments::AngleBracketed(_)) {
            return (without_argument_names(ty), Recording::Callback);
        }
        if let Some(builtin) = primitive(&name) {
            return (ty.clone(), builtin);
        }
        if let Some(target) = self.renames.get(&name) {
            return self.classify(&named(target), depth + 1);
        }
        if let Some(target) = self.aliases.get(&name) {
            return self.classify(target, depth + 1);
        }
        if let Some(structure) = self.structs.get(&name) {
            return (named(&name), self.newtype_recording(structure, depth));
        }
        if self.unions.contains(&name) {
            return (named(&name), Recording::Opaque);
        }
        (ty.clone(), Recording::Opaque)
    }

    /// bindgen renders a C enum as a one-field tuple struct over its
    /// underlying integer, which records as that integer.
    fn newtype_recording(&self, structure: &ItemStruct, depth: usize) -> Recording {
        let syn::Fields::Unnamed(fields) = &structure.fields else {
            return Recording::Opaque;
        };
        let [field] = &fields.unnamed.iter().collect::<Vec<_>>()[..] else {
            return Recording::Opaque;
        };
        self.integer_sign(&field.ty, depth + 1)
            .unwrap_or(Recording::Opaque)
    }

    fn integer_sign(&self, ty: &Type, depth: usize) -> Option<Recording> {
        if depth > 16 {
            return None;
        }
        let Type::Path(path) = ty else {
            return None;
        };
        let name = path.path.segments.last()?.ident.to_string();
        if let Some(sign) = integer_sign(&name) {
            return Some(sign);
        }
        if let Some(target) = self.renames.get(&name) {
            return self.integer_sign(&named(target), depth + 1);
        }
        self.aliases
            .get(&name)
            .and_then(|target| self.integer_sign(target, depth + 1))
    }

    /// The `RecordArg` impls the generated trampolines need on top of the
    /// scalar and pointer ones `shim` ships.
    pub fn record_arg_impls(&self, entry_points: &[ForeignItemFn]) -> TokenStream {
        let mut needed: BTreeMap<String, (Type, Recording)> = BTreeMap::new();
        for argument in entry_points.iter().flat_map(|entry| entry.sig.inputs.iter()) {
            let syn::FnArg::Typed(argument) = argument else {
                continue;
            };
            if let Some((canonical, recording)) = self.recording(&argument.ty) {
                needed.insert(canonical.to_token_stream().to_string(), (canonical, recording));
            }
        }

        let impls = needed.values().map(|(ty, recording)| {
            let value = match recording {
                Recording::Unsigned => quote!(ArgValue::Unsigned(self.0 as u64)),
                Recording::Signed => quote!(ArgValue::Signed(self.0 as i64)),
                Recording::Callback => {
                    quote!(ArgValue::HostPtr(self.map_or(0, |entry| entry as *const c_void as u64)))
                }
                Recording::Builtin | Recording::Opaque => quote!(ArgValue::Null),
            };
            quote! {
                impl RecordArg for #ty {
                    fn record(&self) -> ArgValue {
                        #value
                    }
                }
            }
        });
        quote!(#(#impls)*)
    }
}

/// A callback type spelled without its parameter names: the same type, but
/// one an impl can be written for without tripping `non_snake_case`.
fn without_argument_names(ty: &Type) -> Type {
    let mut ty = ty.clone();
    if let Type::Path(path) = &mut ty
        && let Some(segment) = path.path.segments.last_mut()
        && let PathArguments::AngleBracketed(arguments) = &mut segment.arguments
    {
        for argument in arguments.args.iter_mut() {
            if let syn::GenericArgument::Type(Type::BareFn(callback)) = argument {
                for input in callback.inputs.iter_mut() {
                    input.name = None;
                }
            }
        }
    }
    ty
}

fn named(name: &str) -> Type {
    let ident = syn::Ident::new(name, proc_macro2::Span::call_site());
    syn::parse_quote!(#ident)
}

fn primitive(name: &str) -> Option<Recording> {
    let known = [
        "bool", "f32", "f64", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32",
        "i64", "i128", "isize", "c_char", "c_schar", "c_uchar", "c_short", "c_ushort", "c_int",
        "c_uint", "c_long", "c_ulong", "c_longlong", "c_ulonglong", "c_float", "c_double",
    ];
    known.contains(&name).then_some(Recording::Builtin)
}

fn integer_sign(name: &str) -> Option<Recording> {
    let unsigned = [
        "u8",
        "u16",
        "u32",
        "u64",
        "usize",
        "c_uchar",
        "c_ushort",
        "c_uint",
        "c_ulong",
        "c_ulonglong",
    ];
    let signed = [
        "i8",
        "i16",
        "i32",
        "i64",
        "isize",
        "c_char",
        "c_schar",
        "c_short",
        "c_int",
        "c_long",
        "c_longlong",
    ];
    if unsigned.contains(&name) {
        return Some(Recording::Unsigned);
    }
    signed.contains(&name).then_some(Recording::Signed)
}
