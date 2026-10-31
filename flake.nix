{
  description = "gputrace devShell";

  inputs = {
    nixpkgs.url = "https://channels.nixos.org/nixpkgs-unstable/nixexprs.tar.zst";
  };

  # The CUDA toolkit is only needed to build gputrace (bindgen codegen and
  # test workloads); it is never linked into the shipped shim. Pull the
  # prebuilt unfree binaries from the CUDA cache instead of compiling them.
  nixConfig = {
    extra-substituters = [ "https://cuda-maintainers.cachix.org" ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  outputs = { self, nixpkgs, }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
  in {

    devShells.${system}.default = pkgs.mkShell {
      buildInputs = with pkgs; [
        cargo
        rustc
        rustfmt
        clippy
        rust-analyzer

        # Dev-only CUDA toolchain: cuda.h (Driver API) for bindgen, nvcc to
        # build capture targets. The released shim links no CUDA and runs on
        # any Linux with just the NVIDIA driver present.
        cudaPackages.cuda_cudart
        cudaPackages.cuda_nvcc

        # bindgen needs libclang to parse the CUDA driver headers.
        rustPlatform.bindgenHook

        (python314.withPackages (ps: with ps; [ ]))
      ];
      env.RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
    };
  };
}
