# TorchLite

TorchLite is a tiny, readable Haskell autograd playground with a CPU backend
(hmatrix/BLAS) and a ROCm GPU backend (rocBLAS + HIP kernels). It includes a
small grokking demo as a proof-of-concept.

Experimental artifacts live under `archive/` and are not part of the build.

## Requirements

- Stack (recommended) or Cabal.
- BLAS for `hmatrix` (OpenBLAS/MKL/BLIS).
- ROCm (optional) under `/opt/rocm` for the GPU backend.

## Build

```
stack build
```

## ROCm setup

Build the ROCm helper library once:
```
cbits/build_rocm_backend.sh
```

Note: the ROCm static library is linked via an absolute path in
`torchlite/torchlite.cabal`:
```
extra-lib-dirs: /home/marcel/Work/simple-neural/torchlite/cbits/rocm_build
```
Update this path if you move the repo.

## Demos

### 1) MLP demo (CPU)

Learns `y = x1 + x2` with a small MLP.

```
stack exec torchlite-demo -- --epochs 50 --batches 100 --batch-size 128 --lr 0.01 --threads 8
```

### 2) Grokking demo (POC)

Modular addition with a tiny 1-layer transformer.

CPU backend:
```
stack exec torchlite-grok -- \
  --backend cpu --modulus 113 --train-frac 0.3 \
  --batch-size 256 --epochs 20000 \
  --lr 0.001 --weight-decay 1.0 \
  --d-model 128 --heads 4 --d-mlp 512 \
  --eval-every 200 --threads 32 \
  --log-dir analysis/grok_cpu
```

ROCm backend:
```
stack exec torchlite-grok -- \
  --backend rocm --modulus 113 --train-frac 0.3 \
  --batch-size 256 --epochs 20000 \
  --lr 0.001 --weight-decay 1.0 \
  --d-model 128 --heads 4 --d-mlp 512 \
  --eval-every 200 --threads 32 \
  --log-dir analysis/grok_rocm \
  +RTS -N32
```

Notes:
- With large batches you get fewer optimizer steps per epoch. If loss looks
  flat, reduce batch size or increase epochs.
- `+RTS -N32` controls GHC runtime threads, not model size.

## Outputs

Grokking logs are written to `grokking.csv` under `--log-dir`.
You can plot using the provided script:
```
torchlite/scripts/plot_svg.py analysis/grok_rocm/grokking.csv analysis/grok_rocm
```

## Repository layout

- `torchlite/src/` — autograd core + backends
- `torchlite/app/` — CLI demos
- `cbits/` — ROCm C++/HIP kernels and build script
- `archive/` — the original simpleNeural codebase and data (not built)

## License

Same as upstream project unless noted in individual files.
