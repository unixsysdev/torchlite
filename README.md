# simpleNeural

[![Build Status](https://travis-ci.org/kevchn/simple-neural.svg?branch=master)](https://travis-ci.org/kevchn/simple-neural)

A Haskell neural net playground with multiple backends (BLAS CPU and ROCm GPU), plus a new CPU-only autograd sandbox in `torchlite/`.

This repo started from the original simple-neural project by kevchn. At this point, the only thing still shared is the MNIST data format and the basic project name; the codebase has been refactored for performance, configurability, and backend comparison.

## Highlights

- Configurable CLI (no hardcoded paths or sizes).
- Multiple backends for side-by-side comparison:
  - `cpu` (BLAS via hmatrix)
  - `rocm` (rocBLAS + HIP)
- Mini-batch training for ROCm with tunable batch size.
- Model save/load with a shared text format across backends.
- Fully commented reference sources in `src_commented/` (comments on every line).
- A separate `torchlite/` subproject for building PyTorch-like models from scratch.

## Requirements

- Haskell toolchain (stack or cabal).
- BLAS for `hmatrix` (OpenBLAS/MKL).
- ROCm optional (if using `--backend rocm`): libraries under `/opt/rocm`.

## Build

```
stack build
```

This builds both the main MNIST executable and the `torchlite` demo.

## Usage

Train (CPU backend):
```
simple-neural train \
  --backend cpu \
  --train-images data/train-images-idx3-ubyte.gz \
  --train-labels data/train-labels-idx1-ubyte.gz \
  --layers 784,30,10 \
  --learning-rate 0.002 \
  --epochs 1 \
  --cpu-threads 8 \
  --model-out data/model.txt
```

Test:
```
simple-neural test \
  --backend cpu \
  --test-images data/t10k-images-idx3-ubyte.gz \
  --test-labels data/t10k-labels-idx1-ubyte.gz \
  --cpu-threads 8 \
  --model-in data/model.txt
```

Backend selection:
```
--backend cpu
--backend rocm
```

## TorchLite (from-scratch autograd)

The `torchlite/` folder is a clean autograd sandbox with pluggable backends:
- `cpu`: BLAS-backed matrices via `hmatrix`
- `rocm`: device-resident tensors + rocBLAS + HIP kernels

If you want the ROCm backend, build the helper library first:
```
cbits/build_rocm_backend.sh
```

Build and run the demo:
```
stack exec torchlite-demo -- --epochs 50 --batches 100 --batch-size 128 --lr 0.01 --threads 8
```

Demo behavior:
- Trains an MLP to learn `y = x1 + x2` on random data.
- Reports average loss per epoch and a test mean absolute error.

Grokking demo (modular addition):
```
stack exec torchlite-grok -- --modulus 97 --train-frac 0.5 --epochs 200 --batch-size 256 --lr 0.01 --weight-decay 1e-4 --d-model 128 --heads 4 --d-mlp 512 --threads 8
```
ROCm backend:
```
stack exec torchlite-grok -- --backend rocm --modulus 113 --train-frac 0.3 --epochs 25000 --batch-size 3829 --lr 0.001 --weight-decay 1.0 --threads 32
```

What to watch:
- Early epochs: high train accuracy, low test accuracy (memorization).
- Later epochs: test accuracy rises (grokking).

Grokking plots (SVG):
```
torchlite/scripts/plot_svg.py analysis/grokking.csv analysis
```
The training run writes `analysis/grokking.csv` by default. Use `--log-dir` to change it.

GPU notes:
- TorchLite now uses ROCm device-resident tensors, element-wise HIP kernels, and an on-GPU AdamW update.
- The kernels are intentionally straightforward to keep the backend readable.

## Commented sources

There is a fully commented reference copy of the core sources in `src_commented/`:

- `src_commented/Main.hs`
- `src_commented/NNet.hs`
- `src_commented/NNetRocm.hs`

These files are for learning and reading; the build uses `src/` by default.

## CPU threading

The executable is built with the threaded RTS. Use `--cpu-threads` to set both RTS capabilities and BLAS thread counts:
```
simple-neural train --backend cpu --cpu-threads 8 ...
simple-neural test --backend cpu --cpu-threads 8 ...
```

## ROCm backend

ROCm runs mini-batch training and uses rocBLAS for GEMM.

Batch size (default 128):
```
NNET_BATCH_SIZE=256 simple-neural train --backend rocm ...
```

If runtime linking fails:
```
export LD_LIBRARY_PATH=/opt/rocm/lib
```

Notes:
- Large batches improve GPU throughput but can hurt accuracy.
- Smaller batches improve accuracy but reduce GPU speedup.

## Tuning tips

- Speed first (GPU): larger batch sizes (512 to 2048) and more epochs.
- Accuracy first (CPU): smaller learning rate and more epochs.
- Compare runs by keeping data, layers, and epochs constant, then adjusting only batch size and learning rate for ROCm.

## How learning works (plain English)

The network is a stack of layers. Each layer does two things:
1) multiply the input by a weight matrix, and
2) add a bias vector, then apply an activation (ReLU here).

In math form for one layer:
```
z = W * x + b
a = max(0, z)
```

Training repeats this loop:
1) **Forward pass**: run inputs through the layers to get predictions.
2) **Loss**: compare predictions to the correct label (one-hot vector).
3) **Backprop**: compute how much each weight/bias contributed to the error.
4) **Update**: move weights/biases a small step to reduce the error.

The update step is a simple gradient step:
```
W = W - lr * dW
b = b - lr * db
```

For ROCm training we do the same math, but in **mini-batches**:
we stack many inputs together in a matrix and use GEMM (matrix-matrix multiply)
to speed up the heavy parts on the GPU. Smaller batches are closer to classic
SGD (often better accuracy), larger batches are faster (often lower accuracy).
