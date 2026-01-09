# TorchLite usage guide

TorchLite is a tiny, readable Haskell autograd playground. The public API is
CPU-first (`TorchLite.Tensor` + `TorchLite.NN`), with a ROCm backend used by the
grokking demo for experimentation.

## Quick start (this repo)

```
stack build

# MLP demo: learn y = x1 + x2
stack exec torchlite-demo -- --epochs 50 --batches 100 --batch-size 128 --lr 0.01 --threads 8
```

For ROCm runs, build the helper library first:
```
cbits/build_rocm_backend.sh
```

## Use TorchLite in another project

### Stack (local path)

1) Add TorchLite to `stack.yaml`:
```
packages:
  - .
  - /path/to/torchlite
```

2) Add the dependency to your `.cabal`:
```
build-depends: base, torchlite
```

### Stack (git dependency)

```
extra-deps:
  - git: https://github.com/unixsysdev/torchlite.git
    commit: <commit-sha>
```

Note: `torchlite/torchlite.cabal` links ROCm libraries unconditionally. If you
do not have ROCm installed, comment out the ROCm `c-sources`, `include-dirs`,
`extra-lib-dirs`, and `extra-libraries` entries in that file.

### Cabal (cabal.project)

```
packages:
  .
  /path/to/torchlite
```

Then add `torchlite` to `build-depends`.

## Tensor basics and shape conventions

- A `Tensor` wraps an `hmatrix` `Matrix Double`.
- Shapes are `(rows, cols)`.
- Convention: **columns = batch**, rows = features.
  - Example: input `(in, batch)`; output `(out, batch)`.
  - Bias is a column vector `(out, 1)` and is broadcast across columns.

Common constructors:
```
fromMatrix :: Matrix Double -> IO Tensor
fromList   :: Int -> Int -> [Double] -> IO Tensor
zeros      :: Int -> Int -> IO Tensor
ones       :: Int -> Int -> IO Tensor
randn      :: Int -> Int -> IO Tensor
```

Inspect values:
```
value :: Tensor -> IO (Matrix Double)
grad  :: Tensor -> IO (Matrix Double)
```

## Built-in layers

TorchLite exposes small helpers in `TorchLite.NN`:

```
model  <- mlp [2, 64, 64, 1]
params <- pure (mlpParams model)
pred   <- forwardMLP model x
```

`Linear` layers use weight shape `(out, in)` and bias shape `(out, 1)`.

## Defining your own model

You define a model as a record of parameters, a forward function, and a list
of params for the optimizer.

```haskell
import TorchLite.Tensor
import TorchLite.Optim (sgd)

data LinearModel = LinearModel
  { w :: Tensor
  , b :: Tensor
  }

initLinear :: Int -> Int -> IO LinearModel
initLinear inSize outSize = do
  w <- randn outSize inSize
  b <- zeros outSize 1
  pure (LinearModel w b)

forwardLinear :: LinearModel -> Tensor -> IO Tensor
forwardLinear m x = do
  z <- matmul (w m) x
  addBias z (b m)

paramsLinear :: LinearModel -> [Tensor]
paramsLinear m = [w m, b m]
```

### Example: add Fourier features

```haskell
-- x is (1, batch)
feat <- do
  s <- sinT x
  c <- cosT x
  concatRows [s, c]  -- (2, batch)
```

You can then feed `feat` into an MLP.

## Training loop pattern

Typical training steps:

1) Forward pass to compute loss.
2) `backward loss`
3) Optimizer step.
4) `zeroGrad params`

```haskell
pred <- forwardLinear model x
diff <- sub pred y
sq   <- mul diff diff
loss <- meanAll sq
backward loss
sgd 0.01 (paramsLinear model)
zeroGrad (paramsLinear model)
```

For classification, use `crossEntropy`:

```haskell
-- logits: (classes, batch)
-- target: one-hot (classes, batch)
loss <- crossEntropy logits target
```

If you need AdamW:

```haskell
import TorchLite.Optim (initAdamW, adamWStep)

opt <- initAdamW 1e-3 0.9 0.999 1e-8 1.0 params
...
backward loss
adamWStep opt
zeroGrad params
```

## ROCm backend (advanced)

The ROCm path is used by the grokking demo and lives under
`TorchLite.Backend.*` and `TorchLite.Optim.ROCm`. It is not a drop-in
replacement for `TorchLite.Tensor` yet, but you can study or reuse the
`TorchLite.Grokking` training loop for GPU-backed experiments.
