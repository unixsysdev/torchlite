module TorchLite.NN
  ( Linear
  , MLP
  , linear
  , forwardLinear
  , linearParams
  , mlp
  , forwardMLP
  , mlpParams
  ) where

import Control.Monad (foldM, when, zipWithM)
import TorchLite.Tensor
  ( Tensor
  , addBias
  , matmul
  , randn
  , relu
  , zeros
  )

-- A simple linear layer: y = W x + b

data Linear = Linear
  { linearW :: Tensor
  , linearB :: Tensor
  }

linear :: Int -> Int -> IO Linear
linear inSize outSize = do
  w <- randn outSize inSize
  b <- zeros outSize 1
  pure $ Linear w b

forwardLinear :: Linear -> Tensor -> IO Tensor
forwardLinear (Linear w b) x = do
  z <- matmul w x
  addBias z b

linearParams :: Linear -> [Tensor]
linearParams layer = [linearW layer, linearB layer]

-- Multi-layer perceptron with ReLU in hidden layers.

data MLP = MLP
  { mlpLayers :: [Linear]
  }

mlp :: [Int] -> IO MLP
mlp sizes = do
  when (length sizes < 2) $ error "mlp: need at least input and output sizes"
  layers <- zipWithM linear sizes (tail sizes)
  pure $ MLP layers

forwardMLP :: MLP -> Tensor -> IO Tensor
forwardMLP (MLP layers) input = go layers input
  where
    go [] x = pure x
    go [lastLayer] x = forwardLinear lastLayer x
    go (layer:rest) x = do
      z <- forwardLinear layer x
      a <- relu z
      go rest a

mlpParams :: MLP -> [Tensor]
mlpParams (MLP layers) = concatMap linearParams layers
