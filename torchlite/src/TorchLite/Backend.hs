module TorchLite.Backend
  ( Ops(..)
  ) where

import Numeric.LinearAlgebra (Matrix)

data Ops t = Ops
  { opsInit :: IO ()
  , opsFromMatrix :: Matrix Double -> IO t
  , opsZeros :: Int -> Int -> IO t
  , opsOnes :: Int -> Int -> IO t
  , opsRandn :: Int -> Int -> IO t
  , opsValue :: t -> IO (Matrix Double)
  , opsZeroGrad :: [t] -> IO ()
  , opsAdd :: t -> t -> IO t
  , opsMul :: t -> t -> IO t
  , opsMatmul :: t -> t -> IO t
  , opsRelu :: t -> IO t
  , opsAddBias :: t -> t -> IO t
  , opsScale :: Double -> t -> IO t
  , opsSoftmax :: t -> IO t
  , opsConcatRows :: [t] -> IO t
  , opsCrossEntropy :: t -> t -> IO t
  , opsBackward :: t -> IO ()
  }
