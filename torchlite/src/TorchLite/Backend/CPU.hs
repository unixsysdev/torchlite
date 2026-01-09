module TorchLite.Backend.CPU
  ( cpuOps
  ) where

import TorchLite.Backend (Ops(..))
import TorchLite.Tensor
  ( Tensor
  , add
  , addBias
  , backward
  , concatRows
  , crossEntropy
  , fromMatrix
  , matmul
  , mul
  , ones
  , randn
  , relu
  , scale
  , softmax
  , value
  , zeroGrad
  , zeros
  )

cpuOps :: Ops Tensor
cpuOps = Ops
  { opsInit = pure ()
  , opsFromMatrix = fromMatrix
  , opsZeros = zeros
  , opsOnes = ones
  , opsRandn = randn
  , opsValue = value
  , opsZeroGrad = zeroGrad
  , opsAdd = add
  , opsMul = mul
  , opsMatmul = matmul
  , opsRelu = relu
  , opsAddBias = addBias
  , opsScale = scale
  , opsSoftmax = softmax
  , opsConcatRows = concatRows
  , opsCrossEntropy = crossEntropy
  , opsBackward = backward
  }
