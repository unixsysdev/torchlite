module TorchLite.Optim
  ( sgd
  , sgdWithDecay
  , AdamW
  , initAdamW
  , adamWStep
  ) where

import Control.Monad (forM)
import Data.IORef (IORef, modifyIORef', newIORef, readIORef, writeIORef)
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra (Matrix, cols, konst, rows, toLists, fromLists)
import TorchLite.Tensor (Tensor, grad, setValue, value)

-- Stochastic gradient descent.

sgd :: Double -> [Tensor] -> IO ()
sgd lr params = mapM_ step params
  where
    step p = do
      v <- value p
      g <- grad p
      let v' = v - LA.scale lr g
      setValue p v'

-- SGD with weight decay (L2 regularization).

sgdWithDecay :: Double -> Double -> [Tensor] -> IO ()
sgdWithDecay lr decay params = mapM_ step params
  where
    step p = do
      v <- value p
      g <- grad p
      let g' = g + LA.scale decay v
      let v' = v - LA.scale lr g'
      setValue p v'

data AdamW = AdamW
  { adamLR :: Double
  , adamBeta1 :: Double
  , adamBeta2 :: Double
  , adamEps :: Double
  , adamWeightDecay :: Double
  , adamStepRef :: IORef Int
  , adamState :: [(Tensor, IORef (Matrix Double), IORef (Matrix Double))]
  }

initAdamW :: Double -> Double -> Double -> Double -> Double -> [Tensor] -> IO AdamW
initAdamW lr beta1 beta2 eps weightDecay params = do
  stepRef <- newIORef 0
  state <- forM params $ \p -> do
    v <- value p
    let zeros = konst 0 (rows v, cols v)
    mRef <- newIORef zeros
    vRef <- newIORef zeros
    pure (p, mRef, vRef)
  pure $ AdamW lr beta1 beta2 eps weightDecay stepRef state

adamWStep :: AdamW -> IO ()
adamWStep opt = do
  step <- modifyIORef' (adamStepRef opt) (+ 1) >> readIORef (adamStepRef opt)
  let b1 = adamBeta1 opt
  let b2 = adamBeta2 opt
  let lr = adamLR opt
  let eps = adamEps opt
  let wd = adamWeightDecay opt
  let b1t = 1 - b1 ^ step
  let b2t = 1 - b2 ^ step
  mapM_ (stepParam lr b1 b2 b1t b2t eps wd) (adamState opt)
  where
    stepParam lr b1 b2 b1t b2t eps wd (p, mRef, vRef) = do
      g <- grad p
      m <- readIORef mRef
      v <- readIORef vRef
      let m' = LA.scale b1 m + LA.scale (1 - b1) g
      let g2 = matZipWith (*) g g
      let v' = LA.scale b2 v + LA.scale (1 - b2) g2
      writeIORef mRef m'
      writeIORef vRef v'
      let mHat = LA.scale (1 / b1t) m'
      let vHat = LA.scale (1 / b2t) v'
      let denom = LA.cmap (\x -> sqrt x + eps) vHat
      let stepMat = matZipWith (/) mHat denom
      w <- value p
      let decayTerm = LA.scale wd w
      let update = stepMat + decayTerm
      setValue p (w - LA.scale lr update)

matZipWith :: (Double -> Double -> Double) -> Matrix Double -> Matrix Double -> Matrix Double
matZipWith f a b
  | rows a /= rows b || cols a /= cols b = error "matZipWith: shape mismatch"
  | otherwise = fromLists $ zipWith (zipWith f) (toLists a) (toLists b)
