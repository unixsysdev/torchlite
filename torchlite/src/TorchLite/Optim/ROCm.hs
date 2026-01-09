module TorchLite.Optim.ROCm
  ( AdamW
  , initAdamW
  , adamWStep
  ) where

import Control.Monad (forM)
import Data.IORef (IORef, modifyIORef', newIORef, readIORef)
import Foreign.C.Types (CInt(..), CSize(..))
import Foreign.ForeignPtr (ForeignPtr, newForeignPtr, withForeignPtr)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr (FunPtr, Ptr)
import Foreign.Storable (peek, sizeOf)

import TorchLite.Backend.ROCm (RTensor(..))

foreign import ccall unsafe "rocm_alloc"
  c_rocm_alloc :: Ptr (Ptr Double) -> CSize -> IO CInt

foreign import ccall unsafe "&rocm_free"
  c_rocm_free_finalizer :: FunPtr (Ptr Double -> IO ())

foreign import ccall unsafe "rocm_memset"
  c_rocm_memset :: Ptr Double -> CInt -> CSize -> IO CInt

foreign import ccall unsafe "rocm_adamw_step"
  c_rocm_adamw_step :: Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double
                    -> Double -> Double -> Double -> Double -> Double -> Double -> Double -> CSize -> IO CInt

data AdamW = AdamW
  { adamLR :: Double
  , adamBeta1 :: Double
  , adamBeta2 :: Double
  , adamEps :: Double
  , adamWeightDecay :: Double
  , adamStepRef :: IORef Int
  , adamState :: [(RTensor, ForeignPtr Double, ForeignPtr Double, Int)]
  }

initAdamW :: Double -> Double -> Double -> Double -> Double -> [RTensor] -> IO AdamW
initAdamW lr beta1 beta2 eps weightDecay params = do
  stepRef <- newIORef 0
  state <- forM params $ \p -> do
    let count = rRows p * rCols p
    mRef <- allocDevice count
    vRef <- allocDevice count
    memsetDevice mRef count
    memsetDevice vRef count
    pure (p, mRef, vRef, count)
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
    stepParam lr b1 b2 b1t b2t eps wd (p, mRef, vRef, count) =
      withForeignPtr (rValue p) $ \paramPtr ->
        withForeignPtr (rGrad p) $ \gradPtr ->
          withForeignPtr mRef $ \mPtr ->
            withForeignPtr vRef $ \vPtr ->
              rocmCheck "rocm_adamw_step"
                (c_rocm_adamw_step paramPtr gradPtr mPtr vPtr lr b1 b2 eps wd b1t b2t (fromIntegral count))

allocDevice :: Int -> IO (ForeignPtr Double)
allocDevice count = alloca $ \ptrPtr -> do
  let bytes = fromIntegral (count * sizeOf (undefined :: Double))
  rocmCheck "rocm_alloc" (c_rocm_alloc ptrPtr (CSize bytes))
  ptr <- peek ptrPtr
  newForeignPtr c_rocm_free_finalizer ptr

memsetDevice :: ForeignPtr Double -> Int -> IO ()
memsetDevice fp count = withForeignPtr fp $ \ptr ->
  rocmCheck "rocm_memset" (c_rocm_memset ptr 0 (CSize (fromIntegral (count * sizeOf (undefined :: Double)))))

rocmCheck :: String -> IO CInt -> IO ()
rocmCheck label action = do
  rc <- action
  if rc == 0
    then pure ()
    else error (label ++ " failed: " ++ show rc)
