{-# LANGUAGE BangPatterns #-}
module TorchLite.Backend.ROCm
  ( RTensor(..)
  , rocmOps
  ) where

import Control.Monad (forM_, when)
import Data.IORef (modifyIORef', newIORef, readIORef)
import Data.Unique (Unique, hashUnique, newUnique)
import Foreign.C.Types (CInt(..), CSize(..))
import Foreign.ForeignPtr (ForeignPtr, mallocForeignPtrArray, newForeignPtr, withForeignPtr)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr (FunPtr, Ptr)
import Foreign.Storable (peek, sizeOf)
import qualified Data.Set as Set
import qualified Data.Vector.Storable as VS
import Numeric.LinearAlgebra
  ( Matrix
  , (><)
  , flatten
  , cols
  , rows
  , reshape
  )
import System.Random (randomIO)

import TorchLite.Backend (Ops(..))

foreign import ccall unsafe "rocm_init"
  c_rocm_init :: IO CInt


foreign import ccall unsafe "rocm_alloc"
  c_rocm_alloc :: Ptr (Ptr Double) -> CSize -> IO CInt

foreign import ccall unsafe "&rocm_free"
  c_rocm_free_finalizer :: FunPtr (Ptr Double -> IO ())

foreign import ccall unsafe "rocm_copy_h2d"
  c_rocm_copy_h2d :: Ptr Double -> Ptr Double -> CSize -> IO CInt

foreign import ccall unsafe "rocm_copy_d2h"
  c_rocm_copy_d2h :: Ptr Double -> Ptr Double -> CSize -> IO CInt

foreign import ccall unsafe "rocm_memset"
  c_rocm_memset :: Ptr Double -> CInt -> CSize -> IO CInt

foreign import ccall unsafe "rocm_fill"
  c_rocm_fill :: Ptr Double -> Double -> CSize -> IO CInt

foreign import ccall unsafe "rocm_add"
  c_rocm_add :: Ptr Double -> Ptr Double -> Ptr Double -> CSize -> IO CInt

foreign import ccall unsafe "rocm_add_inplace"
  c_rocm_add_inplace :: Ptr Double -> Ptr Double -> CSize -> IO CInt

foreign import ccall unsafe "rocm_mul"
  c_rocm_mul :: Ptr Double -> Ptr Double -> Ptr Double -> CSize -> IO CInt

foreign import ccall unsafe "rocm_scale"
  c_rocm_scale :: Ptr Double -> Ptr Double -> Double -> CSize -> IO CInt

foreign import ccall unsafe "rocm_relu"
  c_rocm_relu :: Ptr Double -> Ptr Double -> CSize -> IO CInt

foreign import ccall unsafe "rocm_relu_backward"
  c_rocm_relu_backward :: Ptr Double -> Ptr Double -> Ptr Double -> CSize -> IO CInt

foreign import ccall unsafe "rocm_add_bias"
  c_rocm_add_bias :: Ptr Double -> Ptr Double -> Ptr Double -> CInt -> CInt -> IO CInt

foreign import ccall unsafe "rocm_sum_columns"
  c_rocm_sum_columns :: Ptr Double -> Ptr Double -> CInt -> CInt -> IO CInt

foreign import ccall unsafe "rocm_softmax_cols"
  c_rocm_softmax_cols :: Ptr Double -> Ptr Double -> CInt -> CInt -> IO CInt

foreign import ccall unsafe "rocm_softmax_backward"
  c_rocm_softmax_backward :: Ptr Double -> Ptr Double -> Ptr Double -> CInt -> CInt -> IO CInt

foreign import ccall unsafe "rocm_write_row"
  c_rocm_write_row :: Ptr Double -> Ptr Double -> CInt -> CInt -> CInt -> IO CInt

foreign import ccall unsafe "rocm_add_row"
  c_rocm_add_row :: Ptr Double -> Ptr Double -> CInt -> CInt -> CInt -> IO CInt

foreign import ccall unsafe "rocm_matmul"
  c_rocm_matmul :: Ptr Double -> Ptr Double -> Ptr Double -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO CInt

data RTensor = RTensor
  { rId :: !Unique
  , rRows :: !Int
  , rCols :: !Int
  , rValue :: !(ForeignPtr Double)
  , rGrad :: !(ForeignPtr Double)
  , rParents :: [RTensor]
  , rBackward :: IO ()
  }

rocmOps :: Ops RTensor
rocmOps = Ops
  { opsInit = rocmInit
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

rocmInit :: IO ()
rocmInit = rocmCheck "rocm_init" c_rocm_init

rocmCheck :: String -> IO CInt -> IO ()
rocmCheck label action = do
  rc <- action
  when (rc /= 0) (error (label ++ " failed: " ++ show rc))

allocDevice :: Int -> IO (ForeignPtr Double)
allocDevice count = alloca $ \ptrPtr -> do
  let bytes = fromIntegral (count * sizeOf (undefined :: Double))
  rocmCheck "rocm_alloc" (c_rocm_alloc ptrPtr (CSize bytes))
  ptr <- peek ptrPtr
  newForeignPtr c_rocm_free_finalizer ptr

copyH2D :: ForeignPtr Double -> VS.Vector Double -> IO ()
copyH2D dst vec = withForeignPtr dst $ \dstPtr ->
  VS.unsafeWith vec $ \srcPtr -> do
    let bytes = fromIntegral (VS.length vec * sizeOf (undefined :: Double))
    rocmCheck "rocm_copy_h2d" (c_rocm_copy_h2d dstPtr srcPtr (CSize bytes))

copyD2H :: ForeignPtr Double -> Int -> IO (VS.Vector Double)
copyD2H src count = do
  host <- mallocForeignPtrArray count
  withForeignPtr src $ \srcPtr ->
    withForeignPtr host $ \dstPtr -> do
      let bytes = fromIntegral (count * sizeOf (undefined :: Double))
      rocmCheck "rocm_copy_d2h" (c_rocm_copy_d2h dstPtr srcPtr (CSize bytes))
  pure (VS.unsafeFromForeignPtr0 host count)

fillDevice :: ForeignPtr Double -> Double -> Int -> IO ()
fillDevice fp val count = withForeignPtr fp $ \ptr ->
  rocmCheck "rocm_fill" (c_rocm_fill ptr val (fromIntegral count))

mkTensor :: Int -> Int -> ForeignPtr Double -> [RTensor] -> (ForeignPtr Double -> IO ()) -> IO RTensor
mkTensor r c val parents backward = do
  u <- newUnique
  grad <- allocDevice (r * c)
  withForeignPtr grad $ \gptr ->
    rocmCheck "rocm_memset" (c_rocm_memset gptr 0 (CSize (fromIntegral (r * c * sizeOf (undefined :: Double)))))
  pure $ RTensor u r c val grad parents (backward grad)

fromMatrix :: Matrix Double -> IO RTensor
fromMatrix val = do
  let r = rows val
  let c = cols val
  dev <- allocDevice (r * c)
  copyH2D dev (flatten val)
  mkTensor r c dev [] (\_ -> pure ())

fromListRT :: Int -> Int -> [Double] -> IO RTensor
fromListRT r c xs
  | length xs /= r * c = error "fromListRT: wrong number of elements"
  | otherwise = fromMatrix ((r >< c) xs)

zeros :: Int -> Int -> IO RTensor
zeros r c = do
  dev <- allocDevice (r * c)
  withForeignPtr dev $ \ptr ->
    rocmCheck "rocm_memset" (c_rocm_memset ptr 0 (CSize (fromIntegral (r * c * sizeOf (undefined :: Double)))))
  mkTensor r c dev [] (\_ -> pure ())

ones :: Int -> Int -> IO RTensor
ones r c = do
  dev <- allocDevice (r * c)
  fillDevice dev 1.0 (r * c)
  mkTensor r c dev [] (\_ -> pure ())

randn :: Int -> Int -> IO RTensor
randn r c = do
  xs <- sequence (replicate (r * c) (bmt 0.01))
  fromListRT r c xs

value :: RTensor -> IO (Matrix Double)
value t = do
  vec <- copyD2H (rValue t) (rRows t * rCols t)
  pure (reshape (rCols t) vec)

zeroGrad :: [RTensor] -> IO ()
zeroGrad ts = forM_ ts $ \t ->
  withForeignPtr (rGrad t) $ \ptr ->
    rocmCheck "rocm_memset" (c_rocm_memset ptr 0 (CSize (fromIntegral (rRows t * rCols t * sizeOf (undefined :: Double)))))

addGrad :: RTensor -> ForeignPtr Double -> IO ()
addGrad t g = do
  let count = rRows t * rCols t
  withForeignPtr (rGrad t) $ \dst ->
    withForeignPtr g $ \src ->
      rocmCheck "rocm_add_inplace" (c_rocm_add_inplace dst src (fromIntegral count))

add :: RTensor -> RTensor -> IO RTensor
add a b = do
  when (rRows a /= rRows b || rCols a /= rCols b) $ error "add: shape mismatch"
  out <- allocDevice (rRows a * rCols a)
  withForeignPtr (rValue a) $ \pa ->
    withForeignPtr (rValue b) $ \pb ->
      withForeignPtr out $ \po ->
        rocmCheck "rocm_add" (c_rocm_add pa pb po (fromIntegral (rRows a * rCols a)))
  mkTensor (rRows a) (rCols a) out [a, b] $ \outGrad -> do
    addGrad a outGrad
    addGrad b outGrad

mul :: RTensor -> RTensor -> IO RTensor
mul a b = do
  when (rRows a /= rRows b || rCols a /= rCols b) $ error "mul: shape mismatch"
  out <- allocDevice (rRows a * rCols a)
  withForeignPtr (rValue a) $ \pa ->
    withForeignPtr (rValue b) $ \pb ->
      withForeignPtr out $ \po ->
        rocmCheck "rocm_mul" (c_rocm_mul pa pb po (fromIntegral (rRows a * rCols a)))
  mkTensor (rRows a) (rCols a) out [a, b] $ \outGrad -> do
    tmpA <- allocDevice (rRows a * rCols a)
    tmpB <- allocDevice (rRows a * rCols a)
    withForeignPtr outGrad $ \pg ->
      withForeignPtr (rValue b) $ \pb ->
        withForeignPtr tmpA $ \ptmp ->
          rocmCheck "rocm_mul" (c_rocm_mul pg pb ptmp (fromIntegral (rRows a * rCols a)))
    withForeignPtr outGrad $ \pg ->
      withForeignPtr (rValue a) $ \pa ->
        withForeignPtr tmpB $ \ptmp ->
          rocmCheck "rocm_mul" (c_rocm_mul pg pa ptmp (fromIntegral (rRows a * rCols a)))
    addGrad a tmpA
    addGrad b tmpB

matmul :: RTensor -> RTensor -> IO RTensor
matmul a b = do
  when (rCols a /= rRows b) $ error "matmul: shape mismatch"
  let m = rRows a
  let n = rCols b
  let k = rCols a
  out <- allocDevice (m * n)
  withForeignPtr (rValue a) $ \pa ->
    withForeignPtr (rValue b) $ \pb ->
      withForeignPtr out $ \po ->
        rocmCheck "rocm_matmul" (c_rocm_matmul pa pb po
          (fromIntegral m) (fromIntegral n) (fromIntegral k)
          (fromIntegral (rRows a)) (fromIntegral (rRows b)) (fromIntegral m)
          0 0)
  mkTensor m n out [a, b] $ \outGrad -> do
    tmpA <- allocDevice (m * k)
    tmpB <- allocDevice (k * n)
    withForeignPtr outGrad $ \pg ->
      withForeignPtr (rValue b) $ \pb ->
        withForeignPtr tmpA $ \ptmp ->
          rocmCheck "rocm_matmul" (c_rocm_matmul pg pb ptmp
            (fromIntegral m) (fromIntegral k) (fromIntegral n)
            (fromIntegral m) (fromIntegral (rRows b)) (fromIntegral m)
            0 1)
    withForeignPtr (rValue a) $ \pa ->
      withForeignPtr outGrad $ \pg ->
        withForeignPtr tmpB $ \ptmp ->
          rocmCheck "rocm_matmul" (c_rocm_matmul pa pg ptmp
            (fromIntegral k) (fromIntegral n) (fromIntegral m)
            (fromIntegral (rRows a)) (fromIntegral m) (fromIntegral k)
            1 0)
    addGrad a tmpA
    addGrad b tmpB

relu :: RTensor -> IO RTensor
relu a = do
  out <- allocDevice (rRows a * rCols a)
  withForeignPtr (rValue a) $ \pa ->
    withForeignPtr out $ \po ->
      rocmCheck "rocm_relu" (c_rocm_relu pa po (fromIntegral (rRows a * rCols a)))
  mkTensor (rRows a) (rCols a) out [a] $ \outGrad -> do
    tmp <- allocDevice (rRows a * rCols a)
    withForeignPtr (rValue a) $ \pa ->
      withForeignPtr outGrad $ \pg ->
        withForeignPtr tmp $ \ptmp ->
          rocmCheck "rocm_relu_backward" (c_rocm_relu_backward pa pg ptmp (fromIntegral (rRows a * rCols a)))
    addGrad a tmp

addBias :: RTensor -> RTensor -> IO RTensor
addBias a b = do
  when (rCols b /= 1) $ error "addBias: bias must be a column"
  when (rRows a /= rRows b) $ error "addBias: shape mismatch"
  out <- allocDevice (rRows a * rCols a)
  withForeignPtr (rValue a) $ \pa ->
    withForeignPtr (rValue b) $ \pb ->
      withForeignPtr out $ \po ->
        rocmCheck "rocm_add_bias" (c_rocm_add_bias pa pb po (fromIntegral (rRows a)) (fromIntegral (rCols a)))
  mkTensor (rRows a) (rCols a) out [a, b] $ \outGrad -> do
    addGrad a outGrad
    tmp <- allocDevice (rRows b)
    withForeignPtr outGrad $ \pg ->
      withForeignPtr tmp $ \ptmp ->
        rocmCheck "rocm_sum_columns" (c_rocm_sum_columns pg ptmp (fromIntegral (rRows a)) (fromIntegral (rCols a)))
    addGrad b tmp

scale :: Double -> RTensor -> IO RTensor
scale c a = do
  out <- allocDevice (rRows a * rCols a)
  withForeignPtr (rValue a) $ \pa ->
    withForeignPtr out $ \po ->
      rocmCheck "rocm_scale" (c_rocm_scale pa po c (fromIntegral (rRows a * rCols a)))
  mkTensor (rRows a) (rCols a) out [a] $ \outGrad -> do
    tmp <- allocDevice (rRows a * rCols a)
    withForeignPtr outGrad $ \pg ->
      withForeignPtr tmp $ \ptmp ->
        rocmCheck "rocm_scale" (c_rocm_scale pg ptmp c (fromIntegral (rRows a * rCols a)))
    addGrad a tmp

softmax :: RTensor -> IO RTensor
softmax a = do
  out <- allocDevice (rRows a * rCols a)
  withForeignPtr (rValue a) $ \pa ->
    withForeignPtr out $ \po ->
      rocmCheck "rocm_softmax_cols" (c_rocm_softmax_cols pa po (fromIntegral (rRows a)) (fromIntegral (rCols a)))
  mkTensor (rRows a) (rCols a) out [a] $ \outGrad -> do
    tmp <- allocDevice (rRows a * rCols a)
    withForeignPtr out $ \ps ->
      withForeignPtr outGrad $ \pg ->
        withForeignPtr tmp $ \ptmp ->
          rocmCheck "rocm_softmax_backward" (c_rocm_softmax_backward ps pg ptmp (fromIntegral (rRows a)) (fromIntegral (rCols a)))
    addGrad a tmp

concatRows :: [RTensor] -> IO RTensor
concatRows [] = error "concatRows: empty list"
concatRows rowsIn = do
  let colsCount = rCols (head rowsIn)
  mapM_ (\t -> when (rRows t /= 1) (error "concatRows: each tensor must be 1 x n")) rowsIn
  mapM_ (\t -> when (rCols t /= colsCount) (error "concatRows: column mismatch")) rowsIn
  let outRows = length rowsIn
  out <- allocDevice (outRows * colsCount)
  forM_ (zip [0 ..] rowsIn) $ \(idx, t) ->
    withForeignPtr (rValue t) $ \pr ->
      withForeignPtr out $ \po ->
        rocmCheck "rocm_write_row" (c_rocm_write_row pr po (fromIntegral idx) (fromIntegral outRows) (fromIntegral colsCount))
  mkTensor outRows colsCount out rowsIn $ \outGrad -> do
    forM_ (zip [0 ..] rowsIn) $ \(idx, t) ->
      withForeignPtr outGrad $ \pg ->
        withForeignPtr (rGrad t) $ \dst ->
          rocmCheck "rocm_add_row" (c_rocm_add_row pg dst (fromIntegral idx) (fromIntegral outRows) (fromIntegral colsCount))

crossEntropy :: RTensor -> RTensor -> IO RTensor
crossEntropy logits target = do
  when (rRows logits /= rRows target || rCols logits /= rCols target) $
    error "crossEntropy: shape mismatch"
  soft <- allocDevice (rRows logits * rCols logits)
  withForeignPtr (rValue logits) $ \pl ->
    withForeignPtr soft $ \ps ->
      rocmCheck "rocm_softmax_cols" (c_rocm_softmax_cols pl ps (fromIntegral (rRows logits)) (fromIntegral (rCols logits)))
  softHost <- copyD2H soft (rRows logits * rCols logits)
  targetHost <- copyD2H (rValue target) (rRows target * rCols target)
  let batch = max 1 (rCols logits)
  let loss = -sum (zipWith (\t s -> t * log s) (VS.toList targetHost) (VS.toList softHost)) / fromIntegral batch
  out <- allocDevice 1
  copyH2D out (VS.fromList [loss])
  mkTensor 1 1 out [logits, target] $ \_ -> do
    let count = rRows logits * rCols logits
    tmpSoft <- allocDevice count
    tmpTarget <- allocDevice count
    tmpGrad <- allocDevice count
    withForeignPtr soft $ \ps ->
      withForeignPtr tmpSoft $ \ptmp ->
        rocmCheck "rocm_scale" (c_rocm_scale ps ptmp 1.0 (fromIntegral count))
    withForeignPtr (rValue target) $ \pt ->
      withForeignPtr tmpTarget $ \ptmp ->
        rocmCheck "rocm_scale" (c_rocm_scale pt ptmp (-1.0) (fromIntegral count))
    withForeignPtr tmpSoft $ \psoft ->
      withForeignPtr tmpTarget $ \ptmp ->
        rocmCheck "rocm_add_inplace" (c_rocm_add_inplace psoft ptmp (fromIntegral count))
    let invBatch = 1.0 / fromIntegral batch
    withForeignPtr tmpSoft $ \psoft ->
      withForeignPtr tmpGrad $ \pgrad ->
        rocmCheck "rocm_scale" (c_rocm_scale psoft pgrad invBatch (fromIntegral count))
    addGrad logits tmpGrad

backward :: RTensor -> IO ()
backward t = do
  topo <- topoSort t
  forM_ topo $ \n -> do
    let count = rRows n * rCols n
    withForeignPtr (rGrad n) $ \ptr ->
      rocmCheck "rocm_memset" (c_rocm_memset ptr 0 (CSize (fromIntegral (count * sizeOf (undefined :: Double)))))
  let count = rRows t * rCols t
  withForeignPtr (rGrad t) $ \ptr ->
    rocmCheck "rocm_fill" (c_rocm_fill ptr 1.0 (fromIntegral count))
  forM_ (reverse topo) rBackward

topoSort :: RTensor -> IO [RTensor]
topoSort t = do
  visitedRef <- newIORef Set.empty
  orderRef <- newIORef []
  let dfs node = do
        let key = hashUnique (rId node)
        visited <- readIORef visitedRef
        if Set.member key visited
          then pure ()
          else do
            modifyIORef' visitedRef (Set.insert key)
            mapM_ dfs (rParents node)
            modifyIORef' orderRef (node :)
  dfs t
  order <- readIORef orderRef
  pure (reverse order)

bmt :: Double -> IO Double
bmt scaleVal = do
  x1 <- randomIO
  x2 <- randomIO
  let x1' = if x1 < 1.0e-12 then 1.0e-12 else x1
  pure $ scaleVal * sqrt (-2 * log x1') * cos (2 * pi * x2)
