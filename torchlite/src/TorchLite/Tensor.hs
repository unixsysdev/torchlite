{-# LANGUAGE BangPatterns #-}
module TorchLite.Tensor
  ( Tensor
  , fromMatrix
  , fromList
  , zeros
  , ones
  , randn
  , value
  , grad
  , setValue
  , zeroGrad
  , add
  , sub
  , mul
  , matmul
  , matmulRocm
  , sinT
  , cosT
  , relu
  , addBias
  , transposeT
  , softmax
  , softmaxRows
  , concatRows
  , logSoftmax
  , crossEntropy
  , sumAll
  , meanAll
  , scale
  , backward
  , backwardAccum
  ) where

import Control.Monad (forM_, replicateM, when)
import Data.IORef (IORef, modifyIORef', newIORef, readIORef, writeIORef)
import Data.List (foldl')
import Data.Set (Set)
import Data.Unique (Unique, hashUnique, newUnique)
import qualified Data.Set as Set
import Foreign (Ptr, castPtr, mallocForeignPtrArray)
import Foreign.C.Types (CDouble(..), CInt(..))
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr)
import qualified Data.Vector.Storable as VS
import Numeric.LinearAlgebra
  ( Matrix
  , Vector
  , (><)
  , (<> )
  , asColumn
  , atIndex
  , cols
  , cmap
  , fromColumns
  , fromLists
  , konst
  , reshape
  , rows
  , sumElements
  , flatten
  , toColumns
  , toLists
  , tr
  )
import qualified Numeric.LinearAlgebra as LA
import System.Random (randomIO)

foreign import ccall unsafe "rocblas_dgemm_host"
  c_rocblas_dgemm_host :: CInt -> CInt -> CInt -> CInt -> CInt
                       -> Ptr CDouble -> CInt
                       -> Ptr CDouble -> CInt
                       -> Ptr CDouble -> CInt
                       -> IO CInt

-- A tiny autograd engine for matrices.

type Mat = Matrix Double

data Tensor = Tensor
  { tId :: !Unique
  , tValue :: !(IORef Mat)
  , tGrad :: !(IORef Mat)
  , tParents :: [Tensor]
  , tBackward :: IO ()
  }

-- Construction helpers.

fromMatrix :: Mat -> IO Tensor
fromMatrix val = mkTensor val [] (\_ -> pure ())

fromList :: Int -> Int -> [Double] -> IO Tensor
fromList r c xs
  | length xs /= r * c = error "fromList: wrong number of elements"
  | otherwise = fromMatrix ((r >< c) xs)

zeros :: Int -> Int -> IO Tensor
zeros r c = fromMatrix (konst 0 (r, c))

ones :: Int -> Int -> IO Tensor
ones r c = fromMatrix (konst 1 (r, c))

randn :: Int -> Int -> IO Tensor
randn r c = do
  xs <- replicateM (r * c) (bmt 0.01)
  fromList r c xs

value :: Tensor -> IO Mat
value t = readIORef (tValue t)

grad :: Tensor -> IO Mat
grad t = readIORef (tGrad t)

setValue :: Tensor -> Mat -> IO ()
setValue t v = writeIORef (tValue t) v

zeroGrad :: [Tensor] -> IO ()
zeroGrad ts =
  forM_ ts $ \t -> do
    v <- value t
    writeIORef (tGrad t) (zerosLike v)

mkTensor :: Mat -> [Tensor] -> (IORef Mat -> IO ()) -> IO Tensor
mkTensor val parents backward = do
  u <- newUnique
  vRef <- newIORef val
  gRef <- newIORef (zerosLike val)
  let t = Tensor u vRef gRef parents (backward gRef)
  pure t

addGrad :: Tensor -> Mat -> IO ()
addGrad t g = modifyIORef' (tGrad t) (+ g)

zerosLike :: Mat -> Mat
zerosLike m = konst 0 (rows m, cols m)

bmt :: Double -> IO Double
bmt scale = do
  x1 <- randomIO
  x2 <- randomIO
  let x1' = if x1 < 1.0e-12 then 1.0e-12 else x1
  pure $ scale * sqrt (-2 * log x1') * cos (2 * pi * x2)

-- Elementwise helpers.

matZipWith :: (Double -> Double -> Double) -> Mat -> Mat -> Mat
matZipWith f a b
  | rows a /= rows b || cols a /= cols b = error "matZipWith: shape mismatch"
  | otherwise = fromLists $ zipWith (zipWith f) (toLists a) (toLists b)

hadamard :: Mat -> Mat -> Mat
hadamard = matZipWith (*)

sumColumns :: Mat -> Vector Double
sumColumns m
  | cols m == 0 = konst 0 (rows m)
  | otherwise = foldl' (+) (konst 0 (rows m)) (toColumns m)

repeatColumns :: Vector Double -> Int -> Mat
repeatColumns v n = fromColumns (replicate n v)

-- Operations.

add :: Tensor -> Tensor -> IO Tensor
add a b = do
  va <- value a
  vb <- value b
  mkTensor (va + vb) [a, b] $ \outGradRef -> do
    outGrad <- readIORef outGradRef
    addGrad a outGrad
    addGrad b outGrad

sub :: Tensor -> Tensor -> IO Tensor
sub a b = do
  va <- value a
  vb <- value b
  mkTensor (va - vb) [a, b] $ \outGradRef -> do
    outGrad <- readIORef outGradRef
    addGrad a outGrad
    addGrad b (LA.scale (-1) outGrad)

mul :: Tensor -> Tensor -> IO Tensor
mul a b = do
  va <- value a
  vb <- value b
  let outVal = hadamard va vb
  mkTensor outVal [a, b] $ \outGradRef -> do
    outGrad <- readIORef outGradRef
    addGrad a (hadamard outGrad vb)
    addGrad b (hadamard outGrad va)

matmul :: Tensor -> Tensor -> IO Tensor
matmul a b = do
  va <- value a
  vb <- value b
  let outVal = va <> vb
  mkTensor outVal [a, b] $ \outGradRef -> do
    outGrad <- readIORef outGradRef
    addGrad a (outGrad <> tr vb)
    addGrad b (tr va <> outGrad)

matmulRocm :: Tensor -> Tensor -> IO Tensor
matmulRocm a b = do
  va <- value a
  vb <- value b
  outVal <- rocblasGemmHost va vb
  mkTensor outVal [a, b] $ \outGradRef -> do
    outGrad <- readIORef outGradRef
    addGrad a (outGrad <> tr vb)
    addGrad b (tr va <> outGrad)

rocblasGemmHost :: Mat -> Mat -> IO Mat
rocblasGemmHost a b = do
  let ar = rows a
  let ac = cols a
  let br = rows b
  let bc = cols b
  when (ac /= br) $ error "rocblasGemmHost: shape mismatch"
  let m = ar
  let n = bc
  let k = ac
  let aVec = flatten a
  let bVec = flatten b
  outFp <- (mallocForeignPtrArray (m * n) :: IO (ForeignPtr Double))
  withForeignPtr outFp $ \outPtr ->
    VS.unsafeWith aVec $ \aPtr ->
      VS.unsafeWith bVec $ \bPtr -> do
        rc <- c_rocblas_dgemm_host
                0
                0
                (fromIntegral m)
                (fromIntegral n)
                (fromIntegral k)
                (castPtr aPtr)
                (fromIntegral ar)
                (castPtr bPtr)
                (fromIntegral br)
                (castPtr outPtr)
                (fromIntegral m)
        when (rc /= 0) $ error ("rocblas_dgemm_host failed: " ++ show rc)
  let outVec = VS.unsafeFromForeignPtr0 outFp (m * n)
  pure (reshape n outVec)

sinT :: Tensor -> IO Tensor
sinT a = do
  va <- value a
  let outVal = cmap sin va
  mkTensor outVal [a] $ \outGradRef -> do
    outGrad <- readIORef outGradRef
    addGrad a (hadamard outGrad (cmap cos va))

cosT :: Tensor -> IO Tensor
cosT a = do
  va <- value a
  let outVal = cmap cos va
  mkTensor outVal [a] $ \outGradRef -> do
    outGrad <- readIORef outGradRef
    addGrad a (hadamard outGrad (cmap (negate . sin) va))

relu :: Tensor -> IO Tensor
relu a = do
  va <- value a
  let outVal = cmap (max 0) va
  mkTensor outVal [a] $ \outGradRef -> do
    outGrad <- readIORef outGradRef
    let mask = cmap (\x -> if x > 0 then 1 else 0) va
    addGrad a (hadamard outGrad mask)

addBias :: Tensor -> Tensor -> IO Tensor
addBias a b = do
  va <- value a
  vb <- value b
  when (cols vb /= 1) $ error "addBias: bias must be a column"
  when (rows va /= rows vb) $ error "addBias: shape mismatch"
  let biasMat = repeatColumns (head (toColumns vb)) (cols va)
  let outVal = va + biasMat
  mkTensor outVal [a, b] $ \outGradRef -> do
    outGrad <- readIORef outGradRef
    addGrad a outGrad
    addGrad b (asColumn (sumColumns outGrad))

concatRows :: [Tensor] -> IO Tensor
concatRows [] = error "concatRows: empty list"
concatRows ts = do
  vals <- mapM value ts
  let colCount = cols (head vals)
  mapM_ (\v -> when (rows v /= 1) (error "concatRows: each tensor must be 1 x n")) vals
  mapM_ (\v -> when (cols v /= colCount) (error "concatRows: column mismatch")) vals
  let outVal = fromLists (concatMap toLists vals)
  mkTensor outVal ts $ \outGradRef -> do
    outGrad <- readIORef outGradRef
    let outRows = toLists outGrad
    forM_ (zip ts outRows) $ \(t, rowVals) -> do
      let rowMat = (1 >< colCount) rowVals
      addGrad t rowMat

transposeT :: Tensor -> IO Tensor
transposeT a = do
  va <- value a
  mkTensor (tr va) [a] $ \outGradRef -> do
    outGrad <- readIORef outGradRef
    addGrad a (tr outGrad)

softmax :: Tensor -> IO Tensor
softmax a = do
  va <- value a
  let colsV = toColumns va
  let (outCols, softCols) = unzip (map softmaxVec colsV)
  let outMat = fromColumns outCols
  mkTensor outMat [a] $ \outGradRef -> do
    outGrad <- readIORef outGradRef
    let gradCols = toColumns outGrad
    let dxCols = zipWith softmaxBackward softCols gradCols
    addGrad a (fromColumns dxCols)

softmaxRows :: Tensor -> IO Tensor
softmaxRows a = do
  t <- transposeT a
  s <- softmax t
  transposeT s

logSoftmax :: Tensor -> IO Tensor
logSoftmax a = do
  va <- value a
  let colsV = toColumns va
  let (outCols, softCols) = unzip (map logSoftmaxVec colsV)
  let outMat = fromColumns outCols
  mkTensor outMat [a] $ \outGradRef -> do
    outGrad <- readIORef outGradRef
    let gradCols = toColumns outGrad
    let dxCols = zipWith logSoftmaxBackward softCols gradCols
    addGrad a (fromColumns dxCols)

logSoftmaxVec :: Vector Double -> (Vector Double, Vector Double)
logSoftmaxVec v =
  let xs = LA.toList v
      maxX = maximum xs
      shifted = map (\x -> x - maxX) xs
      exps = map exp shifted
      denom = sum exps
      logZ = log denom
      out = map (\x -> x - logZ) shifted
      soft = map (/ denom) exps
  in (LA.fromList out, LA.fromList soft)

softmaxVec :: Vector Double -> (Vector Double, Vector Double)
softmaxVec v =
  let xs = LA.toList v
      maxX = maximum xs
      shifted = map (\x -> x - maxX) xs
      exps = map exp shifted
      denom = sum exps
      soft = map (/ denom) exps
  in (LA.fromList soft, LA.fromList soft)

softmaxBackward :: Vector Double -> Vector Double -> Vector Double
softmaxBackward soft g =
  let gs = LA.toList g
      ss = LA.toList soft
      dotGS = sum (zipWith (*) gs ss)
      dx = zipWith (\s gi -> s * (gi - dotGS)) ss gs
  in LA.fromList dx

logSoftmaxBackward :: Vector Double -> Vector Double -> Vector Double
logSoftmaxBackward soft g =
  let gs = LA.toList g
      ss = LA.toList soft
      sumG = sum gs
      dx = zipWith (\s gi -> gi - s * sumG) ss gs
  in LA.fromList dx

crossEntropy :: Tensor -> Tensor -> IO Tensor
crossEntropy logits target = do
  logp <- logSoftmax logits
  prod <- mul target logp
  summed <- sumAll prod
  v <- value logits
  let batch = max 1 (cols v)
  scale ((-1) / fromIntegral batch) summed

sumAll :: Tensor -> IO Tensor
sumAll a = do
  va <- value a
  let outVal = (1 >< 1) [sumElements va]
  mkTensor outVal [a] $ \outGradRef -> do
    outGrad <- readIORef outGradRef
    let g = konst (outGrad `atIndex` (0, 0)) (rows va, cols va)
    addGrad a g

meanAll :: Tensor -> IO Tensor
meanAll a = do
  va <- value a
  let denom = fromIntegral (rows va * cols va)
  s <- sumAll a
  scale (1 / denom) s

scale :: Double -> Tensor -> IO Tensor
scale c a = do
  va <- value a
  mkTensor (LA.scale c va) [a] $ \outGradRef -> do
    outGrad <- readIORef outGradRef
    addGrad a (LA.scale c outGrad)

-- Backprop.

backward :: Tensor -> IO ()
backward t = do
  topo <- topoSort t
  forM_ topo $ \n -> do
    v <- value n
    writeIORef (tGrad n) (zerosLike v)
  v <- value t
  writeIORef (tGrad t) (konst 1 (rows v, cols v))
  forM_ (reverse topo) tBackward

backwardAccum :: Tensor -> IO ()
backwardAccum t = do
  topo <- topoSort t
  v <- value t
  addGrad t (konst 1 (rows v, cols v))
  forM_ (reverse topo) tBackward

-- Topological order (parents before children).

topoSort :: Tensor -> IO [Tensor]
topoSort t = do
  visitedRef <- newIORef Set.empty
  orderRef <- newIORef []
  let dfs node = do
        let key = hashUnique (tId node)
        visited <- readIORef visitedRef
        if Set.member key visited
          then pure ()
          else do
            modifyIORef' visitedRef (Set.insert key)
            mapM_ dfs (tParents node)
            modifyIORef' orderRef (node:)
  dfs t
  order <- readIORef orderRef
  pure (reverse order)
