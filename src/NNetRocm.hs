{-# LANGUAGE BangPatterns #-}
module NNetRocm where

-- GPU-focused backend that mirrors the NNet API.
-- This version uses ROCm/rocBLAS for matrix multiplies and keeps element-wise
-- ops on CPU for clarity and portability.

import Codec.Compression.GZip (decompress)
import Control.Monad (foldM, replicateM, when, zipWithM)
import Data.Bits (shiftL, (.|.))
import Data.List (foldl', maximumBy)
import Data.Ord (comparing)
import Foreign (ForeignPtr, Ptr, castPtr, mallocForeignPtrArray, withForeignPtr)
import Foreign.C.Types (CDouble(..), CInt(..))
import Foreign.ForeignPtr (castForeignPtr)
import System.Environment (lookupEnv)
import System.Random (mkStdGen, randomIO, setStdGen)
import Text.Read (readMaybe)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector.Storable as VS

-- FFI to a small rocBLAS wrapper.
foreign import ccall unsafe "rocblas_dgemm_host"
  c_rocblas_dgemm_host :: CInt -> CInt -> CInt -> CInt -> CInt
                       -> Ptr CDouble -> CInt
                       -> Ptr CDouble -> CInt
                       -> Ptr CDouble -> CInt
                       -> IO CInt

-- Network types.

data Layer = Layer
  { layerBias :: !(VS.Vector Double)
  , layerWeights :: !Mat
  }

newtype Brain = Brain
  { brainLayers :: [Layer]
  }

-- Config types (same shape as NNet).

data TrainConfig = TrainConfig
  { trainImages :: FilePath
  , trainLabels :: FilePath
  , trainLayers :: [Int]
  , trainLearningRate :: Double
  , trainEpochs :: Int
  , trainSamples :: Maybe Int
  , trainSeed :: Maybe Int
  }

data TestConfig = TestConfig
  { testImages :: FilePath
  , testLabels :: FilePath
  , testSamples :: Maybe Int
  }

-- IDX dataset types.

data IdxImages = IdxImages
  { idxCount :: !Int
  , idxRows :: !Int
  , idxCols :: !Int
  , idxImageBytes :: !BS.ByteString
  }

data IdxLabels = IdxLabels
  { lblCount :: !Int
  , lblBytes :: !BS.ByteString
  }

-- Matrix representation: column-major data.

data Mat = Mat
  { matRows :: !Int
  , matCols :: !Int
  , matData :: !(VS.Vector Double)
  }

-- Box-Muller transform for normal-ish initialization.

bmt :: Double -> IO Double
bmt scale = do
  x1 <- randomIO
  x2 <- randomIO
  let x1' = if x1 < 1.0e-12 then 1.0e-12 else x1
  pure $ scale * sqrt (-2 * log x1') * cos (2 * pi * x2)

-- Initialize a network from layer sizes.

newBrain :: [Int] -> IO Brain
newBrain sizes = do
  when (length sizes < 2) $ error "newBrain: need at least input and output sizes"
  layers <- zipWithM mkLayer sizes (tail sizes)
  pure $ Brain layers
  where
    mkLayer inSize outSize = do
      b <- randomVector 0.01 outSize
      w <- randomMatrix 0.01 outSize inSize
      pure $ Layer b w
    randomVector scale size = VS.fromList <$> replicateM size (bmt scale)
    randomMatrix scale rows cols = do
      xs <- replicateM (rows * cols) (bmt scale)
      pure $ Mat rows cols (VS.fromList xs)

-- Basic activation and helpers.

relu :: Double -> Double
relu x
  | x < 0 = 0
  | otherwise = 1

reluMat :: Mat -> Mat
reluMat = matMap (max 0)

reluStepMat :: Mat -> Mat
reluStepMat = matMap relu

-- Matrix helpers (CPU side).

matMap :: (Double -> Double) -> Mat -> Mat
matMap f (Mat r c v) = Mat r c (VS.map f v)

matZipWith :: (Double -> Double -> Double) -> Mat -> Mat -> Mat
matZipWith f (Mat r c a) (Mat r' c' b)
  | r /= r' || c /= c' =
      error ("matZipWith: shape mismatch (" ++ show r ++ "x" ++ show c ++
             " vs " ++ show r' ++ "x" ++ show c' ++ ")")
  | otherwise = Mat r c (VS.zipWith f a b)

matScale :: Double -> Mat -> Mat
matScale s (Mat r c v) = Mat r c (VS.map (s *) v)

matAdd :: Mat -> Mat -> Mat
matAdd = matZipWith (+)

matSub :: Mat -> Mat -> Mat
matSub = matZipWith (-)

matAddBias :: Mat -> VS.Vector Double -> Mat
matAddBias (Mat r c v) bias
  | VS.length bias /= r = error "matAddBias: bias length mismatch"
  | otherwise = Mat r c $ VS.generate (r * c) $ \idx ->
      let i = idx `mod` r
      in v VS.! idx + bias VS.! i

matSumColumns :: Mat -> VS.Vector Double
matSumColumns (Mat r c v) = VS.generate r $ \i ->
  let go j acc
        | j == c = acc
        | otherwise = go (j + 1) (acc + v VS.! (i + j * r))
  in go 0 0

matTranspose :: Mat -> Mat
matTranspose (Mat r c v) = Mat c r $ VS.generate (r * c) $ \idx ->
  let i = idx `mod` c
      j = idx `div` c
  in v VS.! (j + i * r)

matFromLists :: [[Double]] -> Mat
matFromLists rows
  | null rows = Mat 0 0 VS.empty
  | any ((/= c) . length) rows = error "matFromLists: ragged rows"
  | otherwise = Mat r c (VS.fromList colMajor)
  where
    r = length rows
    c = length (head rows)
    colMajor = [ rows !! i !! j | j <- [0 .. c - 1], i <- [0 .. r - 1] ]

matToLists :: Mat -> [[Double]]
matToLists (Mat r c v) =
  [ [ v VS.! (i + j * r) | j <- [0 .. c - 1] ] | i <- [0 .. r - 1] ]

matFromColumns :: Int -> [VS.Vector Double] -> Mat
matFromColumns r cols
  | null cols = Mat r 0 VS.empty
  | any ((/= r) . VS.length) cols = error "matFromColumns: column size mismatch"
  | otherwise = Mat r c (VS.concat cols)
  where
    c = length cols

matToVector :: Mat -> VS.Vector Double
matToVector (Mat r c v)
  | c < 1 = VS.empty
  | otherwise = VS.slice 0 r v

vecScale :: Double -> VS.Vector Double -> VS.Vector Double
vecScale s v = VS.map (s *) v

vecSub :: VS.Vector Double -> VS.Vector Double -> VS.Vector Double
vecSub = VS.zipWith (-)

-- Naive CPU matrix multiplication (used for feed/inference).

matMulCPU :: Mat -> Mat -> Mat
matMulCPU (Mat ar ac av) (Mat br bc bv)
  | ac /= br = error "matMulCPU: shape mismatch"
  | otherwise = Mat ar bc $ VS.generate (ar * bc) $ \idx ->
      let i = idx `mod` ar
          j = idx `div` ar
          go k acc
            | k == ac = acc
            | otherwise =
                let aVal = av VS.! (i + k * ar)
                    bVal = bv VS.! (k + j * br)
                in go (k + 1) (acc + aVal * bVal)
      in go 0 0

-- GPU-backed matrix multiplication using rocBLAS.

data Transpose = NoTrans | Trans
  deriving (Eq)

matMulGpu :: Transpose -> Transpose -> Mat -> Mat -> IO Mat
matMulGpu transA transB a@(Mat ar ac _) b@(Mat br bc _) = do
  let (aRowsEff, aColsEff) = if transA == NoTrans then (ar, ac) else (ac, ar)
  let (bRowsEff, bColsEff) = if transB == NoTrans then (br, bc) else (bc, br)
  when (aColsEff /= bRowsEff) $ error "matMulGpu: shape mismatch"
  let m = aRowsEff
  let n = bColsEff
  let k = aColsEff
  outFp <- (mallocForeignPtrArray (m * n) :: IO (ForeignPtr CDouble))
  let lda = ar
  let ldb = br
  let ldc = m
  withForeignPtr outFp $ \outPtr ->
    VS.unsafeWith (matData a) $ \aPtr ->
      VS.unsafeWith (matData b) $ \bPtr -> do
        let ta = if transA == NoTrans then 0 else 1
        let tb = if transB == NoTrans then 0 else 1
        rc <- c_rocblas_dgemm_host
                (fromIntegral ta)
                (fromIntegral tb)
                (fromIntegral m)
                (fromIntegral n)
                (fromIntegral k)
                (castPtr aPtr)
                (fromIntegral lda)
                (castPtr bPtr)
                (fromIntegral ldb)
                (castPtr outPtr)
                (fromIntegral ldc)
        when (rc /= 0) $ error "matMulGpu: rocBLAS call failed"
  let outVec = VS.unsafeFromForeignPtr0 (castForeignPtr outFp) (m * n)
  pure $ Mat m n outVec

-- Forward pass through one layer (CPU version).

pushThroughLayer :: Mat -> Layer -> Mat
pushThroughLayer as (Layer bs wvs) =
  let z = matMulCPU wvs as
  in matAddBias z bs

-- Forward pass through the full network (CPU).

feedMat :: Mat -> Brain -> Mat
feedMat input (Brain layers) = foldl' step input layers
  where
    step !as layer = reluMat (pushThroughLayer as layer)

feed :: [Double] -> Brain -> [Double]
feed xs net = VS.toList $ matToVector $ feedMat input net
  where
    input = Mat (length xs) 1 (VS.fromList xs)

-- Dataset helpers.

readIdxImages :: FilePath -> IO IdxImages
readIdxImages path = do
  raw <- BL.readFile path
  let bs = BL.toStrict (decompress raw)
  case parseIdxImages bs of
    Left err -> error err
    Right val -> pure val

readIdxLabels :: FilePath -> IO IdxLabels
readIdxLabels path = do
  raw <- BL.readFile path
  let bs = BL.toStrict (decompress raw)
  case parseIdxLabels bs of
    Left err -> error err
    Right val -> pure val

parseIdxImages :: BS.ByteString -> Either String IdxImages
parseIdxImages bs
  | BS.length bs < 16 = Left "parseIdxImages: file too small"
  | magic /= 2051 = Left "parseIdxImages: bad magic"
  | BS.length bs < expected = Left "parseIdxImages: truncated data"
  | otherwise = Right $ IdxImages count rows cols (BS.drop 16 bs)
  where
    magic = getInt32BE bs 0
    count = getInt32BE bs 4
    rows = getInt32BE bs 8
    cols = getInt32BE bs 12
    expected = 16 + count * rows * cols

parseIdxLabels :: BS.ByteString -> Either String IdxLabels
parseIdxLabels bs
  | BS.length bs < 8 = Left "parseIdxLabels: file too small"
  | magic /= 2049 = Left "parseIdxLabels: bad magic"
  | BS.length bs < expected = Left "parseIdxLabels: truncated data"
  | otherwise = Right $ IdxLabels count (BS.drop 8 bs)
  where
    magic = getInt32BE bs 0
    count = getInt32BE bs 4
    expected = 8 + count

getInt32BE :: BS.ByteString -> Int -> Int
getInt32BE bs off =
  (b0 `shiftL` 24) .|. (b1 `shiftL` 16) .|. (b2 `shiftL` 8) .|. b3
  where
    b0 = fromIntegral (BS.index bs off) :: Int
    b1 = fromIntegral (BS.index bs (off + 1)) :: Int
    b2 = fromIntegral (BS.index bs (off + 2)) :: Int
    b3 = fromIntegral (BS.index bs (off + 3)) :: Int

imageSize :: IdxImages -> Int
imageSize imgs = idxRows imgs * idxCols imgs

imageVector :: IdxImages -> Int -> VS.Vector Double
imageVector imgs n = VS.fromList $ (/ 256) . fromIntegral <$> pixels
  where
    size = imageSize imgs
    start = n * size
    pixels = BS.unpack $ BS.take size (BS.drop start (idxImageBytes imgs))

labelValue :: IdxLabels -> Int -> Int
labelValue lbls n = fromIntegral $ BS.index (lblBytes lbls) n

labelVector :: Int -> IdxLabels -> Int -> VS.Vector Double
labelVector classes lbls n =
  let target = labelValue lbls n
  in VS.fromList $ fromIntegral . fromEnum . (target ==) <$> [0 .. classes - 1]

-- Core math for training.

cost :: Double -> Double -> Double
cost a y
  | y == 1 && a >= y = 0
  | otherwise = a - y

costMat :: Mat -> Mat -> Mat
costMat = matZipWith cost

hadamard :: Mat -> Mat -> Mat
hadamard = matZipWith (*)

-- Batch forward pass using GPU matmul.

forwardPassBatch :: Mat -> [Layer] -> IO ([Mat], [Mat])
forwardPassBatch input layers = do
  (actsRev, zsRev) <- foldM step ([input], []) layers
  pure (reverse actsRev, reverse zsRev)
  where
    step (aPrev:as, zs) layer = do
      z <- matMulGpu NoTrans NoTrans (layerWeights layer) aPrev
      let zBias = matAddBias z (layerBias layer)
      let a = reluMat zBias
      pure (a:aPrev:as, zBias:zs)
    step _ _ = error "forwardPassBatch: invalid activation state"

backpropDeltas :: Mat -> [Layer] -> [Mat] -> IO [Mat]
backpropDeltas deltaLast revLayers revZs =
  case (revLayers, revZs) of
    (layerLast:layersRest, _zLast:zsRest) -> do
      (_, deltasRev) <- foldM step (layerLast, [deltaLast]) (zip layersRest zsRest)
      pure deltasRev
    _ -> error "backpropDeltas: empty network"
  where
    step (nextLayer, deltas@(deltaNext:_)) (layer, z) = do
      deltaBase <- matMulGpu Trans NoTrans (layerWeights nextLayer) deltaNext
      let delta = hadamard deltaBase (reluStepMat z)
      pure (layer, delta:deltas)
    step _ _ = error "backpropDeltas: invalid delta state"

updateLayer :: Double -> Double -> Layer -> Mat -> Mat -> IO Layer
updateLayer lr invBatch (Layer b w) delta aPrev = do
  dW <- matMulGpu NoTrans Trans delta aPrev
  let dW' = matScale invBatch dW
  let db = vecScale invBatch (matSumColumns delta)
  let w' = matSub w (matScale lr dW')
  let b' = vecSub b (vecScale lr db)
  pure $ Layer b' w'

trainBatch :: Double -> Mat -> Mat -> Brain -> IO Brain
trainBatch lr input target (Brain layers) = do
  (activations, zs) <- forwardPassBatch input layers
  let deltaLast = hadamard (costMat (last activations) target) (reluStepMat (last zs))
  deltas <- backpropDeltas deltaLast (reverse layers) (reverse zs)
  let invBatch = 1 / fromIntegral (matCols input)
  let updates = zipWith3 (updateLayer lr invBatch) layers deltas (init activations)
  updated <- sequence updates
  pure $ Brain updated

-- Training entrypoint.

train :: TrainConfig -> IO Brain
train cfg = do
  imgs <- readIdxImages (trainImages cfg)
  lbls <- readIdxLabels (trainLabels cfg)
  when (length (trainLayers cfg) < 2) $
    error "train: trainLayers must include input and output sizes"
  let inputSize = imageSize imgs
  let expectedInput = head (trainLayers cfg)
  when (inputSize /= expectedInput) $
    error "train: input size mismatch with trainLayers"
  when (idxCount imgs /= lblCount lbls) $
    error "train: image/label counts do not match"
  let total = idxCount imgs
  let samples = min total (maybe total id (trainSamples cfg))
  when (samples <= 0) $ error "train: sample count must be positive"
  let classes = last (trainLayers cfg)
  batchSize <- readBatchSize samples
  withSeed (trainSeed cfg) $ do
    net0 <- newBrain (trainLayers cfg)
    let epochs = max 0 (trainEpochs cfg)
    let lr = trainLearningRate cfg
    foldM (trainEpoch imgs lbls samples classes lr batchSize) net0 [1 .. epochs]

trainEpoch :: IdxImages -> IdxLabels -> Int -> Int -> Double -> Int -> Brain -> Int -> IO Brain
trainEpoch imgs lbls samples classes lr batchSize net _ =
  foldM step net [0, batchSize .. samples - 1]
  where
    step !acc start = do
      let end = min samples (start + batchSize)
      let count = end - start
      let xs = [ imageVector imgs i | i <- [start .. end - 1] ]
      let ys = [ labelVector classes lbls i | i <- [start .. end - 1] ]
      let xMat = matFromColumns (imageSize imgs) xs
      let yMat = matFromColumns classes ys
      when (count <= 0) $ error "trainEpoch: empty batch"
      trainBatch lr xMat yMat acc

readBatchSize :: Int -> IO Int
readBatchSize samples = do
  env <- lookupEnv "NNET_BATCH_SIZE"
  case env >>= readMaybe of
    Just n | n > 0 -> pure (min samples n)
    _ -> pure (min samples 128)

-- Evaluation helpers (CPU for simplicity).

predict :: Brain -> VS.Vector Double -> Int
predict net v = fst $ maximumBy (comparing snd) $ zip [0 ..] scores
  where
    scores = VS.toList $ matToVector $ feedMat (Mat (VS.length v) 1 v) net

accuracy :: Brain -> IdxImages -> IdxLabels -> Int -> Int
accuracy net imgs lbls samples = (correct * 100) `div` samples
  where
    guesses = predict net . imageVector imgs <$> [0 .. samples - 1]
    answers = labelValue lbls <$> [0 .. samples - 1]
    correct = sum (fromEnum <$> zipWith (==) guesses answers)

-- Testing entrypoint.

testNN :: TestConfig -> Brain -> IO Int
testNN cfg net = do
  imgs <- readIdxImages (testImages cfg)
  lbls <- readIdxLabels (testLabels cfg)
  let total = min (idxCount imgs) (lblCount lbls)
  let samples = min total (maybe total id (testSamples cfg))
  when (samples <= 0) $ error "testNN: sample count must be positive"
  pure $ accuracy net imgs lbls samples

-- Simple text serialization (same format as NNet).

writeNNToFile :: FilePath -> Brain -> IO ()
writeNNToFile fName net = writeFile fName (show (netToLists net))

readNNFile :: FilePath -> IO Brain
readNNFile fName = do
  sNet <- readFile fName
  let net = netFromLists (read sNet :: [([Double], [[Double]])])
  pure net

netToLists :: Brain -> [([Double], [[Double]])]
netToLists (Brain layers) =
  [ (VS.toList b, matToLists w) | Layer b w <- layers ]

netFromLists :: [([Double], [[Double]])] -> Brain
netFromLists xs = Brain [ Layer (VS.fromList b) (matFromLists w) | (b, w) <- xs ]

-- Deterministic initialization when a seed is provided.

withSeed :: Maybe Int -> IO a -> IO a
withSeed Nothing action = action
withSeed (Just seed) action = do
  setStdGen (mkStdGen seed)
  action
