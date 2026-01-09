{-# LANGUAGE BangPatterns #-}
module NNet where

import Codec.Compression.GZip (decompress)
import Control.Monad (replicateM, when, zipWithM)
import Data.Bits (shiftL, (.|.))
import Data.List (foldl', maximumBy)
import Data.Ord (comparing)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import System.Random (mkStdGen, randomIO, setStdGen)

import Numeric.LinearAlgebra (Matrix, Vector, (#>), (<>), asColumn, asRow, cmap,
                              fromList, fromLists, scale, toList, toLists, tr, (><))

-- Network types

data Layer = Layer
  { layerBias :: !(Vector Double)
  , layerWeights :: !(Matrix Double)
  }

newtype Net = Net
  { netLayers :: [Layer]
  }

type Brain = Net

-- Config types

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

-- IDX dataset types

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

-- Box–Muller transform
bmt :: Double -> IO Double
bmt scale = do
  x1 <- randomIO
  x2 <- randomIO
  let x1' = if x1 < 1.0e-12 then 1.0e-12 else x1
  pure $ scale * sqrt (-2 * log x1') * cos (2 * pi * x2)

newBrain :: [Int] -> IO Brain
newBrain sizes = do
  when (length sizes < 2) $ error "newBrain: need at least input and output sizes"
  layers <- zipWithM mkLayer sizes (tail sizes)
  pure $ Net layers
  where
    mkLayer inSize outSize = do
      b <- randomVector 0.01 outSize
      w <- randomMatrix 0.01 outSize inSize
      pure $ Layer b w
    randomVector scale size = fromList <$> replicateM size (bmt scale)
    randomMatrix scale rows cols = do
      xs <- replicateM (rows * cols) (bmt scale)
      pure $ (rows >< cols) xs

sigmoidSimple :: Double -> Double
sigmoidSimple x
  | x < 0 = 0
  | otherwise = 1

pushThroughLayer :: Vector Double -> Layer -> Vector Double
pushThroughLayer as (Layer bs wvs) = (wvs #> as) + bs

feedVec :: Vector Double -> Brain -> Vector Double
feedVec input (Net layers) = foldl' step input layers
  where
    step !as layer = cmap (max 0) (pushThroughLayer as layer)

feed :: [Double] -> Brain -> [Double]
feed xs net = toList $ feedVec (fromList xs) net

-- Dataset helpers
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

imageSize :: IdxImages -> Int
imageSize imgs = idxRows imgs * idxCols imgs

imagePixels :: IdxImages -> Int -> [Int]
imagePixels imgs n = fromIntegral <$> BS.unpack slice
  where
    size = imageSize imgs
    start = n * size
    slice = BS.take size (BS.drop start (idxImageBytes imgs))

imageVector :: IdxImages -> Int -> Vector Double
imageVector imgs n = fromList $ (/ 256) . fromIntegral <$> imagePixels imgs n

labelValue :: IdxLabels -> Int -> Int
labelValue lbls n = fromIntegral $ BS.index (lblBytes lbls) n

labelVector :: Int -> IdxLabels -> Int -> Vector Double
labelVector classes lbls n =
  let target = labelValue lbls n
  in fromList $ fromIntegral . fromEnum . (target ==) <$> [0 .. classes - 1]

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

-- Training and evaluation
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
  when (samples <= 0) $
    error "train: sample count must be positive"
  let classes = last (trainLayers cfg)
  withSeed (trainSeed cfg) $ do
    net0 <- newBrain (trainLayers cfg)
    let epochs = max 0 (trainEpochs cfg)
    let lr = trainLearningRate cfg
    pure $ foldl' (trainEpoch imgs lbls samples classes lr) net0 [1 .. epochs]

trainEpoch :: IdxImages -> IdxLabels -> Int -> Int -> Double -> Brain -> Int -> Brain
trainEpoch imgs lbls samples classes lr net _ = foldl' step net [0 .. samples - 1]
  where
    step !acc i = trainSample lr (imageVector imgs i) (labelVector classes lbls i) acc

trainSample :: Double -> Vector Double -> Vector Double -> Brain -> Brain
trainSample lr input target (Net layers) = Net updated
  where
    (activations, zs) = forwardPass input layers
    deltaLast = hadamard (costVec (last activations) target) (stepVec (last zs))
    deltas = backpropDeltas deltaLast (reverse layers) (reverse zs)
    updated = zipWith3 (updateLayer lr) layers deltas (init activations)

forwardPass :: Vector Double -> [Layer] -> ([Vector Double], [Vector Double])
forwardPass input layers = (reverse actsRev, reverse zsRev)
  where
    (actsRev, zsRev) = foldl' step ([input], []) layers
    step (aPrev:as, zs) layer =
      let z = pushThroughLayer aPrev layer
          a = cmap (max 0) z
      in (a:aPrev:as, z:zs)
    step _ _ = error "forwardPass: invalid activation state"

backpropDeltas :: Vector Double -> [Layer] -> [Vector Double] -> [Vector Double]
backpropDeltas deltaLast revLayers revZs =
  case (revLayers, revZs) of
    (layerLast:layersRest, _zLast:zsRest) ->
      let (_, deltas) = foldl' step (layerLast, [deltaLast]) (zip layersRest zsRest)
      in deltas
    _ -> error "backpropDeltas: empty network"
  where
    step (nextLayer, deltas@(deltaNext:_)) (layer, z) =
      let wNext = layerWeights nextLayer
          delta = hadamard (tr wNext #> deltaNext) (stepVec z)
      in (layer, delta:deltas)
    step _ _ = error "backpropDeltas: invalid delta state"

updateLayer :: Double -> Layer -> Vector Double -> Vector Double -> Layer
updateLayer lr (Layer b w) delta aPrev = Layer b' w'
  where
    !b' = b - scale lr delta
    !w' = w - scale lr (asColumn delta <> asRow aPrev)

stepVec :: Vector Double -> Vector Double
stepVec = cmap sigmoidSimple

costVec :: Vector Double -> Vector Double -> Vector Double
costVec a y = zipVectorWithV cost a y

hadamard :: Vector Double -> Vector Double -> Vector Double
hadamard = zipVectorWithV (*)

zipVectorWithV :: (Double -> Double -> Double) -> Vector Double -> Vector Double -> Vector Double
zipVectorWithV f a b = fromList $ zipWith f (toList a) (toList b)

cost :: Double -> Double -> Double
cost a y
  | y == 1 && a >= y = 0
  | otherwise = a - y

predict :: Brain -> Vector Double -> Int
predict net v = fst $ maximumBy (comparing snd) $ zip [0 ..] (toList (feedVec v net))

accuracy :: Brain -> IdxImages -> IdxLabels -> Int -> Int
accuracy net imgs lbls samples = (correct * 100) `div` samples
  where
    guesses = predict net . imageVector imgs <$> [0 .. samples - 1]
    answers = labelValue lbls <$> [0 .. samples - 1]
    correct = sum (fromEnum <$> zipWith (==) guesses answers)

testNN :: TestConfig -> Brain -> IO Int
testNN cfg net = do
  imgs <- readIdxImages (testImages cfg)
  lbls <- readIdxLabels (testLabels cfg)
  let total = min (idxCount imgs) (lblCount lbls)
  let samples = min total (maybe total id (testSamples cfg))
  when (samples <= 0) $
    error "testNN: sample count must be positive"
  pure $ accuracy net imgs lbls samples

writeNNToFile :: FilePath -> Brain -> IO ()
writeNNToFile fName net = writeFile fName (show (netToLists net))

readNNFile :: FilePath -> IO Brain
readNNFile fName = do
  sNet <- readFile fName
  let net = netFromLists (read sNet :: [([Double], [[Double]])])
  pure net

netToLists :: Brain -> [([Double], [[Double]])]
netToLists (Net layers) = [ (toList b, toLists w) | Layer b w <- layers ]

netFromLists :: [([Double], [[Double]])] -> Brain
netFromLists xs = Net [ Layer (fromList b) (fromLists w) | (b, w) <- xs ]

withSeed :: Maybe Int -> IO a -> IO a
withSeed Nothing action = action
withSeed (Just seed) action = do
  setStdGen (mkStdGen seed)
  action

num2Str :: Integral a => a -> Char
num2Str n = let i = fromIntegral n * 2 `div` 256 in head (show i)
