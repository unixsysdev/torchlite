module TorchLite.Grokking
  ( Config(..)
  , Backend(..)
  , parseArgs
  , runGrokking
  ) where

import Control.Monad (foldM, forM_)
import Data.IORef (newIORef, readIORef, writeIORef)
import Data.List (foldl', sortOn)
import System.Directory (createDirectoryIfMissing)
import System.FilePath ((</>))
import System.Random (mkStdGen, randomRs)
import Text.Read (readMaybe)

import Numeric.LinearAlgebra
  ( Matrix
  , Vector
  , (><)
  , atIndex
  , fromColumns
  , fromList
  , toColumns
  , toList
  )

import TorchLite.Backend (Ops(..))
import TorchLite.Backend.CPU (cpuOps)
import TorchLite.Backend.ROCm (rocmOps)
import TorchLite.Optim (adamWStep, initAdamW)
import qualified TorchLite.Optim.ROCm as ROCmOptim
import TorchLite.Runtime (applyThreads)

data Backend = BackendCPU | BackendROCm
  deriving (Show)

data Config = Config
  { modulus :: Int
  , trainFrac :: Double
  , epochs :: Int
  , batchSize :: Int
  , lr :: Double
  , lrScale :: Double
  , weightDecay :: Double
  , backend :: Backend
  , dModel :: Int
  , heads :: Int
  , dMlp :: Int
  , seed :: Int
  , evalEvery :: Int
  , threads :: Maybe Int
  , logDir :: FilePath
  } deriving (Show)

parseArgs :: [String] -> Config
parseArgs = go defaultCfg
  where
    defaultCfg = Config
      { modulus = 251
      , trainFrac = 0.3
      , epochs = 40000
      , batchSize = 2048
      , lr = 0.001
      , lrScale = 1.0
      , weightDecay = 1.0
      , backend = BackendCPU
      , dModel = 128
      , heads = 4
      , dMlp = 512
      , seed = 1
      , evalEvery = 100
      , threads = Nothing
      , logDir = "analysis"
      }
    go cfg [] = cfg
    go cfg ("--modulus":v:rest) = go cfg { modulus = readInt v (modulus cfg) } rest
    go cfg ("--train-frac":v:rest) = go cfg { trainFrac = readDouble v (trainFrac cfg) } rest
    go cfg ("--epochs":v:rest) = go cfg { epochs = readInt v (epochs cfg) } rest
    go cfg ("--batch-size":v:rest) = go cfg { batchSize = readInt v (batchSize cfg) } rest
    go cfg ("--lr":v:rest) = go cfg { lr = readDouble v (lr cfg) } rest
    go cfg ("--lr-scale":v:rest) = go cfg { lrScale = readDouble v (lrScale cfg) } rest
    go cfg ("--weight-decay":v:rest) = go cfg { weightDecay = readDouble v (weightDecay cfg) } rest
    go cfg ("--backend":v:rest) = go cfg { backend = readBackend v (backend cfg) } rest
    go cfg ("--d-model":v:rest) = go cfg { dModel = readInt v (dModel cfg) } rest
    go cfg ("--heads":v:rest) = go cfg { heads = readInt v (heads cfg) } rest
    go cfg ("--d-mlp":v:rest) = go cfg { dMlp = readInt v (dMlp cfg) } rest
    go cfg ("--seed":v:rest) = go cfg { seed = readInt v (seed cfg) } rest
    go cfg ("--eval-every":v:rest) = go cfg { evalEvery = readInt v (evalEvery cfg) } rest
    go cfg ("--threads":v:rest) = go cfg { threads = Just (readInt v 0) } rest
    go cfg ("--log-dir":v:rest) = go cfg { logDir = v } rest
    go cfg (_:rest) = go cfg rest

readInt :: String -> Int -> Int
readInt s def = maybe def id (readMaybe s)

readDouble :: String -> Double -> Double
readDouble s def = maybe def id (readMaybe s)

readBackend :: String -> Backend -> Backend
readBackend s def =
  case s of
    "cpu" -> BackendCPU
    "rocm" -> BackendROCm
    _ -> def

data OptimOps t opt = OptimOps
  { optimInit :: Double -> Double -> Double -> Double -> Double -> [t] -> IO opt
  , optimStep :: opt -> IO ()
  }

runGrokking :: Config -> IO ()
runGrokking cfg = do
  applyThreads (threads cfg)
  case backend cfg of
    BackendCPU -> runWithOps cpuOps cpuOptimOps cfg
    BackendROCm -> runWithOps rocmOps rocmOptimOps cfg
  where
    cpuOptimOps = OptimOps initAdamW adamWStep
    rocmOptimOps = OptimOps ROCmOptim.initAdamW ROCmOptim.adamWStep

runWithOps :: Ops t -> OptimOps t opt -> Config -> IO ()
runWithOps ops optim cfg = do
  opsInit ops
  putStrLn $ "Config: " ++ show cfg
  putStrLn $ "Params: " ++ show (paramCount cfg)
  let logPath = logDir cfg </> "grokking.csv"
  createDirectoryIfMissing True (logDir cfg)
  writeFile logPath "epoch,train_loss,train_acc,test_acc,test_loss\n"
  let (trainPairs, testPairs) = makePairs cfg
  let lrEff = scaledLR cfg (length trainPairs)
  putStrLn $ "Effective LR: " ++ show lrEff
  model <- initTransformer ops cfg
  let params = transformerParams model
  opt <- optimInit optim lrEff 0.9 0.999 1.0e-8 (weightDecay cfg) params
  let testBatch = makeBatch cfg testPairs
  testRef <- newIORef (0 :: Double, 0 :: Double)
  forM_ [1 .. epochs cfg] $ \epoch -> do
    let trainBatches = trainBatchList cfg epoch trainPairs
    (trainLoss, trainAcc) <- trainEpoch ops model optim opt params trainBatches
    (testAcc, testLoss) <-
      if epoch == 1 || epoch `mod` evalEvery cfg == 0
        then do
          metrics <- evalBatch ops model testBatch
          writeIORef testRef metrics
          pure metrics
        else readIORef testRef
    putStrLn $ "epoch " ++ show epoch ++
      " trainLoss=" ++ show trainLoss ++
      " trainAcc=" ++ show trainAcc ++
      " testAcc=" ++ show testAcc ++
      " testLoss=" ++ show testLoss
    appendFile logPath $
      show epoch ++ "," ++ show trainLoss ++ "," ++ show trainAcc ++ "," ++
      show testAcc ++ "," ++ show testLoss ++ "\n"

paramCount :: Config -> Int
paramCount cfg =
  let vocabIn = modulus cfg + 1
      vocabOut = modulus cfg
      seqLen = 3
      dHead = dModel cfg `div` heads cfg
      dModel' = dModel cfg
      dMlp' = dMlp cfg
      emb = dModel' * vocabIn
      pos = dModel' * seqLen
      attn = heads cfg * (4 * dHead * dModel')
      mlp = dMlp' * dModel' + dMlp' + dModel' * dMlp' + dModel'
      unembed = vocabOut * dModel'
  in emb + pos + attn + mlp + unembed

scaledLR :: Config -> Int -> Double
scaledLR cfg trainCount =
  let denom = max 1 trainCount
      scale = lrScale cfg * fromIntegral (batchSize cfg) / fromIntegral denom
  in lr cfg * scale

data Batch = Batch
  { batchTokA :: Matrix Double
  , batchTokB :: Matrix Double
  , batchTokEq :: Matrix Double
  , batchPosA :: Matrix Double
  , batchPosB :: Matrix Double
  , batchPosE :: Matrix Double
  , batchTarget :: Matrix Double
  , batchLabels :: [Int]
  , batchCount :: Int
  }

makePairs :: Config -> ([(Int, Int)], [(Int, Int)])
makePairs cfg = (trainPairs, testPairs)
  where
    allPairs = [(a, b) | a <- [0 .. modulus cfg - 1], b <- [0 .. modulus cfg - 1]]
    total = length allPairs
    trainCount = floor (trainFrac cfg * fromIntegral total)
    shuffled = shuffle (seed cfg) allPairs
    trainPairs = take trainCount shuffled
    testPairs = drop trainCount shuffled

trainBatchList :: Config -> Int -> [(Int, Int)] -> [Batch]
trainBatchList cfg epoch pairs =
  let shuffled = shuffle (seed cfg + epoch) pairs
      groups = chunk (batchSize cfg) shuffled
  in map (makeBatch cfg) groups

makeBatch :: Config -> [(Int, Int)] -> Batch
makeBatch cfg pairs =
  let vocabIn = modulus cfg + 1
      eqTok = modulus cfg
      seqLen = 3
      inputsA = fromColumns (map (oneHot vocabIn . fst) pairs)
      inputsB = fromColumns (map (oneHot vocabIn . snd) pairs)
      inputsEq = fromColumns (replicate (length pairs) (oneHot vocabIn eqTok))
      labels = map (\(a, b) -> (a + b) `mod` modulus cfg) pairs
      targets = fromColumns (map (oneHot (modulus cfg)) labels)
      posA = positionSelector seqLen 0 (length pairs)
      posB = positionSelector seqLen 1 (length pairs)
      posE = positionSelector seqLen 2 (length pairs)
  in Batch inputsA inputsB inputsEq posA posB posE targets labels (length pairs)

positionSelector :: Int -> Int -> Int -> Matrix Double
positionSelector seqLen idx count = fromColumns (replicate count (oneHot seqLen idx))

shuffle :: Int -> [a] -> [a]
shuffle s xs = map snd $ sortOn fst $ zip (randomRs (0 :: Int, maxBound) (mkStdGen s)) xs

chunk :: Int -> [a] -> [[a]]
chunk _ [] = []
chunk n xs = take n xs : chunk n (drop n xs)

oneHot :: Int -> Int -> Vector Double
oneHot n idx = fromList [if i == idx then 1 else 0 | i <- [0 .. n - 1]]

data Head t = Head
  { headWQ :: t
  , headWK :: t
  , headWV :: t
  , headWO :: t
  }

data Transformer t = Transformer
  { tWE :: t
  , tWpos :: t
  , tHeads :: [Head t]
  , tWin :: t
  , tBin :: t
  , tWout :: t
  , tBout :: t
  , tWU :: t
  , tSeqLen :: Int
  , tDHead :: Int
  }

initTransformer :: Ops t -> Config -> IO (Transformer t)
initTransformer ops cfg = do
  let vocabIn = modulus cfg + 1
  let seqLen = 3
  let dHead = dModel cfg `div` heads cfg
  whenInvalid (dModel cfg `mod` heads cfg /= 0) "d-model must be divisible by heads"
  we <- opsRandn ops (dModel cfg) vocabIn
  wpos <- opsRandn ops (dModel cfg) seqLen
  heads' <- mapM (const (initHead ops dHead (dModel cfg))) [1 .. heads cfg]
  win <- opsRandn ops (dMlp cfg) (dModel cfg)
  bin <- opsZeros ops (dMlp cfg) 1
  wout <- opsRandn ops (dModel cfg) (dMlp cfg)
  bout <- opsZeros ops (dModel cfg) 1
  wu <- opsRandn ops (modulus cfg) (dModel cfg)
  pure $ Transformer we wpos heads' win bin wout bout wu seqLen dHead

initHead :: Ops t -> Int -> Int -> IO (Head t)
initHead ops dHead dModel' = do
  wq <- opsRandn ops dHead dModel'
  wk <- opsRandn ops dHead dModel'
  wv <- opsRandn ops dHead dModel'
  wo <- opsRandn ops dModel' dHead
  pure $ Head wq wk wv wo

whenInvalid :: Bool -> String -> IO ()
whenInvalid cond msg = if cond then error msg else pure ()

transformerParams :: Transformer t -> [t]
transformerParams t =
  [tWE t, tWpos t, tWin t, tBin t, tWout t, tBout t, tWU t] ++
  concatMap headParams (tHeads t)

headParams :: Head t -> [t]
headParams h = [headWQ h, headWK h, headWV h, headWO h]

forwardTransformer :: Ops t -> Transformer t -> Batch -> IO t
forwardTransformer ops t batch = do
  tokA <- opsFromMatrix ops (batchTokA batch)
  tokB <- opsFromMatrix ops (batchTokB batch)
  tokE <- opsFromMatrix ops (batchTokEq batch)
  posA <- opsFromMatrix ops (batchPosA batch)
  posB <- opsFromMatrix ops (batchPosB batch)
  posE <- opsFromMatrix ops (batchPosE batch)
  xAEmb <- opsMatmul ops (tWE t) tokA
  xBEmb <- opsMatmul ops (tWE t) tokB
  xEEmb <- opsMatmul ops (tWE t) tokE
  xAPos <- opsMatmul ops (tWpos t) posA
  xBPos <- opsMatmul ops (tWpos t) posB
  xEPos <- opsMatmul ops (tWpos t) posE
  xA <- opsAdd ops xAEmb xAPos
  xB <- opsAdd ops xBEmb xBPos
  xE <- opsAdd ops xEEmb xEPos
  onesRow <- opsOnes ops 1 (tDHead t)
  onesCol <- opsOnes ops (tDHead t) 1
  headOuts <- mapM (forwardHeadBatch ops t onesRow onesCol xA xB xE) (tHeads t)
  attnSum <- sumTensors ops headOuts
  x1 <- opsAdd ops xE attnSum
  mlpIn <- opsMatmul ops (tWin t) x1
  mlpInB <- opsAddBias ops mlpIn (tBin t)
  mlpAct <- opsRelu ops mlpInB
  mlpOut <- opsMatmul ops (tWout t) mlpAct
  mlpOutB <- opsAddBias ops mlpOut (tBout t)
  x2 <- opsAdd ops x1 mlpOutB
  opsMatmul ops (tWU t) x2

forwardHeadBatch :: Ops t -> Transformer t -> t -> t -> t -> t -> t -> Head t -> IO t
forwardHeadBatch ops t onesRow onesCol xA xB xE h = do
  let dHead = tDHead t
  q <- opsMatmul ops (headWQ h) xE
  kA <- opsMatmul ops (headWK h) xA
  kB <- opsMatmul ops (headWK h) xB
  kE <- opsMatmul ops (headWK h) xE
  vA <- opsMatmul ops (headWV h) xA
  vB <- opsMatmul ops (headWV h) xB
  vE <- opsMatmul ops (headWV h) xE
  sA <- columnDot ops onesRow q kA
  sB <- columnDot ops onesRow q kB
  sE <- columnDot ops onesRow q kE
  scores <- opsConcatRows ops [sA, sB, sE]
  scaled <- opsScale ops (1 / sqrt (fromIntegral dHead)) scores
  weights <- opsSoftmax ops scaled
  wA <- selectRow ops 0 weights
  wB <- selectRow ops 1 weights
  wE <- selectRow ops 2 weights
  ctxA <- applyWeights ops onesCol vA wA
  ctxB <- applyWeights ops onesCol vB wB
  ctxE <- applyWeights ops onesCol vE wE
  ctx <- sumTensors ops [ctxA, ctxB, ctxE]
  opsMatmul ops (headWO h) ctx

columnDot :: Ops t -> t -> t -> t -> IO t
columnDot ops onesRow a b = do
  prod <- opsMul ops a b
  opsMatmul ops onesRow prod

selectRow :: Ops t -> Int -> t -> IO t
selectRow ops idx mat = do
  let rowSel = (1 >< 3) [if i == idx then 1 else 0 | i <- [0 .. 2]]
  sel <- opsFromMatrix ops rowSel
  opsMatmul ops sel mat

applyWeights :: Ops t -> t -> t -> t -> IO t
applyWeights ops onesCol v wRow = do
  weightMat <- opsMatmul ops onesCol wRow
  opsMul ops v weightMat

sumTensors :: Ops t -> [t] -> IO t
sumTensors _ [] = error "sumTensors: empty list"
sumTensors ops (x:xs) = foldM (opsAdd ops) x xs

trainEpoch :: Ops t -> Transformer t -> OptimOps t opt -> opt -> [t] -> [Batch] -> IO (Double, Double)
trainEpoch ops model optim opt params batches = do
  (lossSum, correctSum, countSum) <- foldM step (0, 0, 0) batches
  let lossAvg = if countSum == 0 then 0 else lossSum / fromIntegral countSum
  let acc = if countSum == 0 then 0 else fromIntegral correctSum / fromIntegral countSum
  pure (lossAvg, acc)
  where
    step (lossAcc, correctAcc, countAcc) batch = do
      (lossSum, correct, count) <- trainBatch ops model optim opt params batch
      pure (lossAcc + lossSum, correctAcc + correct, countAcc + count)

trainBatch :: Ops t -> Transformer t -> OptimOps t opt -> opt -> [t] -> Batch -> IO (Double, Int, Int)
trainBatch ops model optim opt params batch = do
  opsZeroGrad ops params
  logits <- forwardTransformer ops model batch
  target <- opsFromMatrix ops (batchTarget batch)
  loss <- opsCrossEntropy ops logits target
  opsBackward ops loss
  optimStep optim opt
  opsZeroGrad ops params
  lossVal <- opsValue ops loss
  predVal <- opsValue ops logits
  let correct = correctCount predVal (batchLabels batch)
  let count = batchCount batch
  pure ((lossVal `atIndex` (0, 0)) * fromIntegral count, correct, count)

evalBatch :: Ops t -> Transformer t -> Batch -> IO (Double, Double)
evalBatch ops model batch = do
  logits <- forwardTransformer ops model batch
  target <- opsFromMatrix ops (batchTarget batch)
  loss <- opsCrossEntropy ops logits target
  lossVal <- opsValue ops loss
  predVal <- opsValue ops logits
  let acc = accuracy predVal (batchLabels batch)
  pure (acc, lossVal `atIndex` (0, 0))

accuracy :: Matrix Double -> [Int] -> Double
accuracy preds labels =
  let correct = correctCount preds labels
  in if null labels then 0 else fromIntegral correct / fromIntegral (length labels)

correctCount :: Matrix Double -> [Int] -> Int
correctCount preds labels =
  let guesses = map argmax (toColumns preds)
  in length (filter id (zipWith (==) guesses labels))

argmax :: Vector Double -> Int
argmax v = fst $ maximumByValue (zip [0 ..] (toList v))

maximumByValue :: Ord b => [(a, b)] -> (a, b)
maximumByValue [] = error "argmax: empty vector"
maximumByValue (x:xs) = foldl' step x xs
  where
    step best cur = if snd cur > snd best then cur else best
