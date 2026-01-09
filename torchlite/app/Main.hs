module Main where

import Control.Monad (forM_)
import Numeric.LinearAlgebra
  ( Matrix
  , Vector
  , (><)
  , atIndex
  , cols
  , fromColumns
  , fromList
  , toColumns
  , toList
  )
import System.Environment (getArgs)
import System.Random (randomIO)
import Text.Read (readMaybe)

import TorchLite.NN (MLP, forwardMLP, mlp, mlpParams)
import TorchLite.Optim (sgd)
import TorchLite.Tensor
  ( Tensor
  , backward
  , fromMatrix
  , meanAll
  , mul
  , sub
  , value
  , zeroGrad
  )
import TorchLite.Runtime (applyThreads)

-- Simple demo: learn y = x1 + x2 on random inputs.

main :: IO ()
main = do
  cfg <- parseArgs <$> getArgs
  putStrLn $ "Config: " ++ show cfg
  applyThreads (threads cfg)
  model <- mlp [2, 64, 64, 1]
  let params = mlpParams model
  forM_ [1 .. epochs cfg] $ \epoch -> do
    lossSum <- trainEpoch cfg model params
    let avgLoss = lossSum / fromIntegral (batches cfg)
    putStrLn $ "epoch " ++ show epoch ++ " loss=" ++ show avgLoss
  testModel model

-- Training config.

data Config = Config
  { epochs :: Int
  , batches :: Int
  , batchSize :: Int
  , lr :: Double
  , threads :: Maybe Int
  } deriving (Show)

parseArgs :: [String] -> Config
parseArgs = go defaultCfg
  where
    defaultCfg = Config { epochs = 50, batches = 100, batchSize = 128, lr = 0.01, threads = Nothing }
    go cfg [] = cfg
    go cfg ("--epochs":v:rest) = go cfg { epochs = readInt v (epochs cfg) } rest
    go cfg ("--batches":v:rest) = go cfg { batches = readInt v (batches cfg) } rest
    go cfg ("--batch-size":v:rest) = go cfg { batchSize = readInt v (batchSize cfg) } rest
    go cfg ("--lr":v:rest) = go cfg { lr = readDouble v (lr cfg) } rest
    go cfg ("--threads":v:rest) = go cfg { threads = Just (readInt v 0) } rest
    go cfg (_:rest) = go cfg rest

readInt :: String -> Int -> Int
readInt s def = maybe def id (readMaybe s)

readDouble :: String -> Double -> Double
readDouble s def = maybe def id (readMaybe s)

-- Training loop.

trainEpoch :: Config -> MLP -> [Tensor] -> IO Double
trainEpoch cfg model params = go 0 0
  where
    go batchIdx acc
      | batchIdx >= batches cfg = pure acc
      | otherwise = do
          (xMat, yMat) <- makeBatch (batchSize cfg)
          x <- fromMatrix xMat
          y <- fromMatrix yMat
          pred <- forwardMLP model x
          diff <- sub pred y
          sq <- mul diff diff
          loss <- meanAll sq
          backward loss
          sgd (lr cfg) params
          zeroGrad params
          lossVal <- value loss
          let lossScalar = lossVal `atIndex` (0, 0)
          go (batchIdx + 1) (acc + lossScalar)

-- Test on a wider range to see generalization.

testModel :: MLP -> IO ()
testModel model = do
  (xMat, yMat) <- makeBatch 1000
  x <- fromMatrix xMat
  y <- fromMatrix yMat
  pred <- forwardMLP model x
  err <- meanAbsError pred y
  putStrLn $ "test mean |error|: " ++ show err

meanAbsError :: Tensor -> Tensor -> IO Double
meanAbsError a b = do
  va <- value a
  vb <- value b
  let colsA = cols va
  let absErrs =
        [ abs (vecAt colA 0 - vecAt colB 0)
        | (colA, colB) <- zip (toColumns va) (toColumns vb)
        ]
  if colsA == 0
    then pure 0
    else pure (sum absErrs / fromIntegral colsA)

-- Random batch generator.

makeBatch :: Int -> IO (Matrix Double, Matrix Double)
makeBatch n = do
  xs <- randMatrix 2 n
  let ys = addTargets xs
  pure (xs, ys)

randMatrix :: Int -> Int -> IO (Matrix Double)
randMatrix r c = do
  xs <- mapM (\_ -> randUniform (-1) 1) [1 .. r * c]
  pure $ (r >< c) xs

randUniform :: Double -> Double -> IO Double
randUniform lo hi = do
  u <- randomIO
  pure $ lo + (hi - lo) * u

addTargets :: Matrix Double -> Matrix Double
addTargets x =
  let colsX = toColumns x
      ys = map (\v -> vecAt v 0 + vecAt v 1) colsX
  in fromColumns (map (\y -> fromList [y]) ys)

vecAt :: Vector Double -> Int -> Double
vecAt v i = toList v !! i
