module Main where

import qualified NNet as Cpu
import qualified NNetRocm as Rocm
import Control.Concurrent (setNumCapabilities)
import Options.Applicative
import Data.Monoid ((<>))
import System.Environment (setEnv)
import Text.Read (readMaybe)

main :: IO ()
main = execParser opts >>= runCommand

runCommand :: Command -> IO ()
runCommand cmd =
  case cmd of
    TrainCmd backend cfg outPath -> runTrain backend cfg outPath
    TestCmd backend cfg modelPath -> runTest backend cfg modelPath

runTrain :: Backend -> CliTrainConfig -> Maybe FilePath -> IO ()
runTrain backend cfg outPath =
  case backend of
    BackendCpu -> do
      applyCpuThreads (cpuThreadsTrain cfg)
      net <- Cpu.train (toCpuTrain cfg)
      maybe (pure ()) (`Cpu.writeNNToFile` net) outPath
    BackendRocm -> do
      net <- Rocm.train (toRocmTrain cfg)
      maybe (pure ()) (`Rocm.writeNNToFile` net) outPath

runTest :: Backend -> CliTestConfig -> FilePath -> IO ()
runTest backend cfg modelPath =
  case backend of
    BackendCpu -> do
      applyCpuThreads (cpuThreadsTest cfg)
      net <- Cpu.readNNFile modelPath
      acc <- Cpu.testNN (toCpuTest cfg) net
      putStrLn $ "Accuracy: " ++ show acc ++ "%"
    BackendRocm -> do
      net <- Rocm.readNNFile modelPath
      acc <- Rocm.testNN (toRocmTest cfg) net
      putStrLn $ "Accuracy: " ++ show acc ++ "%"

-- CLI

data Backend
  = BackendCpu
  | BackendRocm

backendLabel :: Backend -> String
backendLabel backend =
  case backend of
    BackendCpu -> "cpu"
    BackendRocm -> "rocm"

parseBackend :: String -> Either String Backend
parseBackend val =
  case val of
    "cpu" -> Right BackendCpu
    "rocm" -> Right BackendRocm
    _ -> Left "backend must be one of: cpu, rocm"

data Command
  = TrainCmd Backend CliTrainConfig (Maybe FilePath)
  | TestCmd Backend CliTestConfig FilePath

opts :: ParserInfo Command
opts = info (commandP <**> helper)
  ( fullDesc
 <> progDesc "Train or test a simple neural network"
 <> header "simple-neural - configurable MNIST trainer"
  )

commandP :: Parser Command
commandP = hsubparser
  ( command "train" (info trainP (progDesc "Train a model"))
 <> command "test" (info testP (progDesc "Test a model"))
  )

backendP :: Parser Backend
backendP = option (eitherReader parseBackend)
  ( long "backend"
 <> metavar "BACKEND"
 <> value BackendCpu
 <> showDefaultWith backendLabel
 <> help "Backend: cpu, rocm" )

trainP :: Parser Command
trainP = TrainCmd <$> backendP <*> trainConfigP <*> optional modelOutP

data CliTrainConfig = CliTrainConfig
  { trainImages :: FilePath
  , trainLabels :: FilePath
  , trainLayers :: [Int]
  , trainLearningRate :: Double
  , trainEpochs :: Int
  , trainSamples :: Maybe Int
  , trainSeed :: Maybe Int
  , cpuThreadsTrain :: Maybe Int
  }

trainConfigP :: Parser CliTrainConfig
trainConfigP =
  CliTrainConfig
    <$> strOption
        ( long "train-images"
       <> metavar "PATH"
       <> help "Path to training images (idx3-ubyte.gz)" )
    <*> strOption
        ( long "train-labels"
       <> metavar "PATH"
       <> help "Path to training labels (idx1-ubyte.gz)" )
    <*> option (eitherReader parseLayers)
        ( long "layers"
       <> metavar "SIZES"
       <> help "Comma-separated layer sizes, e.g. 784,30,10" )
    <*> option auto
        ( long "learning-rate"
       <> metavar "FLOAT"
       <> help "Learning rate" )
    <*> option auto
        ( long "epochs"
       <> metavar "INT"
       <> help "Number of epochs" )
    <*> optional (option auto
        ( long "train-samples"
       <> metavar "INT"
       <> help "Limit training samples" ))
    <*> optional (option auto
        ( long "seed"
       <> metavar "INT"
       <> help "Random seed for initialization" ))
    <*> optional (option auto
        ( long "cpu-threads"
       <> metavar "INT"
       <> help "CPU thread count (OpenBLAS/OMP + RTS) for --backend cpu" ))

modelOutP :: Parser FilePath
modelOutP = strOption
  ( long "model-out"
 <> metavar "PATH"
 <> help "Path to write the trained model" )

testP :: Parser Command
testP = TestCmd <$> backendP <*> testConfigP <*> modelInP

data CliTestConfig = CliTestConfig
  { testImages :: FilePath
  , testLabels :: FilePath
  , testSamples :: Maybe Int
  , cpuThreadsTest :: Maybe Int
  }

testConfigP :: Parser CliTestConfig
testConfigP =
  CliTestConfig
    <$> strOption
        ( long "test-images"
       <> metavar "PATH"
       <> help "Path to test images (idx3-ubyte.gz)" )
    <*> strOption
        ( long "test-labels"
       <> metavar "PATH"
       <> help "Path to test labels (idx1-ubyte.gz)" )
    <*> optional (option auto
        ( long "test-samples"
       <> metavar "INT"
       <> help "Limit test samples" ))
    <*> optional (option auto
        ( long "cpu-threads"
       <> metavar "INT"
       <> help "CPU thread count (OpenBLAS/OMP + RTS) for --backend cpu" ))

modelInP :: Parser FilePath
modelInP = strOption
  ( long "model-in"
 <> metavar "PATH"
 <> help "Path to read a trained model" )

parseLayers :: String -> Either String [Int]
parseLayers input =
  case traverse readMaybe (splitOnComma input) of
    Just xs
      | length xs >= 2 && all (> 0) xs -> Right xs
    _ -> Left "layers must be 2+ positive integers separated by commas"

splitOnComma :: String -> [String]
splitOnComma [] = [""]
splitOnComma (',':xs) = "" : splitOnComma xs
splitOnComma (x:xs) =
  case splitOnComma xs of
    [] -> [[x]]
    (y:ys) -> (x:y) : ys

applyCpuThreads :: Maybe Int -> IO ()
applyCpuThreads Nothing = pure ()
applyCpuThreads (Just n)
  | n <= 0 = pure ()
  | otherwise = do
      setNumCapabilities n
      setEnv "OPENBLAS_NUM_THREADS" (show n)
      setEnv "OMP_NUM_THREADS" (show n)
      setEnv "MKL_NUM_THREADS" (show n)

-- Config conversions for each backend.

toCpuTrain :: CliTrainConfig -> Cpu.TrainConfig
toCpuTrain cfg = Cpu.TrainConfig
  { Cpu.trainImages = trainImages cfg
  , Cpu.trainLabels = trainLabels cfg
  , Cpu.trainLayers = trainLayers cfg
  , Cpu.trainLearningRate = trainLearningRate cfg
  , Cpu.trainEpochs = trainEpochs cfg
  , Cpu.trainSamples = trainSamples cfg
  , Cpu.trainSeed = trainSeed cfg
  }

toRocmTrain :: CliTrainConfig -> Rocm.TrainConfig
toRocmTrain cfg = Rocm.TrainConfig
  { Rocm.trainImages = trainImages cfg
  , Rocm.trainLabels = trainLabels cfg
  , Rocm.trainLayers = trainLayers cfg
  , Rocm.trainLearningRate = trainLearningRate cfg
  , Rocm.trainEpochs = trainEpochs cfg
  , Rocm.trainSamples = trainSamples cfg
  , Rocm.trainSeed = trainSeed cfg
  }

toCpuTest :: CliTestConfig -> Cpu.TestConfig
toCpuTest cfg = Cpu.TestConfig
  { Cpu.testImages = testImages cfg
  , Cpu.testLabels = testLabels cfg
  , Cpu.testSamples = testSamples cfg
  }

toRocmTest :: CliTestConfig -> Rocm.TestConfig
toRocmTest cfg = Rocm.TestConfig
  { Rocm.testImages = testImages cfg
  , Rocm.testLabels = testLabels cfg
  , Rocm.testSamples = testSamples cfg
  }
