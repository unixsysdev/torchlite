-- ============================================================================
-- FILE: Main.hs
-- PURPOSE: Entry point for the neural network training/testing CLI application
-- ============================================================================

-- | A "module" declaration creates a namespace - this module is called "Main"
-- The "where" keyword starts the module body
module Main where

-- | Import statements bring in code from other modules
-- "qualified" means we must prefix imports from that module with a name (like "Cpu.")
-- This prevents naming conflicts when multiple modules have functions with the same name
import qualified NNet as Cpu          -- Import CPU-based neural network backend, aliased as "Cpu"
import qualified NNetMini as Mini     -- Import mini backend, aliased as "Mini"
import qualified NNetRocm as Rocm     -- Import ROCm/GPU backend, aliased as "Rocm"

-- | Import specific functions from modules without qualifying them
import Control.Concurrent (setNumCapabilities) -- For setting number of CPU threads
import Options.Applicative   -- Library for parsing command-line arguments
import Data.Monoid ((<>))     -- Monoid operator for combining things
import System.Environment (setEnv) -- For setting environment variables
import Text.Read (readMaybe)  -- Safe way to read strings into other types

-- | Type signature: "main" is a value of type "IO ()"
-- IO () means an action that performs Input/Output and returns nothing useful (like void in C)
-- "main" is the special entry point that Haskell looks for when running a program
main :: IO ()

-- | "=" defines a value. Here, main is defined as a sequence of two actions:
-- 1. "execParser opts" - parses command line arguments into a Command value
-- 2. ">>=" - bind operator, passes the result to the next function
-- 3. "runCommand" - processes the parsed command
main = execParser opts >>= runCommand

-- | "runCommand" takes a Command and performs an IO action
-- The "::" notation is a type annotation (type signature)
-- This tells us: runCommand takes a Command and returns an IO action
runCommand :: Command -> IO ()

-- | Pattern matching with "case" expression
-- "cmd" is the input parameter
-- We match against different constructors of the Command type
runCommand cmd =
  case cmd of
    -- | Pattern match: if cmd is "TrainCmd" with fields backend, cfg, outPath
    -- Then call runTrain with those fields
    TrainCmd backend cfg outPath -> runTrain backend cfg outPath

    -- | Pattern match: if cmd is "TestCmd" with fields backend, cfg, modelPath
    -- Then call runTest with those fields
    TestCmd backend cfg modelPath -> runTest backend cfg modelPath

-- | "runTrain" function type signature:
-- Takes: Backend (which computation backend to use)
--        CliTrainConfig (training configuration)
--        Maybe FilePath (optional output path - Maybe means it could be Nothing or Just a path)
-- Returns: IO () - an IO action that returns nothing
runTrain :: Backend -> CliTrainConfig -> Maybe FilePath -> IO ()

-- | Define the runTrain function using pattern matching on the backend
runTrain backend cfg outPath =
  case backend of
    -- | If backend is BackendCpu, execute the following IO actions in sequence
    BackendCpu -> do
      -- | Apply CPU thread settings from the config
      applyCpuThreads (cpuThreadsTrain cfg)

      -- | Train the network using CPU backend
      -- "<-" binds the result of an IO action to a variable ("net")
      net <- Cpu.train (toCpuTrain cfg)

      -- | If outPath is Just path, write the network to file
      -- If outPath is Nothing, do nothing (pure () is a no-op action)
      maybe (pure ()) (`Cpu.writeNNToFile` net) outPath

    -- | If backend is BackendMini (simplified implementation)
    BackendMini -> do
      -- | Train using mini backend
      net <- Mini.train (toMiniTrain cfg)

      -- | Conditionally write the network to file
      maybe (pure ()) (`Mini.writeNNToFile` net) outPath

    -- | If backend is BackendRocm (AMD GPU implementation)
    BackendRocm -> do
      -- | Train using ROCm/GPU backend
      net <- Rocm.train (toRocmTrain cfg)

      -- | Conditionally write the network to file
      maybe (pure ()) (`Rocm.writeNNToFile` net) outPath

-- | "runTest" function type signature:
-- Takes: Backend, CliTestConfig, FilePath (modelPath is required, not Maybe)
-- Returns: IO () - prints accuracy to console
runTest :: Backend -> CliTestConfig -> FilePath -> IO ()

-- | Define runTest function with pattern matching on backend
runTest backend cfg modelPath =
  case backend of
    -- | CPU backend testing
    BackendCpu -> do
      -- | Set CPU thread count
      applyCpuThreads (cpuThreadsTest cfg)

      -- | Read the trained neural network from file
      net <- Cpu.readNNFile modelPath

      -- | Test the network and get accuracy percentage
      acc <- Cpu.testNN (toCpuTest cfg) net

      -- | Print the accuracy to console
      -- "++" concatenates strings
      -- "show" converts a value to its string representation
      putStrLn $ "Accuracy: " ++ show acc ++ "%"

    -- | Mini backend testing
    BackendMini -> do
      -- | Read the network model
      net <- Mini.readNNFile modelPath

      -- | Run the test
      acc <- Mini.testNN (toMiniTest cfg) net

      -- | Print results
      putStrLn $ "Accuracy: " ++ show acc ++ "%"

    -- | ROCm/GPU backend testing
    BackendRocm -> do
      -- | Read the network model
      net <- Rocm.readNNFile modelPath

      -- | Run the test
      acc <- Rocm.testNN (toRocmTest cfg) net

      -- | Print results
      putStrLn $ "Accuracy: " ++ show acc ++ "%"

-- ============================================================================
-- COMMAND LINE INTERFACE (CLI) TYPES AND PARSING
-- ============================================================================

-- | Define a new data type called "Backend" with three possible values (constructors)
-- This is an "algebraic data type" - a sum type (one of these three)
data Backend
  = BackendCpu    -- | Constructor for CPU backend (no associated data)
  | BackendMini   -- | Constructor for mini backend
  | BackendRocm   -- | Constructor for ROCm GPU backend

-- | "backendLabel" converts a Backend to a descriptive String
-- This is used when displaying the default backend value
backendLabel :: Backend -> String
backendLabel backend =
  case backend of
    BackendCpu  -> "cpu"    -- | Convert BackendCpu to string "cpu"
    BackendMini -> "mini"   -- | Convert BackendMini to string "mini"
    BackendRocm -> "rocm"   -- | Convert BackendRocm to string "rocm"

-- | "parseBackend" tries to convert a String into a Backend
-- Returns "Either String Backend" - either an error message (Left) or a Backend (Right)
-- This is a common pattern in Haskell for functions that can fail
parseBackend :: String -> Either String Backend
parseBackend val =
  case val of
    "cpu"   -> Right BackendCpu     -- | If string is "cpu", return Right BackendCpu (success)
    "mini"  -> Right BackendMini    -- | If string is "mini", return Right BackendMini (success)
    "rocm"  -> Right BackendRocm    -- | If string is "rocm", return Right BackendRocm (success)
    _       -> Left "backend must be one of: cpu, mini, rocm"  -- | Otherwise, return error message

-- | "Command" data type represents the top-level command structure
-- Each constructor holds different data associated with that command
data Command
  = TrainCmd Backend CliTrainConfig (Maybe FilePath)  -- | Train command: backend, config, optional output path
  | TestCmd Backend CliTestConfig FilePath            -- | Test command: backend, config, required model path

-- | "opts" defines the overall command-line parser configuration
-- ParserInfo Command wraps a parser that produces a Command value
opts :: ParserInfo Command

-- | Define opts with:
-- "info" - creates parser info from a parser and modifier options
-- "commandP <**> helper" - command parser combined with automatic help generation
-- "fullDesc" - show full help text
-- "progDesc" - program description
-- "header" - header text for help output
-- "<>" combines modifier options (Monoid pattern)
opts = info (commandP <**> helper)
  ( fullDesc
  <> progDesc "Train or test a simple neural network"
  <> header "simple-neural - configurable MNIST trainer"
  )

-- | "commandP" is a parser for the Command type
-- "Parser Command" is a type that describes how to parse command-line args into a Command
-- "hsubparser" creates subcommands (like "git commit", "git push")
commandP :: Parser Command
commandP = hsubparser
  ( command "train" (info trainP (progDesc "Train a model"))   -- | "train" subcommand
  <> command "test"  (info testP  (progDesc "Test a model"))   -- | "test" subcommand
  )

-- | "backendP" parses the --backend command-line option
-- "option" parses a command-line flag that takes a value
-- "eitherReader" converts string using our parseBackend function
backendP :: Parser Backend
backendP = option (eitherReader parseBackend)
  ( long "backend"                     -- | Long form: --backend VALUE
  <> metavar "BACKEND"                 -- | Placeholder name in help text
  <> value BackendCpu                  -- | Default value if not provided
  <> showDefaultWith backendLabel      -- | How to display the default value
  <> help "Backend: cpu, mini, rocm" ) -- | Help description

-- | "trainP" parser for the train command
-- "<$>" is "fmap" - applies a function to the result of a parser
-- "<*>" is "applicative apply" - combines multiple parsers
-- This reads: TrainCmd constructor applied to results of backendP, trainConfigP, and optional modelOutP
trainP :: Parser Command
trainP = TrainCmd <$> backendP <*> trainConfigP <*> optional modelOutP

-- | "CliTrainConfig" record type - holds all configuration for training
-- Each field has a name and a type
data CliTrainConfig = CliTrainConfig
  { trainImages       :: FilePath      -- | Path to training images file
  , trainLabels       :: FilePath      -- | Path to training labels file
  , trainLayers       :: [Int]         -- | List of layer sizes (e.g., [784,30,10])
  , trainLearningRate :: Double        -- | Learning rate (eta) for gradient descent
  , trainEpochs       :: Int           -- | Number of training epochs (full passes through data)
  , trainSamples      :: Maybe Int     -- | Optional limit on number of training samples
  , trainSeed         :: Maybe Int     -- | Optional random seed for reproducibility
  , cpuThreadsTrain   :: Maybe Int     -- | Optional CPU thread count
  }

-- | "trainConfigP" parses all the training configuration options
-- The pattern "CliTrainConfig <$> ... <*> ..." builds the record from parsers
trainConfigP :: Parser CliTrainConfig
trainConfigP =
  CliTrainConfig
    -- | Parse --train-images option (strOption expects a string)
    <$> strOption
        ( long "train-images"           -- | Long flag name
       <> metavar "PATH"                -- | Placeholder in help
       <> help "Path to training images (idx3-ubyte.gz)" )

    -- | Parse --train-labels option
    <*> strOption
        ( long "train-labels"
       <> metavar "PATH"
       <> help "Path to training labels (idx1-ubyte.gz)" )

    -- | Parse --layers option (comma-separated integers)
    <*> option (eitherReader parseLayers)    -- | Custom parser for layer sizes
        ( long "layers"
       <> metavar "SIZES"
       <> help "Comma-separated layer sizes, e.g. 784,30,10" )

    -- | Parse --learning-rate option (auto automatically detects Double type)
    <*> option auto
        ( long "learning-rate"
       <> metavar "FLOAT"
       <> help "Learning rate" )

    -- | Parse --epochs option
    <*> option auto
        ( long "epochs"
       <> metavar "INT"
       <> help "Number of epochs" )

    -- | Parse optional --train-samples option
    -- "optional" makes the parser return Nothing if flag not present
    <*> optional (option auto
        ( long "train-samples"
       <> metavar "INT"
       <> help "Limit training samples" ))

    -- | Parse optional --seed option
    <*> optional (option auto
        ( long "seed"
       <> metavar "INT"
       <> help "Random seed for initialization" ))

    -- | Parse optional --cpu-threads option
    <*> optional (option auto
        ( long "cpu-threads"
       <> metavar "INT"
       <> help "CPU thread count (OpenBLAS/OMP + RTS) for --backend cpu" ))

-- | "modelOutP" parser for the model output file path
modelOutP :: Parser FilePath
modelOutP = strOption
  ( long "model-out"                      -- | Flag: --model-out PATH
 <> metavar "PATH"                        -- | Placeholder in help
 <> help "Path to write the trained model" )

-- | "testP" parser for the test command
testP :: Parser Command
testP = TestCmd <$> backendP <*> testConfigP <*> modelInP

-- | "CliTestConfig" record type - holds all configuration for testing
data CliTestConfig = CliTestConfig
  { testImages     :: FilePath      -- | Path to test images file
  , testLabels     :: FilePath      -- | Path to test labels file
  , testSamples    :: Maybe Int     -- | Optional limit on number of test samples
  , cpuThreadsTest :: Maybe Int     -- | Optional CPU thread count
  }

-- | "testConfigP" parses all the testing configuration options
testConfigP :: Parser CliTestConfig
testConfigP =
  CliTestConfig
    -- | Parse --test-images option
    <$> strOption
        ( long "test-images"
       <> metavar "PATH"
       <> help "Path to test images (idx3-ubyte.gz)" )

    -- | Parse --test-labels option
    <*> strOption
        ( long "test-labels"
       <> metavar "PATH"
       <> help "Path to test labels (idx1-ubyte.gz)" )

    -- | Parse optional --test-samples option
    <*> optional (option auto
        ( long "test-samples"
       <> metavar "INT"
       <> help "Limit test samples" ))

    -- | Parse optional --cpu-threads option
    <*> optional (option auto
        ( long "cpu-threads"
       <> metavar "INT"
       <> help "CPU thread count (OpenBLAS/OMP + RTS) for --backend cpu" ))

-- | "modelInP" parser for the model input file path
modelInP :: Parser FilePath
modelInP = strOption
  ( long "model-in"                       -- | Flag: --model-in PATH
 <> metavar "PATH"                        -- | Placeholder
 <> help "Path to read a trained model" ) -- | Help text

-- | "parseLayers" converts a comma-separated string into a list of integers
-- Returns "Either String [Int]" - either error message or list of ints
parseLayers :: String -> Either String [Int]
parseLayers input =
  -- | "traverse" applies a function to each element of a list
  -- Here: apply "readMaybe" to each string from splitOnComma
  -- "readMaybe" safely converts String to Int (returns Maybe Int)
  case traverse readMaybe (splitOnComma input) of
    -- | If readMaybe succeeds on all elements:
    Just xs
      -- | Guard: check if at least 2 layers AND all positive
      | length xs >= 2 && all (> 0) xs -> Right xs    -- | Return the list of ints
    -- | Otherwise (either parsing failed or guards failed):
    _ -> Left "layers must be 2+ positive integers separated by commas"

-- | "splitOnComma" splits a string by commas
-- This is a recursive function that processes the string character by character
splitOnComma :: String -> [String]
splitOnComma [] = [""]                     -- | Base case: empty string returns list with one empty string
splitOnComma (',':xs) = "" : splitOnComma xs  -- | When we see comma, start new string in result
splitOnComma (x:xs) =                      -- | When we see a non-comma character:
  case splitOnComma xs of                 -- | Recursively process the rest
    [] -> [[x]]                           -- | If result is empty, return single char
    (y:ys) -> (x:y) : ys                  -- | Otherwise prepend char to first string

-- | "applyCpuThreads" configures CPU threading based on optional thread count
applyCpuThreads :: Maybe Int -> IO ()
applyCpuThreads Nothing = pure ()         -- | If Nothing, do nothing (pure () is no-op)
applyCpuThreads (Just n)                  -- | If Just n (n is the thread count):
  | n <= 0 = pure ()                      -- | If n <= 0, do nothing (invalid thread count)
  | otherwise = do                        -- | Otherwise:
      setNumCapabilities n                -- | Set Haskell runtime thread capabilities
      setEnv "OPENBLAS_NUM_THREADS" (show n)  -- | Set OpenBLAS thread count environment variable
      setEnv "OMP_NUM_THREADS" (show n)       -- | Set OpenMP thread count environment variable
      setEnv "MKL_NUM_THREADS" (show n)       -- | Set Intel MKL thread count environment variable

-- ============================================================================
-- CONFIG CONVERSION FUNCTIONS
-- Convert CLI config types to backend-specific config types
-- ============================================================================

-- | "toCpuTrain" converts CLI training config to CPU backend training config
toCpuTrain :: CliTrainConfig -> Cpu.TrainConfig
toCpuTrain cfg = Cpu.TrainConfig
  -- | Record update syntax: { Cpu.fieldName = ourFieldName, ... }
  -- This maps fields from our CliTrainConfig to the backend's TrainConfig
  { Cpu.trainImages = trainImages cfg          -- | Copy images path
  , Cpu.trainLabels = trainLabels cfg          -- | Copy labels path
  , Cpu.trainLayers = trainLayers cfg          -- | Copy layer sizes
  , Cpu.trainLearningRate = trainLearningRate cfg -- | Copy learning rate
  , Cpu.trainEpochs = trainEpochs cfg          -- | Copy epoch count
  , Cpu.trainSamples = trainSamples cfg        -- | Copy sample limit
  , Cpu.trainSeed = trainSeed cfg              -- | Copy random seed
  }

-- | "toMiniTrain" converts CLI training config to mini backend training config
toMiniTrain :: CliTrainConfig -> Mini.TrainConfig
toMiniTrain cfg = Mini.TrainConfig
  { Mini.trainImages = trainImages cfg
  , Mini.trainLabels = trainLabels cfg
  , Mini.trainLayers = trainLayers cfg
  , Mini.trainLearningRate = trainLearningRate cfg
  , Mini.trainEpochs = trainEpochs cfg
  , Mini.trainSamples = trainSamples cfg
  , Mini.trainSeed = trainSeed cfg
  }

-- | "toRocmTrain" converts CLI training config to ROCm backend training config
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

-- | "toCpuTest" converts CLI test config to CPU backend test config
toCpuTest :: CliTestConfig -> Cpu.TestConfig
toCpuTest cfg = Cpu.TestConfig
  { Cpu.testImages = testImages cfg
  , Cpu.testLabels = testLabels cfg
  , Cpu.testSamples = testSamples cfg
  }

-- | "toMiniTest" converts CLI test config to mini backend test config
toMiniTest :: CliTestConfig -> Mini.TestConfig
toMiniTest cfg = Mini.TestConfig
  { Mini.testImages = testImages cfg
  , Mini.testLabels = testLabels cfg
  , Mini.testSamples = testSamples cfg
  }

-- | "toRocmTest" converts CLI test config to ROCm backend test config
toRocmTest :: CliTestConfig -> Rocm.TestConfig
toRocmTest cfg = Rocm.TestConfig
  { Rocm.testImages = testImages cfg
  , Rocm.testLabels = testLabels cfg
  , Rocm.testSamples = testSamples cfg
  }
