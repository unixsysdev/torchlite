-- ============================================================================
-- FILE: NNet.hs
-- PURPOSE: CPU-based neural network implementation for MNIST digit recognition
--          Implements a simple feedforward neural network with backpropagation
-- ============================================================================

-- | LANGUAGE pragma enables Haskell language extensions
-- BangPatterns (!) allows strict evaluation patterns for performance
{-# LANGUAGE BangPatterns #-}

-- | Module declaration - this module is named "NNet"
module NNet where

-- | Import statement: brings in the "decompress" function from GZip compression library
-- This is used to decompress .gz files containing MNIST data
import Codec.Compression.GZip (decompress)

-- | Import multiple functions from Control.Monad module:
-- "replicateM" - repeat an IO action n times and collect results
-- "when" - perform an IO action conditionally (like an if statement for IO)
-- "zipWithM" - combine two lists with a function that produces IO actions
import Control.Monad (replicateM, when, zipWithM)

-- | Import bitwise operations from Data.Bits:
-- "shiftL" - shift bits left (multiply by powers of 2)
-- ".|." - bitwise OR operation
import Data.Bits (shiftL, (.|.))

-- | Import list processing functions:
-- "foldl'" - left fold with strict evaluation (processes list from left to right)
-- "maximumBy" - find maximum element using a comparison function
import Data.List (foldl', maximumBy)

-- | Import "comparing" - creates a comparison function from a selector function
import Data.Ord (comparing)

-- | Import ByteString modules for efficient binary data processing:
-- "BS" - strict ByteString (data stored in memory all at once)
-- "BL" - lazy ByteString (data processed in chunks as needed)
-- "qualified" means we prefix with "BS." or "BL."
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL

-- | Import random number generation:
-- "mkStdGen" - create a random number generator from a seed
-- "randomIO" - generate a random value in the IO monad
-- "setStdGen" - set the global random generator
import System.Random (mkStdGen, randomIO, setStdGen)

-- | Import linear algebra functions from the hmatrix library
-- These provide matrix and vector operations for neural network computations
import Numeric.LinearAlgebra (
    Matrix,                    -- | Matrix type from hmatrix
    Vector,                    -- | Vector type from hmatrix
    (#>),                      -- | Matrix-vector multiplication operator
    (<>),                      -- | Matrix-matrix multiplication operator
    asColumn,                  -- | Convert a vector to a column matrix
    asRow,                     -- | Convert a vector to a row matrix
    cmap,                      -- | Apply a function to every element
    fromList,                  -- | Create a vector from a list
    fromLists,                 -- | Create a matrix from a list of lists
    scale,                     -- | Multiply all elements by a scalar
    toList,                    -- | Convert a vector to a list
    toLists,                   -- | Convert a matrix to a list of lists
    tr,                        -- | Matrix transpose
    (><)                       -- | Reshape/construct matrix from flat list
  )

-- ============================================================================
-- NETWORK DATA TYPES
-- Define the structure of neural networks and their layers
-- ============================================================================

-- | "Layer" represents a single layer in a neural network
-- This is a record type with named fields
-- The "!" symbols are bang patterns, indicating strict evaluation for performance
data Layer = Layer
  { layerBias    :: !(Vector Double)    -- | Bias vector - one value per neuron in next layer
  , layerWeights :: !(Matrix Double)    -- | Weight matrix - connections from previous layer
  }

-- | "Net" is a "newtype" wrapper around a list of layers
-- newtype has zero runtime overhead (optimizes away)
-- It provides type safety and documentation
newtype Net = Net
  { netLayers :: [Layer]    -- | The list of layers that make up the network
  }

-- | "Brain" is a type alias - just another name for "Net"
-- Type aliases don't create new types, just provide alternative names
-- This makes the code more readable
type Brain = Net

-- ============================================================================
-- CONFIGURATION DATA TYPES
-- Types that hold training and testing configuration parameters
-- ============================================================================

-- | "TrainConfig" holds all parameters needed for training
data TrainConfig = TrainConfig
  { trainImages       :: FilePath    -- | Path to training images file (MNIST data)
  , trainLabels       :: FilePath    -- | Path to training labels file
  , trainLayers       :: [Int]       -- | List of layer sizes (e.g., [784,30,10])
  , trainLearningRate :: Double      -- | Learning rate (eta) for gradient descent
  , trainEpochs       :: Int         -- | Number of complete passes through training data
  , trainSamples      :: Maybe Int   -- | Optional: limit number of training samples
  , trainSeed         :: Maybe Int   -- | Optional: random seed for reproducibility
  }

-- | "TestConfig" holds all parameters needed for testing
data TestConfig = TestConfig
  { testImages  :: FilePath    -- | Path to test images file
  , testLabels  :: FilePath    -- | Path to test labels file
  , testSamples :: Maybe Int   -- | Optional: limit number of test samples
  }

-- ============================================================================
-- IDX DATASET TYPES
-- Types for parsing MNIST dataset files (IDX format)
-- ============================================================================

-- | "IdxImages" represents parsed MNIST image data
data IdxImages = IdxImages
  { idxCount      :: !Int              -- | Number of images in the dataset
  , idxRows       :: !Int              -- | Height of each image in pixels
  , idxCols       :: !Int              -- | Width of each image in pixels
  , idxImageBytes :: !BS.ByteString    -- | Raw image pixel data as bytes
  }

-- | "IdxLabels" represents parsed MNIST label data
data IdxLabels = IdxLabels
  { lblCount :: !Int              -- | Number of labels in the dataset
  , lblBytes :: !BS.ByteString    -- | Raw label data as bytes
  }

-- ============================================================================
-- RANDOM NUMBER GENERATION
-- Box-Muller transform for generating normally-distributed random numbers
-- ============================================================================

-- | "bmt" implements the Box-Muller transform
-- Converts uniformly distributed random numbers to normally distributed
-- Type signature: takes a Double (scale), returns IO Double (action producing Double)
bmt :: Double -> IO Double
bmt scale = do
  -- | Generate two random numbers between 0 and 1
  x1 <- randomIO    -- | x1: first uniform random number
  x2 <- randomIO    -- | x2: second uniform random number

  -- | Avoid log(0) error by ensuring x1' is at least 1.0e-12
  let x1' = if x1 < 1.0e-12 then 1.0e-12 else x1

  -- | Apply Box-Muller transform formula
  -- This produces a normally distributed random number scaled by the scale factor
  pure $ scale * sqrt (-2 * log x1') * cos (2 * pi * x2)

-- ============================================================================
-- NETWORK INITIALIZATION
-- Create a new neural network with random weights and biases
-- ============================================================================

-- | "newBrain" creates a new neural network from layer sizes
-- Takes list of layer sizes (e.g., [784,30,10]) and returns IO Brain
newBrain :: [Int] -> IO Brain
newBrain sizes = do
  -- | Validate input: need at least input size and output size
  when (length sizes < 2) $ error "newBrain: need at least input and output sizes"

  -- | Create layers by pairing each layer size with the next layer's size
  -- zipWithM applies mkLayer to pairs of (inSize, outSize)
  layers <- zipWithM mkLayer sizes (tail sizes)

  -- | Return the network wrapped in the Net constructor
  pure $ Net layers
  where
    -- | "mkLayer" creates a single layer with random weights and biases
    -- Takes input size and output size, returns IO Layer
    mkLayer inSize outSize = do
      -- | Create random bias vector with standard deviation 0.01
      b <- randomVector 0.01 outSize

      -- | Create random weight matrix (outSize x inSize) with std dev 0.01
      w <- randomMatrix 0.01 outSize inSize

      -- | Return the Layer with bias and weights
      pure $ Layer b w

    -- | "randomVector" creates a vector of random numbers using Box-Muller transform
    -- "scale" is standard deviation, "size" is vector length
    -- replicateM repeats the bmt action 'size' times and collects results
    randomVector scale size = fromList <$> replicateM size (bmt scale)

    -- | "randomMatrix" creates a matrix of random numbers
    -- Creates rows*cols random values and reshapes into matrix
    randomMatrix scale rows cols = do
      xs <- replicateM (rows * cols) (bmt scale)
      pure $ (rows >< cols) xs

-- ============================================================================
-- ACTIVATION FUNCTIONS
-- Mathematical functions that introduce non-linearity into the network
-- ============================================================================

-- | "sigmoidSimple" is a simplified activation function
-- Actually implements ReLU (Rectified Linear Unit) despite the name
-- Type: Double -> Double (pure function, no side effects)
sigmoidSimple :: Double -> Double
sigmoidSimple x
  | x < 0       = 0     -- | If input is negative, output 0 (ReLU)
  | otherwise   = 1     -- | If input is non-negative, output 1 (step function)

-- | Note: This is actually a step/ReLU hybrid, not a true sigmoid
-- A true sigmoid would be: 1 / (1 + exp(-x))

-- ============================================================================
-- FORWARD PASS (INFERENCE)
-- Feed input through the network to get predictions
-- ============================================================================

-- | "pushThroughLayer" computes the output of a single layer
-- Takes: input activation vector, and a Layer
-- Returns: output vector after applying weights and bias
pushThroughLayer :: Vector Double -> Layer -> Vector Double
pushThroughLayer as (Layer bs wvs) = (wvs #> as) + bs
  -- | wvs #> as     : matrix-vector multiplication (weights times input)
  -- | + bs          : add bias vector to result

-- | "feedVec" feeds an input vector through the entire network
-- Takes: input vector and a Brain (network)
-- Returns: output vector from the network
feedVec :: Vector Double -> Brain -> Vector Double
feedVec input (Net layers) = foldl' step input layers
  where
    -- | "step" processes one layer
    -- "!" in !as is a bang pattern (strict evaluation for performance)
    step !as layer = cmap (max 0) (pushThroughLayer as layer)
      -- | pushThroughLayer as layer : compute layer output
      -- | cmap (max 0)              : apply ReLU activation (negative -> 0)

-- | "feed" is a convenience wrapper that works with lists instead of vectors
-- Takes: list of Doubles (input), and a Brain
-- Returns: list of Doubles (output)
feed :: [Double] -> Brain -> [Double]
feed xs net = toList $ feedVec (fromList xs) net
  -- | fromList xs  : convert input list to Vector
  -- | feedVec ...  : run the network
  -- | toList       : convert output Vector back to list

-- ============================================================================
-- DATASET LOADING
-- Read and parse MNIST dataset files from disk
-- ============================================================================

-- | "readIdxImages" reads and parses an MNIST images file
-- Returns IO IdxImages (an action that produces the parsed image data)
readIdxImages :: FilePath -> IO IdxImages
readIdxImages path = do
  -- | Read the compressed file as lazy ByteString
  raw <- BL.readFile path

  -- | Decompress the gzip data and convert to strict ByteString
  let bs = BL.toStrict (decompress raw)

  -- | Parse the binary data, handling any parse errors
  case parseIdxImages bs of
    Left err   -> error err       -- | If parse fails, crash with error message
    Right val  -> pure val        -- | If parse succeeds, return the parsed data

-- | "readIdxLabels" reads and parses an MNIST labels file
readIdxLabels :: FilePath -> IO IdxLabels
readIdxLabels path = do
  -- | Read the compressed file
  raw <- BL.readFile path

  -- | Decompress and convert to strict ByteString
  let bs = BL.toStrict (decompress raw)

  -- | Parse the binary data
  case parseIdxLabels bs of
    Left err   -> error err       -- | Handle parse errors
    Right val  -> pure val        -- | Return parsed labels

-- | "imageSize" calculates the number of pixels in each image
imageSize :: IdxImages -> Int
imageSize imgs = idxRows imgs * idxCols imgs
  -- | For MNIST: 28 * 28 = 784 pixels per image

-- | "imagePixels" extracts the pixel values for a specific image
-- Takes: IdxImages, image index n
-- Returns: list of Int pixel values (0-255)
imagePixels :: IdxImages -> Int -> [Int]
imagePixels imgs n = fromIntegral <$> BS.unpack slice
  where
    -- | Calculate size of one image in bytes
    size = imageSize imgs

    -- | Calculate starting byte position for image n
    start = n * size

    -- | Extract the bytes for this image
    -- BS.drop start   : skip to the start position
    -- BS.take size    : take 'size' bytes
    slice = BS.take size (BS.drop start (idxImageBytes imgs))
    -- | fromIntegral <$> BS.unpack : convert each Word8 to Int
    -- | <$> is infix fmap (applies function to each element)

-- | "imageVector" converts an image to a normalized Vector
-- Normalizes pixel values to range [0, 1) by dividing by 256
imageVector :: IdxImages -> Int -> Vector Double
imageVector imgs n = fromList $ (/ 256) . fromIntegral <$> imagePixels imgs n
  -- | imagePixels imgs n           : get pixel values as [Int]
  -- | fromIntegral <$>             : convert [Int] to [Double]
  -- | (/ 256) <$>                  : divide each by 256 (normalize to 0-1 range)
  -- | fromList                     : convert [Double] to Vector Double

-- | "labelValue" extracts the label (digit 0-9) for a specific sample
labelValue :: IdxLabels -> Int -> Int
labelValue lbls n = fromIntegral $ BS.index (lblBytes lbls) n
  -- | BS.index     : get the nth byte from the ByteString
  -- | fromIntegral : convert Word8 to Int

-- | "labelVector" converts a label to a one-hot encoded vector
-- One-hot encoding: [0,0,1,0,0,0,0,0,0,0] for digit 2 (10 classes)
labelVector :: Int -> IdxLabels -> Int -> Vector Double
labelVector classes lbls n =
  let target = labelValue lbls n                    -- | Get the actual label value
  in fromList $ fromIntegral . fromEnum . (target ==) <$> [0 .. classes - 1]
  -- | [0 .. classes - 1]     : generate list [0,1,2,...,classes-1]
  -- | (target ==)           : partially apply equality check
  -- |                       : for target=2: checks if each element equals 2
  -- |                       : produces [False, False, True, False, ...]
  -- | fromEnum              : convert Bool to Int (False->0, True->1)
  -- | fromIntegral          : convert Int to Double
  -- | Result for digit 2 with 10 classes: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

-- ============================================================================
-- BINARY DATA PARSING
-- Parse the IDX file format used by MNIST
-- ============================================================================

-- | "parseIdxImages" parses the IDX image file format
-- Returns Either String IdxImages (error message or parsed data)
parseIdxImages :: BS.ByteString -> Either String IdxImages
parseIdxImages bs
  -- | Check if file is too small to contain header (must be at least 16 bytes)
  | BS.length bs < 16 = Left "parseIdxImages: file too small"

  -- | Check magic number (must be 2051 for image files)
  | magic /= 2051 = Left "parseIdxImages: bad magic"

  -- | Check if file contains expected amount of data
  | BS.length bs < expected = Left "parseIdxImages: truncated data"

  -- | If all checks pass, return the parsed IdxImages structure
  | otherwise = Right $ IdxImages count rows cols (BS.drop 16 bs)
  where
    -- | Parse the header fields using getInt32BE
    magic = getInt32BE bs 0      -- | Magic number at offset 0
    count = getInt32BE bs 4      -- | Number of images at offset 4
    rows  = getInt32BE bs 8      -- | Image height at offset 8
    cols  = getInt32BE bs 12     -- | Image width at offset 12
    expected = 16 + count * rows * cols  -- | Expected total file size

-- | "parseIdxLabels" parses the IDX label file format
parseIdxLabels :: BS.ByteString -> Either String IdxLabels
parseIdxLabels bs
  -- | Check if file is too small (must be at least 8 bytes for header)
  | BS.length bs < 8 = Left "parseIdxLabels: file too small"

  -- | Check magic number (must be 2049 for label files)
  | magic /= 2049 = Left "parseIdxLabels: bad magic"

  -- | Check if file contains expected amount of data
  | BS.length bs < expected = Left "parseIdxLabels: truncated data"

  -- | If all checks pass, return the parsed IdxLabels structure
  | otherwise = Right $ IdxLabels count (BS.drop 8 bs)
  where
    magic = getInt32BE bs 0      -- | Magic number at offset 0
    count = getInt32BE bs 4      -- | Number of labels at offset 4
    expected = 8 + count         -- | Expected total file size

-- | "getInt32BE" reads a 32-bit big-endian integer from a ByteString
-- BE = Big Endian (most significant byte first)
getInt32BE :: BS.ByteString -> Int -> Int
getInt32BE bs off =
  (b0 `shiftL` 24) .|. (b1 `shiftL` 16) .|. (b2 `shiftL` 8) .|. b3
  where
    -- | Read 4 bytes starting at offset 'off'
    -- BS.index gets the byte at the specified position
    b0 = fromIntegral (BS.index bs off) :: Int       -- | First byte (most significant)
    b1 = fromIntegral (BS.index bs (off + 1)) :: Int -- | Second byte
    b2 = fromIntegral (BS.index bs (off + 2)) :: Int -- | Third byte
    b3 = fromIntegral (BS.index bs (off + 3)) :: Int -- | Fourth byte (least significant)
    -- | shiftL 24, 16, 8 shifts each byte to its correct position
    -- | .|. combines the bytes using bitwise OR

-- ============================================================================
-- TRAINING AND EVALUATION
-- Functions for training the network and testing its accuracy
-- ============================================================================

-- | "train" is the main training function
-- Takes training configuration and returns a trained Brain (network)
train :: TrainConfig -> IO Brain
train cfg = do
  -- | Load the training data from files
  imgs <- readIdxImages (trainImages cfg)    -- | Load images
  lbls <- readIdxLabels (trainLabels cfg)    -- | Load labels

  -- | Validate that layer configuration includes input and output sizes
  when (length (trainLayers cfg) < 2) $
    error "train: trainLayers must include input and output sizes"

  -- | Validate that input layer size matches image size
  let inputSize = imageSize imgs
  let expectedInput = head (trainLayers cfg)
  when (inputSize /= expectedInput) $
    error "train: input size mismatch with trainLayers"

  -- | Validate that number of images matches number of labels
  when (idxCount imgs /= lblCount lbls) $
    error "train: image/label counts do not match"

  -- | Determine how many samples to use
  let total = idxCount imgs
  let samples = min total (maybe total id (trainSamples cfg))

  -- | Validate sample count is positive
  when (samples <= 0) $
    error "train: sample count must be positive"

  -- | Extract number of output classes from last layer size
  let classes = last (trainLayers cfg)

  -- | Set random seed if provided, then initialize and train
  withSeed (trainSeed cfg) $ do
    -- | Initialize a new random network
    net0 <- newBrain (trainLayers cfg)

    -- | Extract training parameters
    let epochs = max 0 (trainEpochs cfg)      -- | Number of training epochs
    let lr = trainLearningRate cfg            -- | Learning rate

    -- | Train for the specified number of epochs
    -- foldl' applies trainEpoch repeatedly, starting with net0
    pure $ foldl' (trainEpoch imgs lbls samples classes lr) net0 [1 .. epochs]

-- | "trainEpoch" trains the network for one epoch (pass through all data)
-- Takes: images, labels, sample count, class count, learning rate, network, epoch number
-- Returns: updated network after one epoch
trainEpoch :: IdxImages -> IdxLabels -> Int -> Int -> Double -> Brain -> Int -> Brain
trainEpoch imgs lbls samples classes lr net _ = foldl' step net [0 .. samples - 1]
  where
    -- | "step" trains on a single sample
    -- Updates the network using stochastic gradient descent
    step !acc i = trainSample lr (imageVector imgs i) (labelVector classes lbls i) acc
      -- | imageVector imgs i         : get input vector for sample i
      -- | labelVector classes lbls i : get target output vector
      -- | acc                         : current network state
      -- | trainSample                 : update network with this sample

-- | "trainSample" performs one step of backpropagation on a single training sample
-- This is the core learning algorithm
trainSample :: Double -> Vector Double -> Vector Double -> Brain -> Brain
trainSample lr input target (Net layers) = Net updated
  where
    -- | Forward pass: compute activations and weighted inputs for all layers
    (activations, zs) = forwardPass input layers

    -- | Compute error at output layer
    -- cost function: difference between output and target
    -- stepVec: derivative of activation function
    deltaLast = hadamard (costVec (last activations) target) (stepVec (last zs))

    -- | Backpropagate error through the network
    deltas = backpropDeltas deltaLast (reverse layers) (reverse zs)

    -- | Update all layers using computed gradients
    updated = zipWith3 (updateLayer lr) layers deltas (init activations)

-- | "forwardPass" computes activations for all layers
-- Returns: (list of activation vectors, list of weighted input vectors)
forwardPass :: Vector Double -> [Layer] -> ([Vector Double], [Vector Double])
forwardPass input layers = (reverse actsRev, reverse zsRev)
  where
    -- | Process each layer, accumulating activations and weighted inputs
    -- Starts with ([input], []) - input is first activation, no weighted inputs yet
    (actsRev, zsRev) = foldl' step ([input], []) layers

    -- | "step" processes one layer
    step (aPrev:as, zs) layer =
      let -- | Compute weighted input: z = W*a + b
          z = pushThroughLayer aPrev layer

          -- | Compute activation: a = ReLU(z)
          a = cmap (max 0) z
      in (a:aPrev:as, z:zs)     -- | Prepend to accumulators

    -- | This case should never happen with valid input
    step _ _ = error "forwardPass: invalid activation state"

-- | "backpropDeltas" computes error terms (deltas) for all layers using backpropagation
-- Implements the backward pass of the backpropagation algorithm
backpropDeltas :: Vector Double -> [Layer] -> [Vector Double] -> [Vector Double]
backpropDeltas deltaLast revLayers revZs =
  case (revLayers, revZs) of
    -- | Normal case: at least one layer and one weighted input
    (layerLast:layersRest, _zLast:zsRest) ->
      let -- | Propagate errors backward through remaining layers
          (_, deltas) = foldl' step (layerLast, [deltaLast]) (zip layersRest zsRest)
      in deltas    -- | Return list of deltas for all layers

    -- | Error case: empty network
    _ -> error "backpropDeltas: empty network"
  where
    -- | "step" computes delta for one layer during backpropagation
    step (nextLayer, deltas@(deltaNext:_)) (layer, z) =
      let -- | Backpropagate error: delta = (W_next^T * delta_next) ⊙ stepVec(z)
          -- | tr wNext    : transpose of next layer's weight matrix
          -- | #> deltaNext: matrix-vector multiplication
          wNext = layerWeights nextLayer
          delta = hadamard (tr wNext #> deltaNext) (stepVec z)
      in (layer, delta:deltas)    -- | Prepend delta to list

    -- | This case should never happen
    step _ _ = error "backpropDeltas: invalid delta state"

-- | "updateLayer" updates weights and biases for one layer using gradient descent
-- w = w - lr * gradient
-- b = b - lr * gradient
updateLayer :: Double -> Layer -> Vector Double -> Vector Double -> Layer
updateLayer lr (Layer b w) delta aPrev = Layer b' w'
  where
    -- | "!" marks strict evaluation (compute immediately, not lazily)
    -- | Update bias: b' = b - lr * delta
    !b' = b - scale lr delta

    -- | Update weights: w' = w - lr * (delta * aPrev^T)
    -- | asColumn delta : convert delta to column matrix
    -- | asRow aPrev     : convert previous activation to row matrix
    -- | <>              : matrix multiplication (outer product)
    !w' = w - scale lr (asColumn delta <> asRow aPrev)

-- | "stepVec" computes derivative of activation function element-wise
-- Uses our simplified ReLU/step function
stepVec :: Vector Double -> Vector Double
stepVec = cmap sigmoidSimple
  -- | Apply sigmoidSimple to each element
  -- | For ReLU: derivative is 0 for x<0, 1 for x>=0
  -- | Our sigmoidSimple: 0 for x<0, 1 for x>=0

-- | "costVec" computes cost (error) for each output neuron
-- Element-wise application of cost function
costVec :: Vector Double -> Vector Double -> Vector Double
costVec a y = zipVectorWithV cost a y
  -- | Apply cost function to pairs of (output, target)

-- | "hadamard" computes element-wise (Hadamard) product of two vectors
-- Also called Schur product: c[i] = a[i] * b[i]
hadamard :: Vector Double -> Vector Double -> Vector Double
hadamard = zipVectorWithV (*)
  -- | Multiply corresponding elements

-- | "zipVectorWithV" applies a function to pairs of vector elements
-- Similar to zipWith but for Vector type
zipVectorWithV :: (Double -> Double -> Double) -> Vector Double -> Vector Double -> Vector Double
zipVectorWithV f a b = fromList $ zipWith f (toList a) (toList b)
  -- | toList a     : convert first Vector to list
  -- | toList b     : convert second Vector to list
  -- | zipWith f    : apply f to pairs of elements
  -- | fromList     : convert result back to Vector

-- | "cost" computes the error for a single output neuron
-- Using a modified hinge loss
cost :: Double -> Double -> Double
cost a y
  | y == 1 && a >= y = 0      -- | No cost if target is 1 and output >= 1
  | otherwise = a - y         -- | Otherwise, cost is the difference

-- ============================================================================
-- PREDICTION AND ACCURACY
-- Functions for making predictions and evaluating network performance
-- ============================================================================

-- | "predict" makes a prediction for a single input
-- Returns the index (class number) with the highest activation
predict :: Brain -> Vector Double -> Int
predict net v = fst $ maximumBy (comparing snd) $ zip [0 ..] (toList (feedVec v net))
  -- | feedVec v net         : run input through network
  -- | toList                : convert output Vector to list
  -- | zip [0 ..]            : pair each output with its index: [(0,0.1), (1,0.9), ...]
  -- | maximumBy (comparing snd): find pair with highest second element (output value)
  -- | fst                   : extract the index (class number) from the pair

-- | "accuracy" computes the percentage of correct predictions
-- Returns number of correct predictions per 100 samples
accuracy :: Brain -> IdxImages -> IdxLabels -> Int -> Int
accuracy net imgs lbls samples = (correct * 100) `div` samples
  where
    -- | Make predictions for all samples
    guesses = predict net . imageVector imgs <$> [0 .. samples - 1]
      -- | imageVector imgs i  : get input vector for sample i
      -- | predict net         : make prediction
      -- | <$>                 : apply to each sample index

    -- | Get actual labels for all samples
    answers = labelValue lbls <$> [0 .. samples - 1]
      -- | labelValue lbls i   : get actual label for sample i

    -- | Count how many predictions were correct
    correct = sum (fromEnum <$> zipWith (==) guesses answers)
      -- | zipWith (==)        : compare each guess with answer
      -- | fromEnum            : convert Bool to Int (False->0, True->1)
      -- | sum                 : count the True values

-- | "testNN" tests a trained network and returns accuracy percentage
testNN :: TestConfig -> Brain -> IO Int
testNN cfg net = do
  -- | Load test data
  imgs <- readIdxImages (testImages cfg)
  lbls <- readIdxLabels (testLabels cfg)

  -- | Determine how many samples to test
  let total = min (idxCount imgs) (lblCount lbls)
  let samples = min total (maybe total id (testSamples cfg))

  -- | Validate sample count
  when (samples <= 0) $
    error "testNN: sample count must be positive"

  -- | Compute and return accuracy
  pure $ accuracy net imgs lbls samples

-- ============================================================================
-- MODEL SERIALIZATION
-- Save and load trained networks to/from files
-- ============================================================================

-- | "writeNNToFile" saves a network to a text file
-- Uses Haskell's "show" to convert to string representation
writeNNToFile :: FilePath -> Brain -> IO ()
writeNNToFile fName net = writeFile fName (show (netToLists net))
  -- | netToLists net        : convert network to list-of-lists representation
  -- | show                  : convert to string
  -- | writeFile             : write string to file

-- | "readNNFile" loads a network from a text file
-- Uses Haskell's "read" to parse string representation
readNNFile :: FilePath -> IO Brain
readNNFile fName = do
  -- | Read file contents
  sNet <- readFile fName

  -- | Parse string into network structure
  let net = netFromLists (read sNet :: [([Double], [[Double]])])
    -- | read sNet                    : parse string
    -- | :: [([Double], [[Double]])]  : type annotation to tell read what to parse
    -- | netFromLists                 : convert lists to network

  -- | Return the network
  pure net

-- | "netToLists" converts a network to a list-of-lists representation
-- Each layer becomes: (bias list, weight matrix rows)
netToLists :: Brain -> [([Double], [[Double]])]
netToLists (Net layers) = [ (toList b, toLists w) | Layer b w <- layers ]
  -- | List comprehension: for each Layer in layers
  -- | toList b    : convert bias Vector to list
  -- | toLists w   : convert weight Matrix to list of lists

-- | "netFromLists" converts a list-of-lists representation back to a network
netFromLists :: [([Double], [[Double]])] -> Brain
netFromLists xs = Net [ Layer (fromList b) (fromLists w) | (b, w) <- xs ]
  -- | List comprehension: for each (bias list, weight matrix) pair
  -- | fromList b   : convert bias list to Vector
  -- | fromLists w  : convert weight lists to Matrix
  -- | Layer        : create Layer structure
  -- | Net          : wrap in Net

-- ============================================================================
-- RANDOM SEED MANAGEMENT
-- Ensure reproducibility when needed
-- ============================================================================

-- | "withSeed" sets a random seed if provided, then runs an action
-- Returns: IO a - an action producing the same type as the input action
withSeed :: Maybe Int -> IO a -> IO a
withSeed Nothing action = action                      -- | No seed: just run action
withSeed (Just seed) action = do                      -- | With seed:
  setStdGen (mkStdGen seed)                          -- | Set random generator
  action                                             -- | Run the action

-- ============================================================================
-- UTILITY FUNCTIONS
-- Miscellaneous helper functions
-- ============================================================================

-- | "num2Str" converts a number to a character
-- This function appears to be for some specific formatting purpose
-- Formula: (n * 2 / 256) then convert to string and take first character
num2Str :: Integral a => a -> Char
num2Str n = let i = fromIntegral n * 2 `div` 256 in head (show i)
  -- | fromIntegral n    : convert to a numeric type that supports division
  -- | * 2 `div` 256     : multiply by 2 and integer divide by 256
  -- | show i            : convert to string
  -- | head              : take first character
