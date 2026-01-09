-- ============================================================================
-- FILE: NNetRocm.hs
-- PURPOSE: GPU-based neural network implementation using AMD ROCm/rocBLAS
--          This is an alternative backend to NNet.hs that uses GPU acceleration
--          for matrix operations while keeping element-wise operations on CPU
-- ============================================================================

-- | LANGUAGE pragma enables Haskell language extensions
-- BangPatterns (!) allows strict evaluation patterns for performance
{-# LANGUAGE BangPatterns #-}

-- | Module declaration - this module is named "NNetRocm"
module NNetRocm where

-- | Top-level module documentation comments:
-- This is a GPU-focused backend that mirrors the NNet API.
-- This version uses ROCm/rocBLAS for matrix multiplies and keeps element-wise
-- ops on CPU for clarity and portability.

-- ============================================================================
-- IMPORT SECTION
-- Bring in all the modules and functions we need
-- ============================================================================

-- | Import GZip decompression for reading compressed MNIST files
import Codec.Compression.GZip (decompress)

-- | Import various monadic functions:
-- "foldM" - monadic fold (like foldl but with IO actions)
-- "replicateM" - repeat an IO action n times
-- "when" - perform IO action conditionally
-- "zipWithM" - combine two lists with an IO action
import Control.Monad (foldM, replicateM, when, zipWithM)

-- | Import bitwise operations for parsing binary data
import Data.Bits (shiftL, (.|.))

-- | Import list processing functions
import Data.List (foldl', maximumBy)

-- | Import comparison function creator
import Data.Ord (comparing)

-- | Import Foreign Function Interface (FFI) types and functions
-- These allow Haskell code to call C functions
import Foreign (
    ForeignPtr,          -- | Foreign pointer (pointer managed by garbage collector)
    Ptr,                 -- | Plain pointer type
    castPtr,             -- | Cast one pointer type to another
    mallocForeignPtrArray,  -- | Allocate array of foreign pointers
    withForeignPtr       -- | Execute action with raw pointer from ForeignPtr
  )

-- | Import C types for FFI
-- CDouble is C's double type, CInt is C's int type
import Foreign.C.Types (CDouble(..), CInt(..))

-- | Import function to cast between foreign pointer types
import Foreign.ForeignPtr (castForeignPtr)

-- | Import environment variable lookup
import System.Environment (lookupEnv)

-- | Import random number generation
import System.Random (mkStdGen, randomIO, setStdGen)

-- | Import safe read function (returns Maybe instead of crashing)
import Text.Read (readMaybe)

-- | Import ByteString modules for binary data
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL

-- | Import Storable Vector
-- VS.Vector is a vector type that stores data contiguously in memory
-- This is efficient for passing to C/GPU code
import qualified Data.Vector.Storable as VS

-- ============================================================================
-- FOREIGN FUNCTION INTERFACE (FFI)
-- Declare the C functions we'll call from ROCm/rocBLAS
-- ============================================================================

-- | "foreign import ccall" declares a foreign C function
-- "unsafe" means the call won't trigger Haskell garbage collection
-- "rocblas_dgemm_host" is the name of the C function
-- This is a matrix multiplication function from rocBLAS (ROCm Basic Linear Algebra Subprograms)
foreign import ccall unsafe "rocblas_dgemm_host"
  c_rocblas_dgemm_host :: CInt       -- | transA: transpose flag for matrix A (0=no, 1=yes)
                       -> CInt       -- | transB: transpose flag for matrix B (0=no, 1=yes)
                       -> CInt       -- | m: number of rows in result matrix
                       -> CInt       -- | n: number of columns in result matrix
                       -> CInt       -- | k: inner dimension (cols of A / rows of B)
                       -> Ptr CDouble -- | alpha: scalar multiplier (unused here, hardcoded in C)
                       -> CInt       -- | lda: leading dimension of matrix A
                       -> Ptr CDouble -- | Pointer to matrix A data
                       -> CInt       -- | ldb: leading dimension of matrix B
                       -> Ptr CDouble -- | Pointer to matrix B data
                       -> CInt       -- | ldc: leading dimension of result matrix
                       -> Ptr CDouble -- | Pointer to output matrix (result)
                       -> IO CInt    -- | Returns status code (0=success)

-- ============================================================================
-- NETWORK DATA TYPES
-- Similar to NNet.hs but uses GPU-friendly data structures
-- ============================================================================

-- | "Layer" represents a single layer in the neural network
-- Uses VS.Vector (storable vector) instead of hmatrix Vector
-- Uses Mat (custom matrix type) instead of hmatrix Matrix
data Layer = Layer
  { layerBias    :: !(VS.Vector Double)    -- | Bias vector as storable vector (efficient for GPU)
  , layerWeights :: !Mat                   -- | Weight matrix using custom Mat type
  }

-- | "Brain" is a newtype wrapper for the neural network
-- Contains a list of layers
newtype Brain = Brain
  { brainLayers :: [Layer]    -- | List of layers that make up the network
  }

-- ============================================================================
-- CONFIGURATION DATA TYPES
-- Same structure as NNet.hs
-- ============================================================================

-- | "TrainConfig" holds all training parameters
data TrainConfig = TrainConfig
  { trainImages       :: FilePath    -- | Path to training images file
  , trainLabels       :: FilePath    -- | Path to training labels file
  , trainLayers       :: [Int]       -- | Layer sizes (e.g., [784,30,10])
  , trainLearningRate :: Double      -- | Learning rate for gradient descent
  , trainEpochs       :: Int         -- | Number of training epochs
  , trainSamples      :: Maybe Int   -- | Optional limit on training samples
  , trainSeed         :: Maybe Int   -- | Optional random seed
  }

-- | "TestConfig" holds all testing parameters
data TestConfig = TestConfig
  { testImages  :: FilePath    -- | Path to test images file
  , testLabels  :: FilePath    -- | Path to test labels file
  , testSamples :: Maybe Int   -- | Optional limit on test samples
  }

-- ============================================================================
-- IDX DATASET TYPES
-- Same as NNet.hs - for parsing MNIST data files
-- ============================================================================

-- | "IdxImages" represents parsed MNIST image data
data IdxImages = IdxImages
  { idxCount      :: !Int              -- | Number of images in dataset
  , idxRows       :: !Int              -- | Image height in pixels
  , idxCols       :: !Int              -- | Image width in pixels
  , idxImageBytes :: !BS.ByteString    -- | Raw image pixel data
  }

-- | "IdxLabels" represents parsed MNIST label data
data IdxLabels = IdxLabels
  { lblCount :: !Int              -- | Number of labels in dataset
  , lblBytes :: !BS.ByteString    -- | Raw label data
  }

-- ============================================================================
-- MATRIX DATA TYPE
-- Custom matrix type optimized for GPU operations
-- ============================================================================

-- | "Mat" represents a matrix in column-major format
-- Column-major means data is stored column by column (like Fortran, BLAS, LAPACK)
-- This is the format expected by rocBLAS
data Mat = Mat
  { matRows :: !Int                  -- | Number of rows in the matrix
  , matCols :: !Int                  -- | Number of columns in the matrix
  , matData :: !(VS.Vector Double)   -- | Matrix data in column-major order
  }

-- ============================================================================
-- RANDOM NUMBER GENERATION
-- Box-Muller transform for normally-distributed random initialization
-- ============================================================================

-- | "bmt" implements the Box-Muller transform
-- Converts uniform random numbers to normal distribution
bmt :: Double -> IO Double
bmt scale = do
  -- | Generate two uniform random numbers between 0 and 1
  x1 <- randomIO    -- | First uniform random number
  x2 <- randomIO    -- | Second uniform random number

  -- | Prevent log(0) error by ensuring x1' is at least 1.0e-12
  let x1' = if x1 < 1.0e-12 then 1.0e-12 else x1

  -- | Apply Box-Muller transform formula
  -- Produces normally distributed random number scaled by 'scale'
  pure $ scale * sqrt (-2 * log x1') * cos (2 * pi * x2)

-- | "newBrain" creates a new neural network with random weights and biases
-- Takes list of layer sizes and returns IO Brain
newBrain :: [Int] -> IO Brain
newBrain sizes = do
  -- | Validate input: need at least input and output sizes
  when (length sizes < 2) $ error "newBrain: need at least input and output sizes"

  -- | Create layers by pairing consecutive layer sizes
  -- zipWithM applies mkLayer to pairs (inSize, outSize)
  layers <- zipWithM mkLayer sizes (tail sizes)

  -- | Return the Brain with the created layers
  pure $ Brain layers
  where
    -- | "mkLayer" creates one layer with random weights and biases
    mkLayer inSize outSize = do
      -- | Create random bias vector
      b <- randomVector 0.01 outSize

      -- | Create random weight matrix
      w <- randomMatrix 0.01 outSize inSize

      -- | Return the Layer
      pure $ Layer b w

    -- | "randomVector" creates a vector of normally-distributed random numbers
    -- Uses Box-Muller transform for each element
    randomVector scale size = VS.fromList <$> replicateM size (bmt scale)
      -- | replicateM size (bmt scale) : generate 'size' random numbers
      -- | VS.fromList                 : convert list to storable vector

    -- | "randomMatrix" creates a matrix of normally-distributed random numbers
    randomMatrix scale rows cols = do
      -- | Generate rows*cols random values
      xs <- replicateM (rows * cols) (bmt scale)

      -- | Create Mat structure with the values
      pure $ Mat rows cols (VS.fromList xs)

-- ============================================================================
-- ACTIVATION FUNCTIONS
-- Define ReLU (Rectified Linear Unit) and its derivative
-- ============================================================================

-- | "relu" implements the ReLU activation function
-- Actually a step function: 0 if x < 0, 1 otherwise
relu :: Double -> Double
relu x
  | x < 0       = 0     -- | Negative values map to 0
  | otherwise   = 1     -- | Non-negative values map to 1

-- | "reluMat" applies ReLU activation to all matrix elements
-- Uses max 0 which is the actual ReLU (returns x if x > 0, else 0)
reluMat :: Mat -> Mat
reluMat = matMap (max 0)

-- | "reluStepMat" applies the relu step function (derivative) to all elements
-- Used in backpropagation for computing deltas
reluStepMat :: Mat -> Mat
reluStepMat = matMap relu

-- ============================================================================
-- MATRIX OPERATIONS (CPU SIDE)
-- Implementations that work with our custom Mat type
-- ============================================================================

-- | "matMap" applies a function to every element of a matrix
matMap :: (Double -> Double) -> Mat -> Mat
matMap f (Mat r c v) = Mat r c (VS.map f v)
  -- | VS.map f v : apply function f to each element of vector v

-- | "matZipWith" combines two matrices element-wise with a function
matZipWith :: (Double -> Double -> Double) -> Mat -> Mat -> Mat
matZipWith f (Mat r c a) (Mat r' c' b)
  -- | Check that matrices have the same dimensions
  | r /= r' || c /= c' =
      error ("matZipWith: shape mismatch (" ++ show r ++ "x" ++ show c ++
             " vs " ++ show r' ++ "x" ++ show c' ++ ")")
  -- | Apply function element-wise
  | otherwise = Mat r c (VS.zipWith f a b)

-- | "matScale" multiplies all matrix elements by a scalar
matScale :: Double -> Mat -> Mat
matScale s (Mat r c v) = Mat r c (VS.map (s *) v)
  -- | s * : multiply each element by scalar s

-- | "matAdd" adds two matrices element-wise
matAdd :: Mat -> Mat -> Mat
matAdd = matZipWith (+)

-- | "matSub" subtracts two matrices element-wise
matSub :: Mat -> Mat -> Mat
matSub = matZipWith (-)

-- | "matAddBias" adds a bias vector to each column of a matrix
-- Important for neural network layer computation: z = W*a + b
matAddBias :: Mat -> VS.Vector Double -> Mat
matAddBias (Mat r c v) bias
  -- | Validate bias vector length matches number of rows
  | VS.length bias /= r = error "matAddBias: bias length mismatch"
  -- | Add bias element j to each element in row j
  | otherwise = Mat r c $ VS.generate (r * c) $ \idx ->
      let i = idx `mod` r             -- | Row index
      in v VS.! idx + bias VS.! i     -- | Add bias to element
  -- | VS.generate creates a vector by applying a function to each index
  -- | idx is the flat index into the column-major data

-- | "matSumColumns" sums each row of a matrix, producing a vector
-- Used for averaging gradients across a batch
matSumColumns :: Mat -> VS.Vector Double
matSumColumns (Mat r c v) = VS.generate r $ \i ->
  let -- | Inner recursive function to sum across columns
      go j acc
        | j == c    = acc              -- | Base case: processed all columns
        | otherwise = go (j + 1) (acc + v VS.! (i + j * r))  -- | Add element and continue
  in go 0 0
  -- | In column-major: element at row i, column j is at index i + j*r
  -- | i + j * r : compute flat index for row i, column j

-- | "matTranspose" transposes a matrix (rows become columns, columns become rows)
matTranspose :: Mat -> Mat
matTranspose (Mat r c v) = Mat c r $ VS.generate (r * c) $ \idx ->
  let i = idx `mod` c    -- | New row index (was column)
      j = idx `div` c    -- | New column index (was row)
  in v VS.! (j + i * r) -- | Get element from original matrix
  -- | Original at row j, column i: index j + i*r

-- | "matFromLists" creates a Mat from a list of lists (row-major format)
-- Converts from row-major to column-major storage
matFromLists :: [[Double]] -> Mat
matFromLists rows
  | null rows = Mat 0 0 VS.empty                    -- | Handle empty matrix
  -- | Check that all rows have the same length
  | any ((/= c) . length) rows = error "matFromLists: ragged rows"
  -- | Convert to column-major format
  | otherwise = Mat r c (VS.fromList colMajor)
  where
    r = length rows           -- | Number of rows
    c = length (head rows)    -- | Number of columns
    -- | Build column-major list by iterating columns first, then rows
    colMajor = [ rows !! i !! j | j <- [0 .. c - 1], i <- [0 .. r - 1] ]
    -- | For column j, row i: access rows !! i !! j

-- | "matToLists" converts a Mat to a list of lists (row-major format)
-- Converts from column-major to row-major for display/storage
matToLists :: Mat -> [[Double]]
matToLists (Mat r c v) =
  [ [ v VS.! (i + j * r) | j <- [0 .. c - 1] ] | i <- [0 .. r - 1] ]
  -- | For each row i: for each column j: get element at flat index i + j*r

-- | "matFromColumns" creates a Mat from a list of column vectors
-- Useful for creating batch matrices from multiple samples
matFromColumns :: Int -> [VS.Vector Double] -> Mat
matFromColumns r cols
  | null cols = Mat r 0 VS.empty                     -- | Handle empty list
  -- | Check all columns have the same length (number of rows)
  | any ((/= r) . VS.length) cols = error "matFromColumns: column size mismatch"
  -- | Concatenate all columns
  | otherwise = Mat r c (VS.concat cols)
  where
    c = length cols    -- | Number of columns

-- | "matToVector" extracts the first column of a matrix as a vector
-- Used for getting output from single-input feedforward
matToVector :: Mat -> VS.Vector Double
matToVector (Mat r c v)
  | c < 1     = VS.empty                      -- | Handle empty matrix
  | otherwise = VS.slice 0 r v               -- | Take first r elements (first column)
  -- | VS.slice start length vec : extract 'length' elements starting at 'start'

-- | "vecScale" multiplies all vector elements by a scalar
vecScale :: Double -> VS.Vector Double -> VS.Vector Double
vecScale s v = VS.map (s *) v
  -- | s * : multiply each element by scalar s

-- | "vecSub" subtracts two vectors element-wise
vecSub :: VS.Vector Double -> VS.Vector Double -> VS.Vector Double
vecSub = VS.zipWith (-)

-- ============================================================================
-- CPU MATRIX MULTIPLICATION
-- Naive implementation for forward pass/inference
-- ============================================================================

-- | "matMulCPU" performs matrix multiplication on CPU
-- Computes C = A * B using naive O(n^3) algorithm
matMulCPU :: Mat -> Mat -> Mat
matMulCPU (Mat ar ac av) (Mat br bc bv)
  -- | Validate dimensions: columns of A must equal rows of B
  | ac /= br = error "matMulCPU: shape mismatch"
  -- | Compute product
  | otherwise = Mat ar bc $ VS.generate (ar * bc) $ \idx ->
      let i = idx `mod` ar        -- | Row index in result
          j = idx `div` ar        -- | Column index in result
          -- | Inner loop: compute dot product of row i of A and column j of B
          go k acc
            | k == ac    = acc    -- | Base case: processed all elements
            | otherwise =
                let -- | Get A[i,k]
                    aVal = av VS.! (i + k * ar)
                    -- | Get B[k,j] - remember B is column-major
                    bVal = bv VS.! (k + j * br)
                in go (k + 1) (acc + aVal * bVal)    -- | Add product and continue
      in go 0 0

-- ============================================================================
-- GPU MATRIX MULTIPLICATION
-- Uses ROCm/rocBLAS for fast GPU computation
-- ============================================================================

-- | "Transpose" data type for specifying matrix transpose operations
data Transpose = NoTrans | Trans
  deriving (Eq)    -- | Enable equality comparison

-- | "matMulGpu" performs matrix multiplication on GPU using rocBLAS
-- Returns IO Mat because GPU operations are in the IO monad
matMulGpu :: Transpose -> Transpose -> Mat -> Mat -> IO Mat
matMulGpu transA transB a@(Mat ar ac _) b@(Mat br bc _) = do
  -- | Compute effective dimensions after considering transposes
  let (aRowsEff, aColsEff) = if transA == NoTrans then (ar, ac) else (ac, ar)
  let (bRowsEff, bColsEff) = if transB == NoTrans then (br, bc) else (bc, br)

  -- | Validate dimensions for multiplication
  when (aColsEff /= bRowsEff) $ error "matMulGpu: shape mismatch"

  -- | Extract dimensions for BLAS call
  let m = aRowsEff    -- | Rows of result
  let n = bColsEff    -- | Columns of result
  let k = aColsEff    -- | Inner dimension

  -- | Allocate memory for output matrix
  -- mallocForeignPtrArray allocates memory managed by Haskell GC
  outFp <- (mallocForeignPtrArray (m * n) :: IO (ForeignPtr CDouble))

  -- | Compute leading dimensions (stride between columns in column-major format)
  let lda = ar    -- | Leading dimension of A
  let ldb = br    -- | Leading dimension of B
  let ldc = m     -- | Leading dimension of result

  -- | Perform GPU matrix multiplication using FFI
  withForeignPtr outFp $ \outPtr ->
    VS.unsafeWith (matData a) $ \aPtr ->
      VS.unsafeWith (matData b) $ \bPtr -> do
        -- | Convert Transpose to C int (0 for NoTrans, 1 for Trans)
        let ta = if transA == NoTrans then 0 else 1
        let tb = if transB == NoTrans then 0 else 1

        -- | Call the C function via FFI
        rc <- c_rocblas_dgemm_host
                (fromIntegral ta)      -- | transA as C int
                (fromIntegral tb)      -- | transB as C int
                (fromIntegral m)       -- | m as C int
                (fromIntegral n)       -- | n as C int
                (fromIntegral k)       -- | k as C int
                (castPtr aPtr)         -- | Pointer to A data, cast to CDouble*
                (fromIntegral lda)     -- | lda as C int
                (castPtr bPtr)         -- | Pointer to B data
                (fromIntegral ldb)     -- | ldb as C int
                (castPtr outPtr)       -- | Pointer to output
                (fromIntegral ldc)     -- | ldc as C int

        -- | Check for errors
        when (rc /= 0) $ error "matMulGpu: rocBLAS call failed"

  -- | Convert result back to Haskell Mat
  let outVec = VS.unsafeFromForeignPtr0 (castForeignPtr outFp) (m * n)
    -- | castForeignPtr : cast from CDouble to Double
    -- | unsafeFromForeignPtr0 : create vector from foreign pointer

  -- | Return the result matrix
  pure $ Mat m n outVec

-- ============================================================================
-- FORWARD PASS (INFERENCE)
-- CPU-based forward pass using matMulCPU
-- ============================================================================

-- | "pushThroughLayer" computes output of one layer
-- z = W*a + b where W is weights, a is input, b is bias
pushThroughLayer :: Mat -> Layer -> Mat
pushThroughLayer as (Layer bs wvs) =
  let -- | Compute weighted input: z = W * a
      z = matMulCPU wvs as
  in matAddBias z bs    -- | Add bias: z + b

-- | "feedMat" feeds an input matrix through the entire network
-- Works with matrices (useful for batching)
feedMat :: Mat -> Brain -> Mat
feedMat input (Brain layers) = foldl' step input layers
  where
    -- | Process each layer
    step !as layer = reluMat (pushThroughLayer as layer)
      -- | pushThroughLayer : compute layer output
      -- | reluMat          : apply ReLU activation

-- | "feed" is a convenience wrapper for lists (single input)
feed :: [Double] -> Brain -> [Double]
feed xs net = VS.toList $ matToVector $ feedMat input net
  where
    -- | Convert list to column matrix
    input = Mat (length xs) 1 (VS.fromList xs)

-- ============================================================================
-- DATASET LOADING
-- Same as NNet.hs - read and parse MNIST files
-- ============================================================================

-- | "readIdxImages" reads and decompresses an MNIST image file
readIdxImages :: FilePath -> IO IdxImages
readIdxImages path = do
  -- | Read compressed file
  raw <- BL.readFile path

  -- | Decompress and convert to strict ByteString
  let bs = BL.toStrict (decompress raw)

  -- | Parse the binary data
  case parseIdxImages bs of
    Left err   -> error err    -- | Handle parse errors
    Right val  -> pure val     -- | Return parsed data

-- | "readIdxLabels" reads and decompresses an MNIST label file
readIdxLabels :: FilePath -> IO IdxLabels
readIdxLabels path = do
  -- | Read compressed file
  raw <- BL.readFile path

  -- | Decompress and convert to strict ByteString
  let bs = BL.toStrict (decompress raw)

  -- | Parse the binary data
  case parseIdxLabels bs of
    Left err   -> error err    -- | Handle parse errors
    Right val  -> pure val     -- | Return parsed data

-- | "parseIdxImages" parses IDX image file format
parseIdxImages :: BS.ByteString -> Either String IdxImages
parseIdxImages bs
  -- | Validate file size
  | BS.length bs < 16 = Left "parseIdxImages: file too small"

  -- | Check magic number (must be 2051)
  | magic /= 2051 = Left "parseIdxImages: bad magic"

  -- | Validate data size
  | BS.length bs < expected = Left "parseIdxImages: truncated data"

  -- | Return parsed structure
  | otherwise = Right $ IdxImages count rows cols (BS.drop 16 bs)
  where
    magic = getInt32BE bs 0      -- | Magic number at offset 0
    count = getInt32BE bs 4      -- | Image count at offset 4
    rows  = getInt32BE bs 8      -- | Row count at offset 8
    cols  = getInt32BE bs 12     -- | Column count at offset 12
    expected = 16 + count * rows * cols    -- | Expected file size

-- | "parseIdxLabels" parses IDX label file format
parseIdxLabels :: BS.ByteString -> Either String IdxLabels
parseIdxLabels bs
  -- | Validate file size
  | BS.length bs < 8 = Left "parseIdxLabels: file too small"

  -- | Check magic number (must be 2049)
  | magic /= 2049 = Left "parseIdxLabels: bad magic"

  -- | Validate data size
  | BS.length bs < expected = Left "parseIdxLabels: truncated data"

  -- | Return parsed structure
  | otherwise = Right $ IdxLabels count (BS.drop 8 bs)
  where
    magic = getInt32BE bs 0      -- | Magic number at offset 0
    count = getInt32BE bs 4      -- | Label count at offset 4
    expected = 8 + count         -- | Expected file size

-- | "getInt32BE" reads a 32-bit big-endian integer from ByteString
getInt32BE :: BS.ByteString -> Int -> Int
getInt32BE bs off =
  (b0 `shiftL` 24) .|. (b1 `shiftL` 16) .|. (b2 `shiftL` 8) .|. b3
  where
    -- | Read 4 bytes and convert to Int
    b0 = fromIntegral (BS.index bs off) :: Int       -- | Most significant byte
    b1 = fromIntegral (BS.index bs (off + 1)) :: Int -- | Second byte
    b2 = fromIntegral (BS.index bs (off + 2)) :: Int -- | Third byte
    b3 = fromIntegral (BS.index bs (off + 3)) :: Int -- | Least significant byte

-- ============================================================================
-- IMAGE AND LABEL PROCESSING
-- Convert MNIST data to neural network format
-- ============================================================================

-- | "imageSize" calculates pixels per image
imageSize :: IdxImages -> Int
imageSize imgs = idxRows imgs * idxCols imgs

-- | "imageVector" extracts and normalizes one image as a vector
imageVector :: IdxImages -> Int -> VS.Vector Double
imageVector imgs n = VS.fromList $ (/ 256) . fromIntegral <$> pixels
  where
    -- | Calculate image size and starting position
    size = imageSize imgs
    start = n * size

    -- | Extract pixel bytes
    pixels = BS.unpack $ BS.take size (BS.drop start (idxImageBytes imgs))
    -- | fromIntegral <$> : convert Word8 to Double
    -- | (/ 256)           : normalize to [0, 1)

-- | "labelValue" extracts the label (0-9) for one sample
labelValue :: IdxLabels -> Int -> Int
labelValue lbls n = fromIntegral $ BS.index (lblBytes lbls) n
  -- | BS.index : get byte at position n

-- | "labelVector" converts label to one-hot encoded vector
labelVector :: Int -> IdxLabels -> Int -> VS.Vector Double
labelVector classes lbls n =
  let target = labelValue lbls n                    -- | Get the label
  in VS.fromList $ fromIntegral . fromEnum . (target ==) <$> [0 .. classes - 1]
  -- | [0 .. classes - 1] : generate class indices
  -- | (target ==)       : partial application of equality
  -- | fromEnum          : Bool to Int (False->0, True->1)
  -- | fromIntegral      : Int to Double

-- ============================================================================
-- COST FUNCTIONS AND MATH HELPERS
-- ============================================================================

-- | "cost" computes error for a single output neuron
cost :: Double -> Double -> Double
cost a y
  | y == 1 && a >= y = 0      -- | No cost if correctly confident
  | otherwise = a - y         -- | Otherwise, difference is error

-- | "costMat" applies cost function element-wise to matrices
costMat :: Mat -> Mat -> Mat
costMat = matZipWith cost

-- | "hadamard" computes element-wise product of two matrices
hadamard :: Mat -> Mat -> Mat
hadamard = matZipWith (*)

-- ============================================================================
-- GPU-ACCELERATED TRAINING
-- Uses GPU for forward pass and backpropagation
-- ============================================================================

-- | "forwardPassBatch" performs forward pass for a batch using GPU
-- Returns IO because GPU operations are in IO monad
forwardPassBatch :: Mat -> [Layer] -> IO ([Mat], [Mat])
forwardPassBatch input layers = do
  -- | foldM is monadic fold (like foldl but with IO)
  (actsRev, zsRev) <- foldM step ([input], []) layers

  -- | Reverse to get correct order (input first, output last)
  pure (reverse actsRev, reverse zsRev)
  where
    -- | "step" processes one layer monadically
    step (aPrev:as, zs) layer = do
      -- | GPU matrix multiplication: z = W * a
      z <- matMulGpu NoTrans NoTrans (layerWeights layer) aPrev

      -- | Add bias: z + b
      let zBias = matAddBias z (layerBias layer)

      -- | Apply ReLU activation
      let a = reluMat zBias

      -- | Accumulate results
      pure (a:aPrev:as, zBias:zs)

    -- | Error case (shouldn't happen)
    step _ _ = error "forwardPassBatch: invalid activation state"

-- | "backpropDeltas" computes error terms using backpropagation with GPU
backpropDeltas :: Mat -> [Layer] -> [Mat] -> IO [Mat]
backpropDeltas deltaLast revLayers revZs =
  case (revLayers, revZs) of
    -- | Normal case: have layers to process
    (layerLast:layersRest, _zLast:zsRest) -> do
      -- | Backpropagate through remaining layers
      (_, deltasRev) <- foldM step (layerLast, [deltaLast]) (zip layersRest zsRest)

      -- | Return deltas
      pure deltasRev

    -- | Error case: empty network
    _ -> error "backpropDeltas: empty network"
  where
    -- | "step" computes delta for one layer
    step (nextLayer, deltas@(deltaNext:_)) (layer, z) = do
      -- | GPU matrix multiply: delta_base = W_next^T * delta_next
      deltaBase <- matMulGpu Trans NoTrans (layerWeights nextLayer) deltaNext

      -- | Apply activation derivative: delta = delta_base ⊙ stepVec(z)
      let delta = hadamard deltaBase (reluStepMat z)

      -- | Accumulate
      pure (layer, delta:deltas)

    -- | Error case
    step _ _ = error "backpropDeltas: invalid delta state"

-- | "updateLayer" updates weights and biases for one layer
-- Returns IO Layer because GPU operations are used
updateLayer :: Double -> Double -> Layer -> Mat -> Mat -> IO Layer
updateLayer lr invBatch (Layer b w) delta aPrev = do
  -- | GPU matrix multiply: dW = delta * aPrev^T
  dW <- matMulGpu NoTrans Trans delta aPrev

  -- | Scale by 1/batch_size
  let dW' = matScale invBatch dW

  -- | Sum columns for bias gradient
  let db = vecScale invBatch (matSumColumns delta)

  -- | Update weights: w = w - lr * dW
  let w' = matSub w (matScale lr dW')

  -- | Update bias: b = b - lr * db
  let b' = vecSub b (vecScale lr db)

  -- | Return updated layer
  pure $ Layer b' w'

-- | "trainBatch" trains on one batch of samples
trainBatch :: Double -> Mat -> Mat -> Brain -> IO Brain
trainBatch lr input target (Brain layers) = do
  -- | Forward pass through all layers
  (activations, zs) <- forwardPassBatch input layers

  -- | Compute output layer error
  let deltaLast = hadamard (costMat (last activations) target) (reluStepMat (last zs))

  -- | Backpropagate errors
  deltas <- backpropDeltas deltaLast (reverse layers) (reverse zs)

  -- | Compute inverse batch size for averaging
  let invBatch = 1 / fromIntegral (matCols input)

  -- | Update all layers
  let updates = zipWith3 (updateLayer lr invBatch) layers deltas (init activations)

  -- | Execute all updates (sequence turns [IO Layer] into IO [Layer])
  updated <- sequence updates

  -- | Return updated network
  pure $ Brain updated

-- ============================================================================
-- TRAINING ENTRY POINT
-- Orchestrates the entire training process
-- ============================================================================

-- | "train" is the main training function
train :: TrainConfig -> IO Brain
train cfg = do
  -- | Load training data
  imgs <- readIdxImages (trainImages cfg)
  lbls <- readIdxLabels (trainLabels cfg)

  -- | Validate layer configuration
  when (length (trainLayers cfg) < 2) $
    error "train: trainLayers must include input and output sizes"

  -- | Validate input size matches image size
  let inputSize = imageSize imgs
  let expectedInput = head (trainLayers cfg)
  when (inputSize /= expectedInput) $
    error "train: input size mismatch with trainLayers"

  -- | Validate data consistency
  when (idxCount imgs /= lblCount lbls) $
    error "train: image/label counts do not match"

  -- | Determine number of samples to use
  let total = idxCount imgs
  let samples = min total (maybe total id (trainSamples cfg))

  -- | Validate sample count
  when (samples <= 0) $ error "train: sample count must be positive"

  -- | Extract number of output classes
  let classes = last (trainLayers cfg)

  -- | Read batch size from environment or use default
  batchSize <- readBatchSize samples

  -- | Set random seed if provided, then train
  withSeed (trainSeed cfg) $ do
    -- | Initialize network
    net0 <- newBrain (trainLayers cfg)

    -- | Extract training parameters
    let epochs = max 0 (trainEpochs cfg)
    let lr = trainLearningRate cfg

    -- | Train for all epochs (foldM for monadic fold)
    foldM (trainEpoch imgs lbls samples classes lr batchSize) net0 [1 .. epochs]

-- | "trainEpoch" trains for one epoch using mini-batches
trainEpoch :: IdxImages -> IdxLabels -> Int -> Int -> Double -> Int -> Brain -> Int -> IO Brain
trainEpoch imgs lbls samples classes lr batchSize net _ =
  -- | Process batches: [0, batchSize, 2*batchSize, ...]
  foldM step net [0, batchSize .. samples - 1]
  where
    -- | "step" processes one batch
    step !acc start = do
      -- | Compute batch end (don't go past samples)
      let end = min samples (start + batchSize)

      -- | Compute actual batch size
      let count = end - start

      -- | Collect input vectors for this batch
      let xs = [ imageVector imgs i | i <- [start .. end - 1] ]

      -- | Collect target vectors for this batch
      let ys = [ labelVector classes lbls i | i <- [start .. end - 1] ]

      -- | Convert lists of vectors to batch matrices
      let xMat = matFromColumns (imageSize imgs) xs
      let yMat = matFromColumns classes ys

      -- | Validate batch is not empty
      when (count <= 0) $ error "trainEpoch: empty batch"

      -- | Train on this batch
      trainBatch lr xMat yMat acc

-- | "readBatchSize" reads batch size from environment variable
readBatchSize :: Int -> IO Int
readBatchSize samples = do
  -- | Try to read NNET_BATCH_SIZE environment variable
  env <- lookupEnv "NNET_BATCH_SIZE"

  -- | Parse the value or use default
  case env >>= readMaybe of
    Just n | n > 0 -> pure (min samples n)    -- | Use env value if valid
    _         -> pure (min samples 128)        -- | Otherwise use 128

-- ============================================================================
-- PREDICTION AND ACCURACY
-- Functions for testing the trained network
-- ============================================================================

-- | "predict" makes a prediction for a single input
predict :: Brain -> VS.Vector Double -> Int
predict net v = fst $ maximumBy (comparing snd) $ zip [0 ..] scores
  where
    -- | Feed input through network and get output
    scores = VS.toList $ matToVector $ feedMat (Mat (VS.length v) 1 v) net
    -- | zip [0 ..] : pair outputs with indices
    -- | maximumBy (comparing snd) : find index with highest output
    -- | fst : extract the index (class number)

-- | "accuracy" computes percentage of correct predictions
accuracy :: Brain -> IdxImages -> IdxLabels -> Int -> Int
accuracy net imgs lbls samples = (correct * 100) `div` samples
  where
    -- | Make predictions for all samples
    guesses = predict net . imageVector imgs <$> [0 .. samples - 1]

    -- | Get actual labels
    answers = labelValue lbls <$> [0 .. samples - 1]

    -- | Count correct predictions
    correct = sum (fromEnum <$> zipWith (==) guesses answers)

-- | "testNN" tests a trained network and returns accuracy
testNN :: TestConfig -> Brain -> IO Int
testNN cfg net = do
  -- | Load test data
  imgs <- readIdxImages (testImages cfg)
  lbls <- readIdxLabels (testLabels cfg)

  -- | Determine number of samples to test
  let total = min (idxCount imgs) (lblCount lbls)
  let samples = min total (maybe total id (testSamples cfg))

  -- | Validate sample count
  when (samples <= 0) $ error "testNN: sample count must be positive"

  -- | Compute and return accuracy
  pure $ accuracy net imgs lbls samples

-- ============================================================================
-- MODEL SERIALIZATION
-- Save and load trained networks
-- ============================================================================

-- | "writeNNToFile" saves a network to a text file
writeNNToFile :: FilePath -> Brain -> IO ()
writeNNToFile fName net = writeFile fName (show (netToLists net))

-- | "readNNFile" loads a network from a text file
readNNFile :: FilePath -> IO Brain
readNNFile fName = do
  -- | Read file content
  sNet <- readFile fName

  -- | Parse and construct network
  let net = netFromLists (read sNet :: [([Double], [[Double]])])

  -- | Return network
  pure net

-- | "netToLists" converts network to list-of-lists format
netToLists :: Brain -> [([Double], [[Double]])]
netToLists (Brain layers) =
  [ (VS.toList b, matToLists w) | Layer b w <- layers ]
  -- | Convert each layer's bias and weights to lists

-- | "netFromLists" converts list-of-lists format to network
netFromLists :: [([Double], [[Double]])] -> Brain
netFromLists xs = Brain [ Layer (VS.fromList b) (matFromLists w) | (b, w) <- xs ]
  -- | Convert each layer's bias and weights back to internal types

-- ============================================================================
-- RANDOM SEED MANAGEMENT
-- ============================================================================

-- | "withSeed" sets random seed if provided, then runs action
withSeed :: Maybe Int -> IO a -> IO a
withSeed Nothing action = action                      -- | No seed: run as-is
withSeed (Just seed) action = do                      -- | With seed:
  setStdGen (mkStdGen seed)                          -- | Set random generator
  action                                             -- | Run action
