module TorchLite.Runtime
  ( applyThreads
  ) where

import Control.Concurrent (setNumCapabilities)
import System.Environment (setEnv)

-- Apply thread settings for GHC RTS and common BLAS backends.

applyThreads :: Maybe Int -> IO ()
applyThreads Nothing = pure ()
applyThreads (Just n)
  | n <= 0 = pure ()
  | otherwise = do
      setNumCapabilities n
      setEnv "OPENBLAS_NUM_THREADS" (show n)
      setEnv "OMP_NUM_THREADS" (show n)
      setEnv "MKL_NUM_THREADS" (show n)
