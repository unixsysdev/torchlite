module Main where

import System.Environment (getArgs)

import TorchLite.Grokking (parseArgs, runGrokking)

main :: IO ()
main = do
  cfg <- parseArgs <$> getArgs
  runGrokking cfg
