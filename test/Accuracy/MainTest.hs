module Main where
import NNet
import Test.Tasty
import Test.Tasty.HUnit
import System.IO.Unsafe
--import Test.Framework.Providers.QuickCheck2 (testProperty)

ftest_imageF = "data/t10k-images-idx3-ubyte.gz"
ftest_labelF = "data/t10k-labels-idx1-ubyte.gz"
train_imageF = "data/train-images-idx3-ubyte.gz"
train_labelF = "data/train-labels-idx1-ubyte.gz"

fnnF = readNNFile "data/eg-trained-model.txt"

trainCfg :: TrainConfig
trainCfg = TrainConfig
  { trainImages = train_imageF
  , trainLabels = train_labelF
  , trainLayers = [784, 30, 10]
  , trainLearningRate = 0.002
  , trainEpochs = 1
  , trainSamples = Just 10000
  , trainSeed = Nothing
  }

testCfg :: TestConfig
testCfg = TestConfig
  { testImages = ftest_imageF
  , testLabels = ftest_labelF
  , testSamples = Just 10000
  }

test_NN = do
  net <- train trainCfg
  testNN testCfg net
--main = htfMain htf_thisModulesTests
{-main = do
    putStrLn "This test always fails!"
    exitFailure-}

main = 
  do
    val <- test_NN
    defaultMain $
      testGroup "Tests"
        [ testGroup "Checking if the NNet is working well enough"
            [ 
                  testCase ("Accuracy test: " ++ show val ++ "%") $
                  val `compare` 74 @?= GT 

            ]
        ]
