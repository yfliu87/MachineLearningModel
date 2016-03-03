import Utils
import Evaluation
from pyspark import SparkConf, SparkContext, mllib
from pyspark.mllib.tree import RandomForest

def trainModel(trainingData):
	print "\nTrainning Random Forest model started!"
	Utils.logTime()

	model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, 
											numTrees=3, featureSubsetStrategy="auto", impurity='gini',
											maxDepth=5, maxBins=32)

	print '\nTraining Random Forest model finished'
	Utils.logTime()
	return model


def trainOptimalModel(trainingData, testData):
	print "\nTraining optimal Random Forest model started!"
	Utils.logTime()

	numTreesVals = [3,5,8]
	featureSubsetStrategyVals = ['auto','all','sqrt','log2','onethird']
	impurityVals = ['gini', 'entropy']
	maxDepthVals = [3,4,5,6,7]
	maxBinsVals = [8,16,32]

	optimalModel = None
	optimalNumTrees = None
	optimalFeatureSubsetStrategy = None
	optimalMaxDepth = None
	optimalImpurity = None
	optimalBinsVal = None
	minError = None

	try:
		for curNumTree in numTreesVals:
			for curFeatureSubsetStrategy in featureSubsetStrategyVals:
				for curImpurity in impurityVals:
					for curMaxDepth in maxDepthVals:
						for curMaxBins in maxBinsVals:
							model = RandomForest.trainClassifier(trainingData, 
																numClasses=2, 
																categoricalFeaturesInfo={}, 
														 		numTrees=curNumTree,
														 		featureSubsetStrategy=curFeatureSubsetStrategy,
														 		impurity=curImpurity, 
														 		maxDepth=curMaxDepth,
														 		maxBins=curMaxBins)
							testErr = Evaluation.evaluate(model, testData)
							if testErr < minError or not minError:
								minError = testErr
								optimalNumTrees = curNumTree
								optimalFeatureSubsetStrategy = curFeatureSubsetStrategy
								optimalImpurity = curImpurity
								optimalMaxDepth = curMaxDepth
								optimalBinsVal = curMaxBins
								optimalModel = model
	except:
		msg = "\nException during model training with below parameters:"
		msg += "\tnum trees: " + str(optimalNumTrees)
		msg += "\tfeature subset strategy: " + optimalFeatureSubsetStrategy
		msg += "\timpurity: " + str(curImpurity)
		msg += "\tmaxDepth: " + str(curMaxDepth)
		msg += "\tmaxBins: " + str(curMaxBins)
		Utls.logMessage(msg)

	logMessage(optimalModel, optimalNumTrees, optimalFeatureSubsetStrategy, optimalMaxDepth, optimalImpurity, optimalBinsVal, minError)
	return optimalModel 


def logMessage(optimalModel,optimalNumTrees, optimalFeatureSubsetStrategy, optimalMaxDepth, optimalImpurity, optimalBinsVal, minError):

	msg = "\nTraining optimal Random Forest model finished:"
	msg += "\tMin Test Error : " + str(minError)
	msg += "\toptimal num trees : " + str(optimalNumTrees)
	msg += "\toptimal feature subset strategy: " + str(optimalFeatureSubsetStrategy)
	msg += "\toptimal impurity : " + str(optimalImpurity)
	msg += "\toptimal max depth : " + str(optimalMaxDepth)
	msg += "\toptimal bins val : " + str(optimalBinsVal)
	Utils.logMessage(msg)s
	Utils.logTime()
	#print "\toptimal model : " + optimalModel.toDebugString()