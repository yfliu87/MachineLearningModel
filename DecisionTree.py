import Utils
import Evaluation
from pyspark import SparkConf, SparkContext, mllib
from pyspark.mllib.tree import DecisionTree
import evaluation


def trainModel(trainingData):
	print '\nTraining Decision Tree model started'
	Utils.logTime()

	model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5,maxBins=32)
	print '\nTraining Decision Tree model finished'
	Utils.logTime()
	return model


def trainOptimalModel(trainingData, testData):
	print "\nTraining optimal Decision Tree model started!"
	Utils.logTime()

	impurityVals = ['gini', 'entropy']
	maxDepthVals = [3,4,5,6,7]
	maxBinsVals = [8,16,32]

	optimalModel = None
	optimalMaxDepth = None
	optimalImpurity = None
	optimalBinsVal = None
	minError = None

	try:
		for curImpurity in impurityVals:
			for curMaxDepth in maxDepthVals:
				for curMaxBins in maxBinsVals:
					model = DecisionTree.trainClassifier(trainingData, 
														 numClasses=2, 
														 categoricalFeaturesInfo={}, 
														 impurity=curImpurity, 
														 maxDepth=curMaxDepth,
														 maxBins=curMaxBins)
					testErr, PR, ROC = Evaluation.evaluate(model, testData)
					if testErr < minError or not minError:
						minError = testErr
						optimalImpurity = curImpurity
						optimalMaxDepth = curMaxDepth
						optimalBinsVal = curMaxBins
						optimalModel = model
	except:
		msg = "\nException during model training with below parameters:"
		msg += "\timpurity: " + str(curImpurity)
		msg += "\tmaxDepth: " + str(curMaxDepth)
		msg += "\tmaxBins: " + str(curMaxBins)
		Utils.logMessage(msg)

	logMessage(optimalModel, optimalMaxDepth, optimalImpurity, optimalBinsVal, minError)
	return optimalModel 


def logMessage(optimalModel,optimalMaxDepth, optimalImpurity, optimalBinsVal, minError):

	msg = "\nTraining optimal Decision Tree model finished:"
	msg += "\tMin Test Error : " + str(minError)
	msg += "\toptimal impurity : " + str(optimalImpurity)
	msg += "\toptimal max depth : " + str(optimalMaxDepth)
	msg += "\toptimal bins val : " + str(optimalBinsVal)
	Utils.logMessage(msg)
	Utils.logTime()
	#print "\toptimal model : " + optimalModel.toDebugString()


