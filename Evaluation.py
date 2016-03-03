import Utils

def evaluate(model, testData, logMessage=False):
	predictions = model.predict(testData.map(lambda item: item.features))
	labelsAndPredictions = testData.map(lambda item: item.label).zip(predictions)
	error = labelsAndPredictions.filter(lambda (v,p): v != p).count()/float(testData.count())

	if logMessage:
		Utils.logMessage('\nTest Error : ' + str(error))

	return error
