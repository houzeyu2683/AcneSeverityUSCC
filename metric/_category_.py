
import sklearn.metrics

class Category:

    def __init__(self, score, prediction, target):

        self.score = score
        self.prediction = prediction
        self.target = target
        return
    
    def getAccuracy(self):

        measure = sklearn.metrics.accuracy_score
        measurement = measure(y_true=self.target, y_pred=self.prediction)
        result = float(round(measurement, 3))
        # result = measurement
        return(result)

    def getConfusionTable(self):

        measure = sklearn.metrics.confusion_matrix
        measurement = measure(y_true=self.target, y_pred=self.prediction)
        result = measurement.tolist()
        return(result)

    def getReport(self):

        measure = sklearn.metrics.classification_report
        measurement = measure(y_true=self.target, y_pred=self.prediction)
        result = measurement
        return(result)

    def getAreaUnderCurve(self):

        level = set(self.target)
        assert len(level)==2, 'More than 2 classification.'
        measure = sklearn.metrics.roc_auc_score
        measurement = measure(y_true=self.target, y_score=self.score[:,1])
        result = float(round(measurement, 3))
        return(result)

    pass

