
import sklearn.metrics

class classification:

    def __init__(self, score, prediction, target):

        self.score = score
        self.prediction = prediction
        self.target = target
        return
    
    def accuracy(self):

        measure = sklearn.metrics.accuracy_score
        measurement = measure(y_true=self.target, y_pred=self.prediction)
        result = round(measurement, 3)
        return(result)

    def confusion(self):

        measure = sklearn.metrics.confusion_matrix
        measurement = measure(y_true=self.target, y_pred=self.prediction)
        result = measurement
        return(result)

    def report(self):

        measure = sklearn.metrics.classification_report
        measurement = measure(y_true=self.target, y_pred=self.prediction)
        result = measurement
        return(result)

    def auc(self):

        measure = sklearn.metrics.roc_auc_score
        measurement = measure(y_true=self.target, y_score=self.score[:,0])
        result = measurement
        return(result)

    pass


        # summary['{} target'.format(title)]     = feedback.target
        # summary['{} accuracy'.format(title)]        = sklearn.metrics.accuracy_score(feedback.target, feedback.prediction)
        # summary['{} confusion table'.format(title)] = sklearn.metrics.confusion_matrix(feedback.target, feedback.prediction)
        # summary['{} report'.format(title)]          = sklearn.metrics.classification_report(feedback.target, feedback.prediction)
