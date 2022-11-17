

##
##  Packages.
from sklearn import metrics


##
##  Class for metric.
class metric:

    ##  Mean absolute error.
    def mae(target, likelihood):

        score = metrics.mean_absolute_error(y_true=target, y_pred=likelihood)
        return(score)

    ##  Area under curve.
    def auc(target, likelihood):

        score = metrics.roc_auc_score(y_true=target, y_score=likelihood)
        return(score)

    ##  Accuracy rate.
    def ar(target, prediction):

        score = metrics.accuracy_score(y_true=target, y_pred=prediction)
        return(score)

    ##  Cross entropy loss.
    def cel(target, likelihood, label=None):
        
        score = metrics.log_loss(y_true=target, y_pred=likelihood, labels=label)
        return(score)

    ##  Confusion matrix.
    def cm(target, prediction):
        
        table = metrics.confusion_matrix(y_true=target, y_pred=prediction)
        score = str(table.ravel().tolist())
        return(score)

