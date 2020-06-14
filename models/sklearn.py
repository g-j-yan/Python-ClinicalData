
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def SKLearnKnnClassifier(training, testing, training_labels, testing_labels, k):
    '''leverage Scikit-learn to implement the k nearest neighbors algorithm
    Args:
        training: the subset of data corresponding to the training data as a numpy matrix
        testing:  the subset of data corresponding to the testing data as a numpy matrix
        training_labels: the labels for the training data as a numpy array
        testing_labels: the labels for the testing data as a numpy array
        k: the number of nearest neighbors to use

    This should instantiate the class from scikit learn, then call
    `.fit` and `.predict`. It should then compare results similar to
    NNClassifier to return a % correct.
    
    See how easy it is to use scikit learn?
    '''
    # instantiate model
    knn_model = KNeighborsClassifier(n_neighbors = k)
    # fit model to training data
    knn_model.fit(training, training_labels)
    # predict test labels
    predictions = knn_model.predict(testing)
    # return % where prediction matched actual
    match_rate = sum(predictions == testing_labels) / len(testing_labels)
    return match_rate

def SKLearnSVMClassifier(training, testing, training_labels, testing_labels):
    '''leverage Scikit-learn to implement the support vector machine classifier
    Args:
        training: the subset of data corresponding to the training data as a numpy matrix
        testing:  the subset of data corresponding to the testing data as a numpy matrix
        training_labels: the labels for the training data as a numpy array
        testing_labels: the labels for the testing data as a numpy array

    This should instantiate the class from scikit learn, then call
    `.fit` and `.predict`. It should then compare results similar to
    NNClassifier to return a % correct.
    
    See how easy it is to use scikit learn?
    '''
    # instantiate model
    svm_model = SVC()
    # fit model to training data
    svm_model.fit(training, training_labels)
    # predict test labels
    predictions = svm_model.predict(testing)
    # return % where prediction matched actual
    match_rate = sum(predictions == testing_labels) / len(testing_labels)
    return match_rate
