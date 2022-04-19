import math
import numpy as np

"""
Function: Calculates the Gaussian Naives Bayes Classifier for the Input Dataset
Static Methods: 3 Static Methods for Pre-Processing, Splitting and Accuracy Scoring
Non-Static Methods: 4 Non-Static Method for fit, posterior calculation, likelihood estimations and predictions
Input: Any Binary dataset with independent dataset (preferably)
"""


class GaussianNB:
    """Initializing the Variables"""

    def __init__(self):

        self.dataSet_features = list
        self.likelihoods = {}
        self.class_priors = {}

        self.X_train = np.array
        self.y_train = np.array
        self.train_size = int
        self.num_feats = int

    """
    :function: Static Method for the Calculation of Accuracy for the predicted output
    :returns: Rounded Float Value of Accuracy Result
    """
    @staticmethod
    def accuracy_score(y_true, y_pred):

        return round(float(sum(y_pred == y_true)) / float(len(y_true)) * 100, 2)

    """
    :function: Static Method for Getting the Target Classes and Input Features
    :returns: Dataframe of Input Variables and Series of Target Classes
    """
    @staticmethod
    def pre_processing(df):

        X = df.drop([df.columns[-1]], axis=1)
        y = df[df.columns[-1]]

        return X, y

    """
    function: Static Method for the Splitting of Test and Train for the X and y Variables
    :returns: x_train: Input features for training; x_test: Input features for testing
    y_train; Target class for training; y_test: Target class for Testing
    """

    @staticmethod
    def train_test_split(x, y, test_train_split=0.25, random_state=None):

        x_test = x.sample(frac=test_train_split, random_state=random_state)
        y_test = y[x_test.index]

        x_train = x.drop(x_test.index)
        y_train = y.drop(y_test.index)

        return x_train, x_test, y_train, y_test

    """
    :function: Tries to fit the Features with the outcome classes of the Training Dataset
    :returns: class posterior calculation and likelihood of the classes
    """

    def fit(self, X, y):

        self.dataSet_features = list(X.columns)
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape[0]
        self.num_feats = X.shape[1]

        # Iterates over the features of the dataset
        for feature in self.dataSet_features:

            # Defining a likelihood dictionary for further use for getting the mean and variance based on the
            # Input feature
            self.likelihoods[feature] = {}

            # Adding the features to the likelihood and class priors dictionaries to add the mean and variance later
            for outcome in np.unique(self.y_train):
                self.likelihoods[feature].update({outcome: {}})
                self.class_priors.update({outcome: 0})

        self._calc_class_prior()
        self._calc_likelihoods()

    """
    :function: calculates the posterior probability 
    :returns: the list of posterior probability outcomes
    """
    def _calc_class_prior(self):

        for outcome in np.unique(self.y_train):
            outcome_count = sum(self.y_train == outcome)
            self.class_priors[outcome] = outcome_count / self.train_size

    """
    
    """
    def _calc_likelihoods(self):

        for feature in self.dataSet_features:

            for target_Class_Feature in np.unique(self.y_train):
                self.likelihoods[feature][target_Class_Feature]['mean'] = self.X_train[feature][
                    self.y_train[self.y_train == target_Class_Feature].index.values.tolist()].mean()

                self.likelihoods[feature][target_Class_Feature]['variance'] = self.X_train[feature][
                    self.y_train[self.y_train == target_Class_Feature].index.values.tolist()].var()

    """
    Function: Predicts Whether the Input Features are for a particular class (target Classes) using Likelihood.
    Returns: The Array of Results for the Likelihood of the Target Classes
    Citation: https://towardsdatascience.com/naive-bayes-explained-9d2b96f4a9c0
    Formula and Basic Understanding have been inspired from the citation given above. 
    """

    def predict(self, X):

        results = []
        X = np.array(X)

        for input_Features in X:
            probs_outcome = {}

            for outcome in np.unique(self.y_train):
                prior = self.class_priors[outcome]
                feature_Likelihood_Estimation = 1

                for dataset_Feature, feature_Values in zip(self.dataSet_features, input_Features):
                    mean = self.likelihoods[dataset_Feature][outcome]['mean']
                    var = self.likelihoods[dataset_Feature][outcome]['variance']
                    feature_Likelihood_Estimation *= (1 / math.sqrt(2 * math.pi * var)) * np.exp(
                        -(feature_Values - mean) ** 2 / (2 * var))

                posterior_numerator = (feature_Likelihood_Estimation * prior)
                probs_outcome[outcome] = posterior_numerator

            result = max(probs_outcome, key=lambda x: probs_outcome[x])
            results.append(result)

        return np.array(results)
