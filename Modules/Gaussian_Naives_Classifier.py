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
        self.likelihoods_list = {}
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
            self.likelihoods_list[feature] = {}

            # Adding the features to the likelihood and class priors dictionaries to add the mean and variance later
            for outcome in np.unique(self.y_train):
                self.likelihoods_list[feature].update({outcome: {}})
                self.class_priors.update({outcome: 0})

        # Call the Method for
        self.calculation_Class_Posterior()
        self.calculation_Class_Likelihood()

    """
    :function: calculates the posterior probability 
    :returns: the list of posterior probability outcomes for the given input
    """
    def calculation_Class_Posterior(self):

        for outcome in np.unique(self.y_train):
            outcome_count = sum(self.y_train == outcome)
            self.class_priors[outcome] = outcome_count / self.train_size

    """
    :function: calculates the maximum possible likelihood of the features
    :returns: list of the likelihood based on the mean and variance
    """
    def calculation_Class_Likelihood(self):

        for feature in self.dataSet_features:

            for target_Class_Feature in np.unique(self.y_train):

                # Getting the likelihoods of input variable "Mean"
                self.likelihoods_list[feature][target_Class_Feature]['mean'] = self.X_train[feature][
                    self.y_train[self.y_train == target_Class_Feature].index.values.tolist()].mean()

                # Getting the likelihoods of input variable "Variance"
                self.likelihoods_list[feature][target_Class_Feature]['variance'] = self.X_train[feature][
                    self.y_train[self.y_train == target_Class_Feature].index.values.tolist()].var()

    """
    Function: Predicts Whether the Input Features are for a particular class (target Classes) using Likelihood.
    Returns: The Array of Results for the Likelihood of the Target Classes
    Citation: https://towardsdatascience.com/naive-bayes-explained-9d2b96f4a9c0
    Formula and Basic Understanding have been inspired from the citation given above. 
    """

    def predict(self, X):

        posterior_Probability_Function_Outcome = []
        X = np.array(X)

        # Iterating thru the Input features of X
        for input_Features in X:

            # Storing the Posterior Outcome
            probs_outcome = {}

            for feature_outcome in np.unique(self.y_train):
                prior = self.class_priors[feature_outcome]
                feature_Likelihood_Estimation = 1

                # Getting the Mean and Variance for the particular feature feature_outcome
                for dataset_Feature, feature_Values in zip(self.dataSet_features, input_Features):

                    # Getting the Value of Mean and Variance
                    mean_Input_Variables = self.likelihoods_list[dataset_Feature][feature_outcome]['mean']
                    variance_Input_Variables = self.likelihoods_list[dataset_Feature][feature_outcome]['variance']

                    # Calculating the Probability Density Function for the given features for the feature_outcome label
                    feature_Likelihood_Estimation *= (1 / math.sqrt(2 * math.pi * variance_Input_Variables)) * np.exp(
                        -(feature_Values - mean_Input_Variables) ** 2 / (2 * variance_Input_Variables))

                # Calculating the Posterior numerator with the Probability Density Function
                posterior_numerator = (feature_Likelihood_Estimation * prior)
                probs_outcome[feature_outcome] = posterior_numerator

            # Getting the final result of the Probability Outcome
            result_probability_list = max(probs_outcome, key=lambda x: probs_outcome[x])

            # Appending the results into the List
            posterior_Probability_Function_Outcome.append(result_probability_list)

        return np.array(posterior_Probability_Function_Outcome)
