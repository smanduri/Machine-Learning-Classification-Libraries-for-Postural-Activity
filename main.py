from DataSetReader import DataSetReader
from Modules.Bernoulli_Naives_Classifier import GaussianNB
import numpy as np
import pandas as pd

if __name__ == "__main__":

    # Testing the Return Function of the DatasetReader.py

    # Deprecated Use for the Multi-Classifier Dataset
    # data = DataSetReader.getCsvDataset()

    data = pd.read_csv("dataset/Iris.csv")

    # Split features and target
    X, y = GaussianNB.pre_processing(data)

    # Split data into Training and Testing Sets
    X_train, X_test, y_train, y_test = GaussianNB.train_test_split(X, y, test_train_split=0.5)

    gnb_clf = GaussianNB()
    gnb_clf.fit(X_train, y_train)

    print("Train Accuracy: {}".format(GaussianNB.accuracy_score(y_train, gnb_clf.predict(X_train))))

    # Query 1:
    predict_target_class = np.array([[5.7, 2.9, 4.2, 1.3]])
    print("Test Feature Input 1:- {} ---> {}".format(predict_target_class, gnb_clf.predict(predict_target_class)))