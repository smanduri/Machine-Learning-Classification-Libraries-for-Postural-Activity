from DataSetReader import DataSetReader
from Modules.Bernoulli_Naives_Classifier import GaussianNB
import numpy as np
import pandas as pd

if __name__ == "__main__":

    # Testing the Return Function of the DatasetReader.py
    #data = DataSetReader.getCsvDataset()

    data = pd.read_csv("dataset/Iris.csv")

    # Split fearures and target
    X, y = GaussianNB.pre_processing(data)

    # Split data into Training and Testing Sets
    X_train, X_test, y_train, y_test = GaussianNB.train_test_split(X, y, test_size=0.1, random_state=0)

    # print(X_train, y_train)

    gnb_clf = GaussianNB()
    gnb_clf.fit(X_train, y_train)

    # print(X_train, y_train)

    print("Train Accuracy: {}".format(GaussianNB.accuracy_score(y_train, gnb_clf.predict(X_train))))
    print("Test Accuracy: {}".format(GaussianNB.accuracy_score(y_test, gnb_clf.predict(X_test))))

    # Query 1:
    query = np.array([[5.7, 2.9, 4.2, 1.3]])
    print("Query 1:- {} ---> {}".format(query, gnb_clf.predict(query)))