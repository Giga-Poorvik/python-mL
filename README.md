# python-mL
!pip install mglearn

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mglearn
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

iris_dataframe = pd.DataFrame(x_train, columns=iris_dataset.feature_names)
print(iris_dataframe)
print("x_train shape:{}".format(x_train.shape))
print("x_test shape:{}".format(x_test.shape))
print("y_train shape:{}".format(y_train.shape))
print("y_test shape:{}".format(y_test.shape))

grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(16,16), marker='o', hist_kwds={'bins':20}, s=10, alpha=.8, cmap=mglearn.cm3)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
# After importing ML, fit the training data using .fit()
# check model pass new input

x_new = np.array([[5, 2.9, 1, 0.2]])
print("x_new.shape: {}".format(x_new.shape))

# Once new i/p is passed the model predicts in one of given class
Prediction = knn.predict(x_new)
print("Prediction: {}".format(Prediction))

print("prediction target name: {}".format(
    iris_dataset['target_names'][Prediction]))

y_pred = knn.predict(x_test)
print("Test set predictions:\n {}".format(y_pred))

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(x_test, y_test)))
