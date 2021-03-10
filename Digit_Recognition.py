import numpy as np
from sklearn.decomposition import PCA
#from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn import decomposition
from sklearn.model_selection import cross_val_score
import time # computation time benchmark
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import neighbors
from sklearn.neighbors import NearestCentroid
from sklearn.decomposition import  KernelPCA
from sklearn.utils import shuffle
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
import time
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# Fetching the dataset from 
# mldata.org/repository/data/viewslug/mnist-original through the sklearn helper


mnist = fetch_openml('mnist_784')

print("Data shape ")
print(mnist.data.shape) 

print("Number of classes")
print(np.unique(mnist.target))



# Keeping 60k out of 70k as train-set
X, y = np.float32(mnist.data[:70000])/ 255., np.float32(mnist.target[:70000])
X, y = shuffle(X,y)
#15k training samples
X_train, y_train = np.float32(X[:15000])/255., np.float32(y[:15000])
X_test, y_test = np.float32(X[60000:])/ 255., np.float32(y[60000:])

print (mnist.data.shape)

'''
model = SVC(C=10, gamma=0.01, kernel="poly")
start = time.time()
model.fit(X_train, y_train)
stop = time.time()
print(f"Training time: {stop - start}s")

y_pred = model.predict(X_test)
print("Accuracy Score - test dataset:", accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

y_predicted_train = model.predict(X_train)
print("Accuracy Score - train dataset:",accuracy_score(y_train, y_predicted_train) )
'''


kernel = ["poly", "rbf"]
n_components=2
gamma = 0.1
i=0
scores = dict()
for gamma in [0.001, 0.01, 0.1]:
	for n_components in [2,150,300, 500, 1000]:
		for kernel in ["poly", "rbf"]:
			kpca = KernelPCA(kernel=kernel, n_components=n_components , gamma= gamma)
			X_train = kpca.fit_transform(X_train)
			X_test = kpca.transform(X_test)
			#print (kpca)
			#print(X_train.shape)
			lda = LinearDiscriminantAnalysis()
			#print (lda)
			X_train = lda.fit_transform(X_train,y_train)
			X_test = lda.transform(X_test)
			

			# ------- ΚΝΝ Αlgorithm -------
			clf = neighbors.KNeighborsClassifier(n_neighbors=8)
			start = time.time()
			clf.fit(X_train, y_train)
			stop = time.time()
			print(f"Training time: {stop - start}s")
			y_predicted = clf.predict(X_test)		
			accuracy_test = accuracy_score(y_test, y_predicted)
			print("Accuracy Score - test dataset:", accuracy_test, "\n")
			y_predicted_train = clf.predict(X_train)
			accuracy_train = accuracy_score(y_train, y_predicted_train)
			print("Accuracy Score - train dataset:", accuracy_train, "\n" )
			# using the subscript notation 
			# Dictionary_Name[New_Key_Name] = New_Key_Value 
			scores['accuracy_test'] = [gamma, n_components, kernel, accuracy_test, accuracy_train]
			print(scores)

			print("Confusion Matrix:\n", confusion_matrix(y_test, y_predicted))

			print("Classification Report:\n",classification_report(y_predicted, y_test))
			
			
			# ------- NCC Αlgorithm -------
			nc = NearestCentroid()
			start = time.time()
			nc.fit(X_train, y_train)
			stop = time.time()
			print(f"Training time: {stop - start}s")
			y_predicted = nc.predict(X_test)		
			accuracy_test = accuracy_score(y_test, y_predicted)
			print("Accuracy Score - test dataset:", accuracy_test, "\n")
			y_predicted_train = nc.predict(X_train)
			accuracy_train = accuracy_score(y_train, y_predicted_train)
			print("Accuracy Score - train dataset:", accuracy_train, "\n" )
			# using the subscript notation 
			# Dictionary_Name[New_Key_Name] = New_Key_Value 
			scores['accuracy_test'] = [gamma, n_components, kernel, accuracy_test, accuracy_train]
			print(scores)

			print("Confusion Matrix:\n", confusion_matrix(y_test, y_predicted))

			print("Classification Report:\n",classification_report(y_predicted, y_test))
			
#print(scores['accuracy_test'])
			

#kNN classification

clf = neighbors.KNeighborsClassifier(n_neighbors=5)
cv_scores =cross_val_score(clf, X_train,y_train, cv=5)
print("cv_scores mean:{}".format(np.mean(cv_scores)))

#create new a knn model
clf = neighbors.KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {"n_neighbors": np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(clf, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X_train, y_train)
print(knn_gscv.best_params_)

clf = neighbors.KNeighborsClassifier(n_neighbors=10)
start = time.time()
clf.fit(X_train, y_train)
stop = time.time()
print(f"Training time: {stop - start}s")

y_predicted = clf.predict(X_test)
print("Accuracy Score - test dataset:", metrics.accuracy_score(y_test,y_predicted), "\n")

y_predicted_train = clf.predict(X_train)
print("Accuracy Score - train dataset:",metrics.accuracy_score(y_train, y_predicted_train), "\n" )

print("Confusion Matrix:\n", confusion_matrix(y_test, y_predicted), "\n")
print("Classification Report:\n", classification_report(y_predicted, y_test))



