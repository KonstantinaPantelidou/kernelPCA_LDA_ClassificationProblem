import numpy as np

from scipy import stats

from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm, metrics
from sklearn import decomposition
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import neighbors
from sklearn.neighbors import NearestCentroid
from sklearn.decomposition import  KernelPCA
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
import time
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

#The dataset will only retain pictures of people that have at least 60 different pictures
faces=fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)
print(type(faces))


X = faces.data
y = faces.target
X, y = np.float32(faces.data[:1348])/ 255., np.float32(faces.target[:1348])
X, y = shuffle(X,y)
X_train, y_train = np.float32(X[:808])/255., np.float32(y[:808])
X_test, y_test = np.float32(X[540:])/ 255., np.float32(y[540:])




#keep 150 faces as a sample for 90% variance

kernel = ["poly", "rbf"]
n_components=2
gamma = 0.1
i=0
scores = dict()
for gamma in [0.01]:
	for n_components in [ 2]:
		for kernel in ["poly"]:
			kpca = KernelPCA(kernel="poly",n_components=500 , gamma= gamma)
			X_train = kpca.fit_transform(X_train)
			X_test = kpca.transform(X_test)
			#print (kpca)
			#print(X_train.shape)
			lda = LinearDiscriminantAnalysis()
			#print (lda)
			X_train = lda.fit_transform(X_train,y_train)
			X_test = lda.transform(X_test)
			
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

			
			
			#NCC algorithm
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
	
'''
#KNN Classifier

score = []
for k in range(1, 10):
 print('Begin KNN with k=',k)
 clf = KNeighborsClassifier(n_neighbors=k)
 print("Train model")
 clf.fit(X_train, y_train)
 print("Compute predictions")
 y_predicted = clf.predict(X_test)
 accuracy = accuracy_score(y_test, y_predicted)
 score.append(accuracy)
 print("Accuracy: ",accuracy)
 #print("Confusion Matrix:\n", confusion_matrix(y_test, y_predicted))


#print(classification_report(y_predicted, y_test))


plt.plot(range(1,10), score)
plt.title('Determining the Optimal Number of Neighbors')
plt.xlabel('K - Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

#print("Confusion Matrix:\n", confusion_matrix(y_test, y_predicted), "\n")
#print("Classification Report:\n", classification_report(y_predicted, y_test))


#----- Second Way -----

#clf = neighbors.KNeighborsClassifier(n_neighbors=5)
#cv_scores =cross_val_score(clf, X_train,y_train, cv=5)
#print("cv_scores mean:{}".format(np.mean(cv_scores)))

#create new a knn model
clf = neighbors.KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {"n_neighbors": np.arange(1, 10)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(clf, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X_train, y_train)
print(knn_gscv.best_params_)

'''