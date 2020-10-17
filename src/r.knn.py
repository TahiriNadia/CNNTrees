# k-nearest neighbors (K-NN)

__date__   = "August 2020"
__author__   = "Nadia Tahiri"


# logistic regression
# Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import os
import numpy  as np

NB_CLUSTERS=5
NB_DEPTH=5808
NB_N=20

def lecture_data(input_file):
    values_i = np.zeros((NB_N,NB_N,NB_DEPTH))
    cluster_i = np.zeros((NB_DEPTH))

    fh = open(input_file, "r")
    depth = 0
    l_i = 1
    while l_i<127776:
        ligne = fh.readline()
        nb_arbre,nb_taxon,nb_cluster,percent_noise,nb_eloignement = ligne.split("\t")
        cluster_i[depth]=int(nb_cluster)
        for i in range(0, int(nb_arbre)):
            ligne = fh.readline()
            values_i [i,:,depth] = ligne.split("\t")

        ligne = fh.readline()
        depth = depth + 1
        l_i = l_i + int(nb_arbre) + 2
    fh.close
    return values_i,cluster_i

def flatten(x):
    x=np.transpose(x,(2,0,1))
    flatX=np.zeros((x.shape[0],x.shape[1]*x.shape[2]))
    for p in range(x.shape[0]):
        flatX[p]=np.ndarray.flatten(x[p])
    return flatX

if os.path.isfile("../data/simulation/simulation_dataset"):
    X_train, y_train = lecture_data("../data/simulation/simulation_dataset")
    for k in range(X_train.shape[2]):
        X_train[:,:,k]=X_train[:,np.random.permutation(X_train.shape[1]),k]
    X_train = flatten(X_train)
else:
    #==Files not exist:
    print("Files not found")


from keras.models import Sequential
from keras.layers import MaxPool1D,Conv1D,Flatten,Dense,Dropout
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

TRAIN = True
SAVE = True
nRows = 20
nCols = 20
X_train = X_train.reshape(X_train.shape[0],nRows*nCols,1)
X_test = X_test.reshape(X_test.shape[0],nRows*nCols,1)
NB_FILTERS = 20
FILTER_SIZE = 2
POOL_SIZE = 1
P = 0.1
NB_CONV_LAYERS = 7

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

   
x_test = X_test[:, :, 0]
x_train = X_train[:, :, 0]


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)#for euclidean dist,choose minkowski power=2
classifier.fit(x_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# making the confusion matrix
# calculate Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# calculate Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %f" % accuracy)

# calculate Recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred, average='weighted')

# calculate Precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, average='macro')

# Method 1: sklearn calculate F1
from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='macro')
# Method 2: Manual Calculation
F1 = 2 * (precision * recall) / (precision + recall)

labels = y_test
predicitons = y_pred

print("Precision (micro): %f" % precision_score(labels, predicitons, average='micro'))
print("Recall (micro):    %f" % recall_score(labels, predicitons, average='micro'))
print("F1 score (micro):  %f" % f1_score(labels, predicitons, average='micro'), end='\n\n')
print("Precision (macro): %f" % precision_score(labels, predicitons, average='macro'))
print("Recall (macro):    %f" % recall_score(labels, predicitons, average='macro'))
print("F1 score (macro):  %f" % f1_score(labels, predicitons, average='macro'), end='\n\n')
print("Precision (weighted): %f" % precision_score(labels, predicitons, average='weighted'))
print("Recall (weighted):    %f" % recall_score(labels, predicitons, average='weighted'))
print("F1 score (weighted):  %f" % f1_score(labels, predicitons, average='weighted'))
