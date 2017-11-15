__date__   = "August 2017"

import os
import numpy  as np
# from keras.models import Sequential
# from keras.layers import Conv1D, MaxPool1D, Dropout, Flatten, Dense

NB_CLUSTERS=5
NB_DEPTH=5808
NB_N=20
def mask_cluster (nb_cluster):
    cluster = np.zeros((1,NB_CLUSTERS))
    cluster[0,nb_cluster-1]=1
    return cluster

def lecture_data ( input_file ):
    values_i = np.zeros((NB_N,NB_N,NB_DEPTH))
    cluster_i = np.zeros((NB_DEPTH,NB_CLUSTERS))

    fh = open ( input_file, "r" )
    depth = 0
    l_i = 1
    while l_i<127776:
        ligne = fh.readline()
        nb_arbre,nb_taxon,nb_cluster,percent_noise,nb_eloignement = ligne.split("\t")
        cluster_i[depth]=mask_cluster(int(nb_cluster))
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

if os.path.isfile("brain"):
    X_train, y_train = lecture_data("brain")
    for k in range(X_train.shape[2]):
        X_train[:,:,k]=X_train[:,np.random.permutation(X_train.shape[1]),k]
    X_train = flatten(X_train)
else:
    #==Files not exist:
    print("Files not found")

# In[]
from keras.models import Sequential
from keras.layers import MaxPool1D,Conv1D,Flatten,Dense,Dropout
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

TRAIN=True
SAVE=True
nRows=20
nCols = 20
X_train=X_train.reshape(X_train.shape[0],nRows*nCols,1)
X_test=X_test.reshape(X_test.shape[0],nRows*nCols,1)
NB_FILTERS=20
FILTER_SIZE=2
POOL_SIZE=1
P=0.1
NB_CONV_LAYERS=7

# In[]
model=Sequential()
model.add(Conv1D(filters=NB_FILTERS,
                        kernel_size=FILTER_SIZE,
                        activation='relu',
                        input_shape=(nRows*nCols,1)))

model.add(MaxPool1D(pool_size=POOL_SIZE))
for _ in range(NB_CONV_LAYERS):
    model.add(Dropout(P))
    model.add(Conv1D(filters=NB_FILTERS,
                        kernel_size=FILTER_SIZE,
                        activation='relu'))
    model.add(MaxPool1D(pool_size=POOL_SIZE))
model.add(Flatten())
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','mae'])
if TRAIN:
    model.fit(X_train,y_train,batch_size=64,epochs=10)
    if SAVE:
        model.save_weights('weights.h5')
else:
    print("Weights loaded")
    model.load_weights('weights.h5')
# In[]
from sklearn.metrics import confusion_matrix
predicted=np.argmax(model.predict(X_test),axis=1)
y_test_nums=np.where(y_test==1)[1]
confMat=confusion_matrix(y_test_nums,predicted)
import matplotlib.pyplot as plt
np.sum(np.diag(confMat))/y_test.shape[0]
