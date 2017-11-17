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

# In[]
# create model
model=Sequential()
model.add(Conv1D(filters=NB_FILTERS,
                        kernel_size=FILTER_SIZE,
                        activation='sigmoid',
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

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','mae'])
if TRAIN:
    # Fit the model
    history = model.fit(X_train,y_train,validation_split=0.33,batch_size=64,epochs=150)
    if SAVE:
        model.save_weights('weights.h5')
else:
    print("Weights loaded")
    model.load_weights('weights.h5')

# evaluate the model
scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')


plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# In[]
from sklearn.metrics import confusion_matrix
predicted=np.argmax(model.predict(X_test),axis=1)
y_test_nums=np.where(y_test==1)[1]
confMat=confusion_matrix(y_test_nums,predicted)
import matplotlib.pyplot as plt
np.sum(np.diag(confMat))/y_test.shape[0]

# In[]
from matplotlib import pyplot
from math import cos, sin, atan


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment))
        pyplot.gca().add_line(line)

    def draw(self, layerType=0):
        for neuron in self.neurons:
            neuron.draw( self.neuron_radius )
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize = 12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize = 12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 12)

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons ):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)

    def draw(self):
        pyplot.figure()
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw( i )
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title( 'Neural Network architecture', fontsize=15 )
        pyplot.show()

class DrawNN():
    def __init__( self, neural_network ):
        self.neural_network = neural_network

    def draw( self ):
        widest_layer = max( self.neural_network )
        network = NeuralNetwork( widest_layer )
        for l in self.neural_network:
            network.add_layer(l)
        network.draw()
        
# In[]
net_profile = []
net_profile.insert(0,5)

for i in range (0,NB_CONV_LAYERS):
    net_profile.insert(i+1,7)

net_profile.insert(NB_CONV_LAYERS+1,5)
network = DrawNN(net_profile)
network.draw()
