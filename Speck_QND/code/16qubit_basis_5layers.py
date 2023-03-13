

import tensorflow as tf
import numpy as np
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import re
import csv
import random
import string
import multiprocessing
import itertools
import random
import csv
import statistics
import string
import random
import pennylane as qml
from tensorflow.keras import regularizers

num_rounds=5
x_train = np.load('./SPECK_NUMPY/x_train.npy')
y_train = np.load('./SPECK_NUMPY/y_train.npy')

x_val = np.load('./SPECK_NUMPY/x_val.npy')
y_val = np.load('./SPECK_NUMPY/y_val.npy')

x_test = np.load('./SPECK_NUMPY/x_test.npy')
y_test = np.load('./SPECK_NUMPY/y_test.npy')

#y_train = y_train.reshape((len(y_train),1))
#y_val = y_val.reshape((len(y_val),1))
#y_test = y_test.reshape((len(y_test),1))

import pennylane as qml
from pennylane import numpy as np

n_qubits = 16

tf.keras.backend.set_floatx('float64')

#dev = qml.device("default.qubit", wires=n_qubits)
dev = qml.device('lightning.qubit', wires=n_qubits)
#dev = qml.device('qulacs.simulator', wires=n_qubits) 

@qml.qnode(dev, diff_method="adjoint")
def qnode(inputs, weights):
    qml.BasisEmbedding(inputs, wires=range(n_qubits))
    qml.RandomLayers(weights=weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

n_layers = 5
weight_shapes = {"weights": (n_layers, n_qubits)}
qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

q1 = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)
q2 = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)
q3 = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)
q4 = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

prediction1_fc = tf.keras.layers.Dense(64, kernel_regularizer=regularizers.L2(0.00001)) 
prediction1_bn = tf.keras.layers.BatchNormalization() 
prediction1_relu = tf.keras.layers.Activation('relu') 

output_layer = keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.L2(0.00001))

# construct the model
inputs = tf.keras.Input(shape=(64,))
#x = clayer_1(inputs)

x1, x2, x3, x4 = tf.split(inputs, 4, axis=1)

x1 = q1(x1)
x2 = q2(x2)
x3 = q3(x3)
x4 = q4(x4)

x = tf.concat([x1, x2, x3, x4], axis=1)



outputs = output_layer(x)

q1.build((32,16))
q2.build((32,16))
q3.build((32,16))
q4.build((32,16))

model = tf.keras.Model(inputs=inputs, outputs=outputs)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(opt, loss="binary_crossentropy", metrics=["acc"])

model.summary()

fitting = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data = (x_val, y_val), verbose=2)

predicted_labels = model.predict(np.array(x_test))

res = np.array(predicted_labels > 0.5, dtype = int)

res = res.reshape((len(x_test)))
y_test = y_test.reshape((len(y_test)))

total = 0

for i in range(len(res)):
    if res[i] == y_test[i]:
        total += 1

print(total/len(x_test))

