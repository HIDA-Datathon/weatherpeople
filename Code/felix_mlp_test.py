#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 17:37:54 2020

@author: felix
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

from tensorflow import keras

import matplotlib.pyplot as plt

def to_binary(y_in):
    y_out = y_in>0
    y_out = y_out.astype(float)
    return y_out

def load_data(output_path = '../Data/nao_index_train.npy',
              x1_path = '../Data/tas_train.npy',
              x2_path = '../Data/psl_train.npy'):
    '''
    Parameters
    ----------
    output_path : TYPE, optional
        DESCRIPTION. The default is './Data/nao_index_train.npy'.
    x1_path : TYPE, optional
        DESCRIPTION. The default is './Data/tas_train.npy'.
    x2_path : TYPE, optional
        DESCRIPTION. The default is './Data/psl_train.npy'.

    Returns
    -------

    x1 : North Atlantic and Tropical Atlantic near surface air temperature, 
        October-November average (one vector of dimension M1 per year= instance)
    x2 : North Atlantic sea-level-pressure, October-November average 
        (one vector of dimension M2 per instance), Shape(N,M2)
    y : North Atlantic Oscillation Index

    '''
    
    y = np.load(output_path)
    x1 = np.load(x1_path)
    x2 = np.load(x2_path)
    return x1, x2, y



pca_components = 200

x1, x2, y = load_data()
pca = PCA(n_components=pca_components)

pca.fit(x1)
x1_reduced = pca.transform(x1)

pca.fit(x2)
x2_reduced = pca.transform(x2)

x_combined = np.concatenate((x1_reduced,x2_reduced), axis=1)


x_train, x_test, y_train, y_test = train_test_split(x_combined, y, test_size=0.12, random_state=123)

def create_reg_model(input_shape, no_nodes=[50,50]):
    model = keras.Sequential()
    for i in range(len(no_nodes)):
        # first layer needs to define input shape
        if not i:
            model.add(keras.layers.Dense(no_nodes[i], input_shape=input_shape,activation="relu"))
            model.add(keras.layers.Dropout(0.5))
        else:
            model.add(keras.layers.Dense(no_nodes[i], activation="relu"))
            model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1,activation='linear'))
    model.summary()
    model.compile(optimizer='adam', loss='mse')      
    return model    
    

def create_class_model(input_shape, no_classes, no_nodes=[50,50]):
    model = keras.Sequential()
    for i in range(len(no_nodes)):
        # first layer needs to define input shape
        if not i:
            model.add(keras.layers.Dense(no_nodes[i], input_shape=input_shape,activation="relu"))
            model.add(keras.layers.Dropout(0.5))
        else:
            model.add(keras.layers.Dense(no_nodes[i], activation="relu"))
            model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(no_classes, activation="softmax"))
    model.summary()
    model.compile(optimizer='adam', loss='mse')      
    return model    


batch_size = 200
epochs = 100
no_nodes=[200, 200, 200]

train_y_bin = to_binary(y_train)
test_y_bin = to_binary(y_test)

model = create_reg_model(input_shape = (pca_components*2,),no_nodes=no_nodes)
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test),verbose=1)



# model = create_class_model(input_shape = (pca_components*2,),no_nodes=no_nodes)
# history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test),verbose=1)




train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

print('MSE train: %.3f'%np.mean(np.square(train_pred-y_train)))
print('MSE test: %.3f'%np.mean(np.square(test_pred-y_test)))




train_pred_bin = to_binary(train_pred)
test_pred_bin = to_binary(test_pred)


print('The train accuracy: %.3f'%accuracy_score(train_pred_bin, train_y_bin))
print('The test accuracy: %.3f'%accuracy_score(test_pred_bin, test_y_bin))

