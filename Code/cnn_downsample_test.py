#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 17:37:54 2020

@author: felix
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow import keras

import seaborn as sn
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




x1, x2, y = load_data()
x1 = x1.reshape((900,54,43))
x2 = x2.reshape((900,54,43))

x_combined = np.concatenate((np.expand_dims(x1,axis=3),np.expand_dims(x2,axis=3)), axis = 3)



### Categorize NAO into 3 categories #####
med_ids = np.where(np.logical_and(-1.4<y, y<1.4))[0]
y[y>=1.4] = 1. 
y[y<=-1.4] = -1.
y[med_ids] = 0.

### One hot encoding of targets ###
one_hot = np.zeros((len(y), 3))
one_hot[y==1.] = [1, 0, 0]
one_hot[y==-1.] = [0, 0, 1]
one_hot[y==0.] = [0, 1, 0]


num_classes = 3
input_shape = (54,43,2)

x_train, x_test, y_train, y_test = train_test_split(x_combined, one_hot, test_size=0.12, random_state=1)

print('train set')
print(y_train.sum(axis=0))
print('test set')
print(y_test.sum(axis=0))

    

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        keras.layers.Conv2D(32, kernel_size=(10, 10), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

batch_size = 100
epochs = 30

# history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test),verbose=1)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

extreme_train_indcs = np.logical_or(y_train[:,0],y_train[:,2])
no_extreme = sum(extreme_train_indcs)
x_train_extreme = x_train[extreme_train_indcs,:,:,:]
y_train_extreme = y_train[extreme_train_indcs,:]

no_training = y_train.shape[0]



for i in range(epochs):
    print("Epoch %d / %d"%(i+1,epochs))
    idcs = np.random.choice(np.arange(no_training),size=int(no_extreme/2))
    x_train_add = x_train[idcs,:,:,:]
    y_train_add = y_train[idcs,:]
    x_temp = np.concatenate((x_train_extreme,x_train_add))
    y_temp = np.concatenate((y_train_extreme,y_train_add))


    history = model.fit(x_temp, y_temp, batch_size=batch_size, epochs=1, validation_data=(x_test,y_test),verbose=1)




# model = create_class_model(input_shape = (pca_components*2,),no_nodes=no_nodes)
# history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test),verbose=1)


# =============================================================================
#           Evaluate
# =============================================================================

train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

train_pred = train_pred.argmax(axis=1)
test_pred = test_pred.argmax(axis=1)

train_true = y_train.argmax(axis=1)
test_true = y_test.argmax(axis=1)

conf_train = confusion_matrix(train_true, train_pred)
conf_test = confusion_matrix(test_true, test_pred)

df_cm = pd.DataFrame(conf_train, index = [0,1,2],
                  columns = [0,1,2])
plt.figure(figsize = (4,4))

sn.heatmap(df_cm, cmap="YlGnBu",  annot=True, fmt="d", cbar=False)
plt.title("Train confusion matrix")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()


df_cm = pd.DataFrame(conf_test, index = [0,1,2],
                  columns = [0,1,2])
plt.figure(figsize = (4,4))

sn.heatmap(df_cm, cmap="YlGnBu",  annot=True, fmt="d", cbar=False)
plt.title("Test confusion matrix")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()