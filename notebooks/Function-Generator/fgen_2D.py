# %%
from numpy.random import seed
seed(123)
#from tensorflow import set_random_seed
#set_random_seed(234)

import sklearn
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import decomposition
import scipy

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Layer, InputSpec, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers, activations, initializers, constraints, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import UnitNorm, Constraint

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# %% Function
#f = lambda x,y:x**2/(x**2+x*y)
f = lambda x,y:np.sin(x)

# %% Plot functon
x = np.arange(-4, 4, 0.1)
y = x
xx, yy = np.meshgrid(x,y)
zz = f(xx, yy)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
fig

# %% Generate Data
# Training Data (Without Noise)
x_train = np.random.uniform(-4, 4, 10000)
y_train = np.random.uniform(-4, 4, 10000)
ff_train = f(x_train, y_train)

train_in = np.array([x_train, y_train]).T
train_out = np.array([ff_train]).T

# Test data
x_test = np.random.uniform(-4, 4, 10000)
y_test = np.random.uniform(-4, 4, 10000)
#xx_test, yy_test = np.meshgrid(x_test,y_test)
ff_test = f(x_test, y_test)

test_in = np.array([x_test, y_test]).T
test_out = np.array([ff_test]).T

# Preprocess
#X_train, X_test = train_test_split(X, test_size=0.01, random_state=123)
#scaler = StandardScaler() #Translate data and scale them such that stddev is 1
#scaler.fit(X_train)
#X_train_scaled = scaler.transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# Plot Training Data
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(y_train, x_train, ff_train, cmap=plt.cm.viridis, linewidth=0.2)
fig

# %%
# Set up
nb_epoch = 100 #Number of runs
batch_size = 16
input_dim = 2 #Number of predictor variables,
learning_rate = 1e-2

# Batch and shuffle data
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_in, train_out)).shuffle(10000).batch(batch_size)

# %% Train
tf.keras.backend.clear_session()
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=2))
model.add(LeakyReLU(alpha=0.03))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='rmsprop', loss='mse')
model.fit(train_ds, epochs= 20)

# %% Preict
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(y_test, x_test, model.predict(test_in).T[0], cmap=plt.cm.viridis, linewidth=0.2)
fig

len(model.predict(test_in).T[0])
