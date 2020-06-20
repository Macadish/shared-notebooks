# %%
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# %% Function
f = lambda x,y:x**2/(x**2+x*y)
#f = lambda x,y:np.sin(x)

# %% Plot functon
x = np.arange(1, 4, 0.1)
y = x
xx, yy = np.meshgrid(x,y)
zz = f(xx, yy)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
fig


# %% Generate Training Data
x_train = np.random.uniform(0.1, 4, 100000)
y_train = np.random.uniform(0.1, 4, 100000)
#xx_train, yy_train = np.meshgrid(x_train,y_train)
ff_train = f(x_train, y_train)
ff_train.max()
x_train.min()
y_train.min()

# %% Plot Training Data
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(y_train, x_train, ff_train, cmap=plt.cm.viridis, linewidth=0.2)
fig


# %% Test data
x_test = np.random.normal(0, 1, 100)
y_test = np.random.normal(0, 1, 100)
#xx_test, yy_test = np.meshgrid(x_test,y_test)
ff = f(x_test, y_test)

# %% Train
model = Sequential()
model.add(Dense(20, activation='relu', input_dim=2))
model.add(LeakyReLU(alpha=0.03))
model.add(Dense(20, activation='relu'))
model.add(LeakyReLU(alpha=0.03))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse')
model.fit(np.vstack([x_train,y_train]).transpose(), ff_train, epochs= 20, batch_size=100)

# %% Preict
model.predict(np.vstack([x_test,y_test]).transpose())
