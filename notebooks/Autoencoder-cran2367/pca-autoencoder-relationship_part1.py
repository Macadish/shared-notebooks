# %%
from numpy.random import seed
seed(123)
#from tensorflow import set_random_seed
#set_random_seed(234)

import sklearn
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import decomposition
import scipy

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Layer, InputSpec
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers, activations, initializers, constraints, Sequential
from keras import backend as K
from keras.constraints import UnitNorm, Constraint

# %%
tf.__version__

# %% [markdown]
# # Generate random multi-dimensional correlated data

# %% [markdown]
# **Step 1**. Set the dimension of the data.
#
# We set the dim small to clear understanding.

# %%
n_dim = 5

# %% [markdown]
# **Step 2.1.** Generate a positive definite symmetric matrix to be used as covariance to generate a random data.
#
# This is a matrix of size n_dim x n_dim.

# %%
cov = sklearn.datasets.make_spd_matrix(n_dim, random_state=None)

# %% [markdown]
# **Step 2.2.** Generate a vector of mean for generating the random data.
#
# This is an np array of size n_dim.

# %%
mu = np.random.normal(0, 0.1, n_dim)

# %% [markdown]
# **Step 3**. Generate the random data, `X`.
#
# The number of samples for `X` is set as `n`.

# %%
n = 1000

X = np.random.multivariate_normal(mu, cov, n)

# %% [markdown]
# **Step 4.** Split the data into train and test.
#
# We split the data into train and test. The test will be used to measure the improvement in Autoencoder after tuning.

# %%
X_train, X_test = train_test_split(X, test_size=0.5, random_state=123)

# %% [markdown]
# # Data preprocessing

# %%
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

# %%
X_train_scaled

# %% [markdown]
# -----------

# %% [markdown]
# # PCA vs Single Layer Linear Autoencoder

# %% [markdown]
# ### Fit Principal Component Analysis (PCA)

# %%
pca = decomposition.PCA(n_components=2)

pca.fit(X_train_scaled)

# %% [markdown]
# ### Fit Single Layer Linear Autoencoder

# %%
nb_epoch = 100
batch_size = 16
input_dim = X_train_scaled.shape[1] #num of predictor variables,
encoding_dim = 2
learning_rate = 1e-3

encoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias = True)
decoder = Dense(input_dim, activation="linear", use_bias = True)

autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer='sgd')
autoencoder                    
autoencoder.summary()

autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=nb_epoch,
                batch_size=batch_size,
                shuffle=True)

# %% [markdown]
# Compare and contrast the outputs.
#
# ### 1. Tied Weights
#
# The weights on Encoder and Decoder are not the same.

# %%
w_encoder = np.round(autoencoder.layers[0].get_weights()[0], 2).T  # W in Figure 2.
w_decoder = np.round(autoencoder.layers[1].get_weights()[0], 2)  # W' in Figure 2.
print('Encoder weights \n', w_encoder)
print('Decoder weights \n', w_decoder)

# %% [markdown]
# ### 2. Weight Orthogonality
# Unlike PCA weights, the weights on Encoder and Decoder are not orthogonal.

# %%
w_pca = pca.components_
np.round(np.dot(w_pca, w_pca.T), 3)

# %%
np.round(np.dot(w_encoder, w_encoder.T), 3)

# %%
np.round(np.dot(w_decoder, w_decoder.T), 3)

# %% [markdown]
# ### 3. Uncorrelated Features
# Unlike PCA features, i.e. Principal Scores, the Encoded features are correlated.

# %%
pca_features = pca.fit_transform(X_train_scaled)
np.round(np.cov(pca_features.T), 5)

# %%
encoder_layer = Model(inputs=autoencoder.inputs, outputs=autoencoder.layers[0].output)
encoded_features = np.array(encoder_layer.predict(X_train_scaled))
print('Encoded feature covariance\n', np.cov(encoded_features.T))

# %% [markdown]
# ### 4. Unit Norm

# %%
print('PCA weights norm, \n', np.sum(w_pca ** 2, axis = 1))
print('Encoder weights norm, \n', np.sum(w_encoder ** 2, axis = 1))
print('Decoder weights norm, \n', np.sum(w_decoder ** 2, axis = 1))

# %% [markdown]
# ### Train Test Reconstruction Accuracy

# %%
train_predictions = autoencoder.predict(X_train_scaled)
print('Train reconstrunction error\n', sklearn.metrics.mean_squared_error(X_train_scaled, train_predictions))
test_predictions = autoencoder.predict(X_test_scaled)
print('Test reconstrunction error\n', sklearn.metrics.mean_squared_error(X_test_scaled, test_predictions))

# %% [markdown]
# --------

# %% [markdown]
# # Well-posed Autoencoder
# ### Constraints for Autoencoder
# Optimizing Autoencoder using PCA principles

# %%
nb_epoch = 100
batch_size = 16
input_dim = X_train_scaled.shape[1] #num of predictor variables,
encoding_dim = 2
learning_rate = 1e-3


# %% [markdown]
# ### 1. Constraint: Tied weights
#
# Make decoder weights equal to encoder.

# %%
class DenseTied(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tied_to=None,
                 **kwargs):
        self.tied_to = tied_to
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            self.kernel = K.transpose(self.tied_to.kernel)
            self._non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


# %% [markdown]
# #### 1.1 Bias=False for Decoder

# %%
encoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias = True)
decoder = DenseTied(input_dim, activation="linear", tied_to=encoder, use_bias = False)

# %%
autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer='sgd')
autoencoder.summary()

autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=3,
                batch_size=batch_size,
                shuffle=True,
                verbose=0)

# %%
#w_encoder = np.round(np.transpose(autoencoder.layers[0].get_weights()[0]), 3)
#w_decoder = np.round(autoencoder.layers[1].get_weights()[0], 3)
#get_weights() not working for class DenseTied because DenseTied.weights contains a tensor element
#which
w_encoder = np.round(np.transpose(autoencoder.layers[0].weights[0].numpy()), 3)
w_decoder = np.round(np.transpose(autoencoder.layers[1].weights[0].numpy()), 3)
print('Encoder weights\n', w_encoder)
print('Decoder weights\n', w_decoder)

# %%
train_predictions = autoencoder.predict(X_train_scaled)
print('Train reconstrunction error\n', sklearn.metrics.mean_squared_error(X_train_scaled, train_predictions))
test_predictions = autoencoder.predict(X_test_scaled)
print('Test reconstrunction error\n', sklearn.metrics.mean_squared_error(X_test_scaled, test_predictions))

# %% [markdown]
# #### 1.2 Bias=True for Decoder

# %%
encoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias = True)
decoder = DenseTied(input_dim, activation="linear", tied_to=encoder, use_bias = True)

autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer='sgd')
autoencoder.summary()

autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                verbose=0)

# %%
#w_encoder = np.round(np.transpose(autoencoder.layers[0].get_weights()[0]), 3)
#w_decoder = np.round(autoencoder.layers[1].get_weights()[1], 3)
w_encoder = np.round(autoencoder.layers[0].weights[0].numpy().transpose(), 3)
w_decoder = np.round(autoencoder.layers[1].weights[1].numpy().transpose(), 3)

print('Encoder weights\n', w_encoder)
print('Decoder weights\n', w_decoder)
print('PCA weights\n', w_pca)


# %%
train_predictions = autoencoder.predict(X_train_scaled)
print('Train reconstrunction error\n', sklearn.metrics.mean_squared_error(X_train_scaled, train_predictions))
test_predictions = autoencoder.predict(X_test_scaled)
print('Test reconstrunction error\n', sklearn.metrics.mean_squared_error(X_test_scaled, test_predictions))


# %% [markdown]
# ### 2. Constraint: Weights orthogonality.

# %%
class WeightsOrthogonalityConstraint (Constraint):
    def __init__(self, encoding_dim, weightage = 1.0, axis = 0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis

    def weights_orthogonality(self, w):
        if(self.axis==1):
            w = K.transpose(w)
        if(self.encoding_dim > 1):
            m = K.dot(K.transpose(w), w) - K.eye(self.encoding_dim)
            return self.weightage * K.sqrt(K.sum(K.square(m)))
        else:
            m = K.sum(w ** 2) - 1.
            return m

    def __call__(self, w):
        return self.weights_orthogonality(w)


# %% [markdown]
# #### 2.1 Encoder weight orthogonality

# %%
encoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias=True, kernel_regularizer=WeightsOrthogonalityConstraint(encoding_dim, weightage=1., axis=0))
decoder = Dense(input_dim, activation="linear", use_bias = True)

autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer='sgd')
autoencoder.summary()

autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                verbose=0)

# %%
w_encoder = autoencoder.layers[0].get_weights()[0]
print('Encoder weights dot product\n', np.round(np.dot(w_encoder.T, w_encoder), 2))

# %%
train_predictions = autoencoder.predict(X_train_scaled)
print('Train reconstrunction error\n', sklearn.metrics.mean_squared_error(X_train_scaled, train_predictions))
test_predictions = autoencoder.predict(X_test_scaled)
print('Test reconstrunction error\n', sklearn.metrics.mean_squared_error(X_test_scaled, test_predictions))

# %% [markdown]
# #### 2.2 Encoder and Decoder Weight orthogonality

# %%
encoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias=True, kernel_regularizer=WeightsOrthogonalityConstraint(encoding_dim, weightage=1., axis=0))
decoder = Dense(input_dim, activation="linear", use_bias = True, kernel_regularizer=WeightsOrthogonalityConstraint(encoding_dim, weightage=1., axis=1))

autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer='sgd')
autoencoder.summary()

autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                verbose=0)

# %%
w_encoder = autoencoder.layers[0].get_weights()[0]
print('Encoder weights dot product\n', np.round(np.dot(w_encoder.T, w_encoder), 2))

w_decoder = autoencoder.layers[1].get_weights()[0]
print('Decoder weights dot product\n', np.round(np.dot(w_decoder, w_decoder.T), 2))


# %% [markdown]
# ### 3. Constraint: Uncorrelated Encoded features

# %%
class UncorrelatedFeaturesConstraint (Constraint):

    def __init__(self, encoding_dim, weightage=1.0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage

    def get_covariance(self, x):
        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))

        x_centered = tf.stack(x_centered_list)
        covariance = K.dot(x_centered, K.transpose(x_centered)) / \
            tf.cast(x_centered.get_shape()[0], tf.float32)

        return covariance

    # Constraint penalty
    def uncorrelated_feature(self, x):
        if(self.encoding_dim <= 1):
            return 0.0
        else:
            output = K.sum(K.square(
                self.covariance - tf.math.multiply(self.covariance, K.eye(self.encoding_dim))))
            return output

    def __call__(self, x):
        self.covariance = self.get_covariance(x)
        return self.weightage * self.uncorrelated_feature(x)


# %%
encoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias=True,
                activity_regularizer=UncorrelatedFeaturesConstraint(encoding_dim, weightage=1.))
decoder = Dense(input_dim, activation="linear", use_bias=True)

autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer='sgd')
autoencoder.summary()

autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                verbose=0)

# %%
encoder_layer = Model(inputs=autoencoder.inputs, outputs=autoencoder.layers[0].output)
encoded_features = np.array(encoder_layer.predict(X_train_scaled))
print('Encoded feature covariance\n', np.round(np.cov(encoded_features.T), 3))

# %%
train_predictions = autoencoder.predict(X_train_scaled)
print('Train reconstrunction error\n', sklearn.metrics.mean_squared_error(X_train_scaled, train_predictions))
test_predictions = autoencoder.predict(X_test_scaled)
print('Test reconstrunction error\n', sklearn.metrics.mean_squared_error(X_test_scaled, test_predictions))

# %% [markdown]
# ### 4. Constraint: Unit Norm

# %% [markdown]
# #### 4.1 Unit Norm constraint on Encoding Layer

# %%
encoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias = True, kernel_constraint=UnitNorm(axis=0))
decoder = Dense(input_dim, activation="linear", use_bias = True)

autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer='sgd')
autoencoder.summary()

autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                verbose=0)

# %%
w_encoder = np.round(autoencoder.layers[0].get_weights()[0], 2).T  # W in Figure 2.
print('Encoder weights norm, \n', np.round(np.sum(w_encoder ** 2, axis = 1),3))

# %%
train_predictions = autoencoder.predict(X_train_scaled)
print('Train reconstrunction error\n', sklearn.metrics.mean_squared_error(X_train_scaled, train_predictions))
test_predictions = autoencoder.predict(X_test_scaled)
print('Test reconstrunction error\n', sklearn.metrics.mean_squared_error(X_test_scaled, test_predictions))

# %% [markdown]
# #### 4.2 Unit Norm constraint on both Encoding and Decoding Layer

# %%
encoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias = True, kernel_constraint=UnitNorm(axis=0))
decoder = Dense(input_dim, activation="linear", use_bias = True, kernel_constraint=UnitNorm(axis=1))

autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer='sgd')
autoencoder.summary()

autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                verbose=0)

# %%
w_encoder = np.round(autoencoder.layers[0].get_weights()[0], 2).T  # W in Figure 2.
w_decoder = np.round(autoencoder.layers[1].get_weights()[0], 2)  # W' in Figure 2.

print('Encoder weights norm, \n', np.round(np.sum(w_encoder ** 2, axis = 1),3))
print('Decoder weights norm, \n', np.round(np.sum(w_decoder ** 2, axis = 1),3))

# %%
train_predictions = autoencoder.predict(X_train_scaled)
print('Train reconstrunction error\n', sklearn.metrics.mean_squared_error(X_train_scaled, train_predictions))
test_predictions = autoencoder.predict(X_test_scaled)
print('Test reconstrunction error\n', sklearn.metrics.mean_squared_error(X_test_scaled, test_predictions))

# %% [markdown]
# ----------

# %% [markdown]
# ## Constraints put together

# %%
encoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias = True, kernel_regularizer=WeightsOrthogonalityConstraint(encoding_dim, weightage=1., axis=0), kernel_constraint=UnitNorm(axis=0))
decoder = DenseTied(input_dim, activation="linear", tied_to=encoder, use_bias = False)

autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer='sgd')
autoencoder.summary()

autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                verbose=0)

# %%
train_predictions = autoencoder.predict(X_train_scaled)
print('Train reconstrunction error\n', sklearn.metrics.mean_squared_error(X_train_scaled, train_predictions))
test_predictions = autoencoder.predict(X_test_scaled)
print('Test reconstrunction error\n', sklearn.metrics.mean_squared_error(X_test_scaled, test_predictions))

# %% [markdown]
# -------------

# %%
