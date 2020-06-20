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
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Layer, InputSpec
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers, activations, initializers, constraints, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import UnitNorm, Constraint

# %%
# Utility function
def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

# Generate Data
n_dim = 5 # 5 dimensional data
cov = sklearn.datasets.make_spd_matrix(n_dim, random_state=1234) # Generate
corr = correlation_from_covariance(cov)
mu = np.random.normal(0, 0.1, n_dim) # Generate mean
n = 1000 # Number of samples
X = np.random.multivariate_normal(mu, cov, n)
#X = X.astype('float32')

# Preprocess data
X_train, X_test = train_test_split(X, test_size=0.01, random_state=123)
scaler = StandardScaler() #Translate data and scale them such that stddev is 1
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Original COV
values, vectors = scipy.linalg.eig(corr) # Vector are columns
idx = np.argsort(values*-1)
values, vectors = values[idx], vectors[:,idx]
ratio = values/values.sum() # Explained Variance Ratio

# Derived COV
pca_analysis = sklearn.decomposition.PCA()
pca_analysis.fit(X_train_scaled)
cov2 = pca_analysis.get_covariance()
vectors2 = pca_analysis.components_ # Vector are rows
values2 = pca_analysis.explained_variance_
ratio2 = pca_analysis.explained_variance_ratio_
cov2, corr
# Compare eigenvectors and eigenvalues
values, values2
vectors.T, vectors2
cross_prod = []
for i in range(n_dim):
    cross_prod.append(np.dot(vectors.T[i], vectors2[i])/(scipy.linalg.norm(vectors.T[i])*scipy.linalg.norm(vectors2[i])))

print(cross_prod)

# %%
nb_epoch = 100 # Number of runs
batch_size = 16
input_dim = X_train_scaled.shape[1] # Number of predictor variables,
encoding_dim = 2
learning_rate = 1e-2

# Batch and shuffle data
train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train_scaled, X_train_scaled)).shuffle(1000).batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((X_test_scaled, X_test_scaled)).batch(batch_size)

# %%
from tensorflow.python.ops import math_ops, sparse_ops, gen_math_ops, nn
from tensorflow.python.framework import tensor_shape

class DenseWeight(tf.keras.layers.Dense):
    # source code for tf.keras.layers.Dense -> https://bit.ly/3ciTIEJ
    def __init__(self, units, w_axis, **kwargs):
        # w_axis (0 or 1) determines which axis the weight vectors are on.
        # For encoder -> w_axis = 0, decoder -> w_axis = 1
        super(DenseWeight, self).__init__(units, **kwargs)
        self.w_axis = w_axis

    def build(self, input_shape):
        super(DenseWeight,self).build(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.kernelshape = [last_dim, self.units]
        self.weightshape = [1,1]
        self.weightshape[self.w_axis] = self.kernelshape[self.w_axis]
        self.kernelall = []
        # Setting separate weight vectors.
        for i in range(self.kernelshape[int(not self.w_axis)]):
            custominit = self.kernel_initializer
            self.kernelall.append(self.add_weight(
            name='w{:d}'.format(i), shape=self.weightshape,
            initializer=custominit, regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint, dtype=self.dtype,
            trainable=True))

    def call(self, inputs, weight_ind=None):
        # Set default
        if weight_ind is None:
            weight_ind = len(self.kernelall)-1
        # Create kernel where the weights above weight_ind are zero so that they don't affect the training.
        self.kerneltrain = [w if w_ind <= weight_ind else tf.zeros(w.shape) for w_ind, w in enumerate(self.kernelall)]
        self.kernel = tf.concat(self.kerneltrain,int(not self.w_axis))
        # Copied from source
        return super(DenseWeight,self).call(inputs)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the layers
        self.d1 = DenseWeight(encoding_dim, w_axis=0, activation="linear",
        input_shape=(input_dim,), use_bias=False, dtype='float32',
        kernel_constraint=tf.keras.constraints.UnitNorm(axis=0))
        #self.d2 = Dense(input_dim, activation="linear", use_bias = True)
        self.d2 = DenseWeight(input_dim, w_axis=1, activation="linear",
        use_bias=False, dtype='float32', kernel_constraint=tf.keras.constraints.UnitNorm(axis=1))
        #self.d2 = MyDenseLayer(input_dim)

    def call(self, x, weight_ind=None):
        # Connecting the layers
        x = self.d1(x, weight_ind)
        x = self.d2(x, weight_ind)
        return x

# Create an instance of the model
tf.keras.backend.clear_session()
model = MyModel()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

# Define training step
@tf.function
def train_step(input, output, weight_ind, var_list):
      with tf.GradientTape() as tape:
        predictions = model(input, weight_ind)
        loss = loss_object(output, predictions)
      # Backpropagation
      gradients = tape.gradient(loss, var_list)
      optimizer.apply_gradients(zip(gradients, var_list))
      # Log metric
      train_loss.update_state(loss)

@tf.function
def test_step(input, output):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(input)
    t_loss = loss_object(output, predictions)
    # Log metric
    test_loss(t_loss)
    #test_accuracy(output, predictions)

a = np.zeros([5,1])
b = np.zeros([1,5])
np.concatenate([a,a],1)
np.concatenate([b,b],0)

nb_epoch = 51 #number of epochs per weight vector
weight_ind = 0
while weight_ind <= encoding_dim-1:
    for epoch in range(nb_epoch):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()

        for num, (input, output) in enumerate(train_ds):
            var_list = []
            untrainable_weights_d1 = [i for ind, i in enumerate(model.d1.weights) if ind != weight_ind]
            untrainable_weights_d2 = [i for ind, i in enumerate(model.d2.weights) if ind != weight_ind]
            untrainable_weights = untrainable_weights_d1 + untrainable_weights_d2
            for v in model.trainable_variables:
                append = True
                for w in untrainable_weights:
                    if v is w:
                        append = False
                        break
                if append:
                    var_list.append(v)
            train_step(input, output, weight_ind, var_list)

        for input, output in test_ds:
            test_step(input, output)

        template = 'Epoch {}, Loss: {}, Test Loss: {}'
        if epoch % 10 == 0:
            print(template.format(epoch,
                            train_loss.result(),
                            test_loss.result()))
            #print(np.concatenate([i.numpy() for i in model.d2.weights]))

    weight_ind+=1

# %%
enc_w = np.concatenate([i.numpy() for i in model.d1.weights],axis=1)
dec_w = np.concatenate([i.numpy() for i in model.d2.weights],axis=0)
np.dot(enc_w.T,enc_w)
np.dot(dec_w,enc_w)

np.dot(dec_w,np.linalg.pinv(dec_w))
np.dot(np.linalg.pinv(enc_w),enc_w)
