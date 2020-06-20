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

# Dense Tied Weights
class DenseTranspose(tf.keras.layers.Layer):
    def __init__(self, units, dense, activation=None, use_bias=True,
                bias_initializer='zeros', **kwargs):
        self.units = units # number of nodes in layer
        self.dense = dense
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.biases = self.add_weight(name = "bias",
                        initializer = self.bias_initializer,
                        shape = [self.units,])
        #super().build(input_shape)

    def call(self, inputs):
        #z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        z = tf.matmul(inputs, self.dense.kernel, transpose_b=True)
        if self.use_bias:
            return self.activation(z + self.biases)
        else:
            return self.activation(z)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the layers
        self.d1 = Dense(encoding_dim, activation="linear",
        input_shape=(input_dim,), use_bias=False, dtype='float32')
        #self.d2 = Dense(input_dim, activation="linear", use_bias = True)
        self.d2 = DenseTranspose(input_dim, self.d1, use_bias=False, activation="linear")
        #self.d2 = MyDenseLayer(input_dim)

    def call(self, x, d1_weight_ind=None):
        # Connecting the layers
        x = self.d1(x)
        x = self.d2(x)
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
def train_step(input, output):
      with tf.GradientTape() as tape:
            predictions = model(input)
            loss = loss_object(output, predictions)
      # Backpropagation
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
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

for epoch in range(nb_epoch):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    test_loss.reset_states()

    for num, (input, output) in enumerate(train_ds):
        train_step(input, output)

    for input, output in test_ds:
        test_step(input, output)

    template = 'Epoch {}, Loss: {}, Test Loss: {}'
    if (epoch) % 10 ==0:
        print(template.format(epoch,
                        train_loss.result(),
                        test_loss.result()))

enc_w = np.concatenate([i.numpy() for i in model.d1.weights],axis=1)
np.dot(enc_w.T,enc_w)
