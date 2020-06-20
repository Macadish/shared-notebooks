# %% [markdown]
# # PCA using autoencoder
#
# When analyzing large datasets, it is important to preprocess the data to prevent potential overfitting (curse of dimensionality). Dimension reduction is one such technique that identifies a small set of features to represent a large dataset. Features are chosen based on how well they capture underlying structure in the data based on certain criteria. For example, in principle component analysis (PCA), features are selected from principle components that best explain the variance in the data i.e., if a dataset varies wildly in a certain direction, the corresponding vector is selected as a feature.
#
# Another way to encode data is to use an autoencoder (AE). The autoencoder first encodes data into a latent space (often of a smaller dimension than the input data) using one or more layers, and decodes points in the latent space back to the original data. The goal is learn features that reduce the difference between the original and reconstructed data.
#
# Autoencoders are often compared to PCA. In fact, an autoencoder that is a single layer deep (1 for encoding, 1 for decoding) and uses linear activation is often thought to function like PCA in the roughest of sense. However, there are several differences between both methods.
# 1. PCA, and covariance, is primarily a linear operation. It cannot learn non-linear features the same way autoencoders can using non-linear kernels. That said, kernel PCA is one way to extend linear PCA to non-linear data.
# 1. PCA eigenvectors are orthorgonal by default. AE weights are not orthorgonal necessarily but instead overlap, sometimes significantly. ~~This non-orthorgonality could potentially contribute to the non-linearity since it scales the importance of certain vectors in a non-linear fashion.~~
# 1. PCA transforms and reconstructs the data using the same transformation (eigenvector) matrix. The encoder and decoder layer on the other hand have different weights.
#
# I will show that it is possible to configure an autoencoder to produce the same results as PCA and not just approximate it!
#
# Note: Performing PCA using autoencoders makes little sense because the code is complicated and computationally intensive compared to PCA. However, if we can recreate PCA using an autoencoder, it means we can treat autoencoders as less of a 'black box' and more of a PCA analog which we can further improve on.

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

# %% [markdown]
# # Data
# 1. Generate data from cov matrix and mu
# 2. Preprocess data. Check out [link](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py)

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

# %% [markdown]
# # PCA
# We first perform a standard PCA. As a recap, PCA is simply the eigendecomposition the covariance matrix. To get a graphical intuition of what PCA does, we need to understand what the covariance matrix. Graphically, for 2D data, the covariance matrix describes the shape of the data on a plane. It is basically a transformation matrix that shapes data distributed in a circle (cov(x,y)=0, var(x)=1, var(y)=1) into a skewed ellipse.
#
# When we perform eigendecomposition, we are essentially breaking down the transformation into separate scaling and skewing/rotating operations. The eigenvector matrix unskews the data, and a diagonal eigenvalue matrix scales the data. Note that the eigenvalues are the variance of the data along the eigenvectors i.e. the lengths of the major and minor axis of the ellipse.
#
# We will perform PCA using
# 1. Original COV
# 2. Derived COV
#
# To check if the vectors are in the same direction, we perform a dot product of the unit vectors. If the value is close to 1 i.e the eigenvectors from both COV matrix have very similar directions.

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

# %% [markdown]
# # Autoencoder
#
# I'll first demonstrate how to create an autoencoder in tensorflow using the basic layers. Since I'll need to customize the training loop later, it is easier to create a tensorflow model using subclass instead of sequential or functional API.
#
# ## Set up model and layers
#
# %%
nb_epoch = 100 #Number of runs
batch_size = 16
input_dim = X_train_scaled.shape[1] #Number of predictor variables,
encoding_dim = 2
learning_rate = 1e-2

# Batch and shuffle data
train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train_scaled, X_train_scaled)).shuffle(1000).batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((X_test_scaled, X_test_scaled)).batch(batch_size)

# Define model
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the layers
        self.d1 = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias=True, dtype='float32')
        self.d2 = Dense(input_dim, activation="linear", use_bias = True)

    def call(self, x):
        # Connecting the layers
        x = self.d1(x)
        return self.d2(x)

# Create an instance of the model
tf.keras.backend.clear_session()
model = MyModel()

# %% [markdown]
# ## Select Loss function, Optimzer and Metric
#
# We define the **loss function** using mean squared error to evaluate the difference between the original and reconstructed data. The model **optimize** its weights using stochastic gradient descent, i.e. training the model using a gradients from a subset of input data. The **loss metric** is the mean of losses (MSE) for all data.
#
# Note: For sequential models, when choosing a loss metric in the `model.compile()` step, you may see 'accuracy' chosen in some examples. They don't make sense in a autoencoder since it is meant for categorial data, not continuous data.
# ```python
# autoencoder.compile(metrics=['accuracy'],
#                     loss='mean_squared_error',
#                     optimizer='sgd')
# ```
# * Tensorflow Issue #34451 - if you pass the string 'accuracy', we infer between binary_accuracy/categorical_accuracy functions
# * See more discussion [here](https://bit.ly/3dsyMwm)


# %%
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

# %% [markdown]
# ## Set up training steps and train network

# %%
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

# %%
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

# %% [markdown]
# # PCA-like Autoencoder
# The goal of creating a PCA-like autoencoder is to provide context to the underlying weights of the neural network. In a neural network, for linear activation, the first layer is essentially a projection of the data onto a space determined by the weights of the layer. The cartoon below illustrates how a neural network projects data onto a vector space determined by the weights. To avoid confusion, all subsequent mentions of weights refer to the weight vectors as shown in the figure.
#
# ![](figures/DL-01.png)
#
# We will now introduce the following features to the autoencoder to replicate PCA:
# 1. encoder and decoder share the same weights but tranposed.*
#     1. tied weight do not extend to biases
# 1. linear activations.
# 1. orthogonal weights.
# 1. unit length weights
# 1. encoder dimension increased and trained iteratively. Weights of previously trained dimensions are kept constant. This replicates the process of finding the eigen vector that best explains the data iteratively.
# 1. loss function that minimizes variance of data in the latent space since we want the weights to function like eigenvectors.**
#
# \*Note that the decoder is not strictly the inverse of the encoder. However, in the case of a PCA, the transpose of the unit eigenvector matrix is also the inverse, so if we transpose the weights of the encoder, we get a decoder that is effectively performing an inverse operation. For encoders with smaller dimension, the inverse operation is technically a pseudoinverse.
#
# \*\* Turns out, we don't need to create such a loss function explicitly since the autoencoder will try to minimize variance naturally for the MSE loss function. See discussion on AE at the end.
#
# As we will show later, **choosing either tied-weights, orthogonality weights or sequential training is sufficient for generating weights that are orthogonal**, though only sequential training ensures that the weights are the same as PCA's eigenvectors.
#
# ## Adding orthogonality regularizer
# To ensure that the weights are orthogonal, we add a regularizer that penalizes weights that are not orthogonal. A quick way to define one is to recognize that if A is an orthonormal matrix, transpose(A) = inverse(A). If we multiply the orthonormal weight matrix by their transpose, we should get an identity matrix. Any deviation means the weights are not orthonormal, and we penalize it by adding the difference to the loss function.
#
# Note that we are not making this a constraint since we want the autoencoder to be able to choose slightly non-orthogonal weights if it results in a better fit, and an orthogonality regularizer is easier to implement.
#
# See [link](https://bit.ly/36PPG5W) for more details on the math.
#

# %%
class Orthogonal(tf.keras.regularizers.Regularizer):
    def __init__(self, encoding_dim, reg_weight = 1.0, axis = 0):
        self.encoding_dim = encoding_dim
        self.reg_weight = reg_weight
        self.axis = axis

    def __call__(self, w):
        if(self.axis==1):
            w = K.transpose(w)
        if(self.encoding_dim > 1):
            m = K.dot(K.transpose(w), w) - K.eye(self.encoding_dim)
            return self.reg_weight * K.sqrt(K.sum(K.square(K.abs(m))))
        else:
            m = K.sum(w ** 2) - 1.
            return m


# %% [markdown]
# ## Sequential training
#
# A unique feature of the PCA is that each eigenvector accounts for some variance in the data in increasing order. However, in an autoencoder, the loss function is defined only by the accuracy of the reconstructed data. If we set the latent space to have the same dimension as the incoming data, it is possible for the autoencoder to find a set of orthogonal weights that transforms the data to a different latent space and back perfectly, **but the weights are more a reflection of the initial condition than useful information**.
#
# If the dimension of the latent space is smaller than that of the input, the weights need to compress the data more efficiently (i.e. account for as much variance in the data as possible), which constraints the weights slightly. However, there are still many ways to 'distribute' the data variance across different weight vectors, so the values are again dependent on the initialization. Put another way, if the encoding layer is 2D, there is a plane that best describes the data variance. As I show later, the autoencoder is capable of finding the plane, but may represent the plane with two different vectors compared to PCA.
#
# If we reduce the encoding layer to a dimension of 1, the one resulting weight vector has to account for as much variance as possible. Naturally, it has to be close to the first principle component (due to the tied weights regularization) and the variance of the projected data must be close to the first eigenvalue.
#
# Once we have found the first weight vector, we can fix it while training the second weight vector, subject to the same constraints as before. The autoencoder should in theory find a second weight vector that accounts for the second most variance of the data. We repeat this for all subsequent weights until the desired encoding dimension is achieved.
#
# Below, we introduce a new layer that generates seperate weight vectors instead of a kernel matrix. By declaring the weight vectors separately, we can train them individually while keeping previously trained vectors fixed. Additionally, all untrained weights should be set to zero so that they don't affect the training. When a weight vector is ready to be trained, change it from zero to the default kernel_initializer so that the model can learn more efficiently. `weight_ind` is defined as the index of the weight that is currently being trained. When calling the layer, the layer uses `weight_ind` to generate the appropriate kernel for training.

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


# %% [markdown]
# ## Tying weights of encoder and decoder
# `DenseTranspose` creates a decoder layer that shares the same weight as the input encoder layer. More details are
# available in the [Link](https://bit.ly/2XTiKoT).
#
# Note that `shape=[self.dense.input_shape[-1]]` only work with a sequential or functional API model. In a subclass model, the layer can't infer the input_shape from the upstream layer until the model is called, but it needs that information before that.
#
# `TiedModel` creates an autoencoder the connects `DenseWeight` and `DenseTranspose`. When calling the model, add `d1_weight_ind` so that layer DenseWeight generates the correct kernel for training.
#

# %%
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

class TiedModel(Model):
    def __init__(self):
        super(TiedModel, self).__init__()
        # Define the layers
        self.d1 = DenseWeight(encoding_dim, w_axis=0,activation="linear",
        input_shape=(input_dim,), use_bias=False, dtype='float32',
        kernel_regularizer=Orthogonal(encoding_dim, reg_weight=1., axis=0),
        kernel_constraint=tf.keras.constraints.UnitNorm(axis=0))
        #self.d2 = Dense(input_dim, activation="linear", use_bias = True)
        self.d2 = DenseTranspose(input_dim, self.d1, use_bias=False, activation="linear")
        #self.d2 = MyDenseLayer(input_dim)

    def call(self, x, d1_weight_ind=None):
        # Connecting the layers
        x = self.d1(x, d1_weight_ind)
        x = self.d2(x)
        return x

# %% [markdown]
# ## Performing sequential training
#
# To specify which vector to train, we define `var_list`, which is used by the optimizer to determine which variable to train. Note that `var_list` should not be defined in `train_step` since it uses python features that are not compatible with `@tf.function`. See [link](https://bit.ly/3dxUT4U).

# %%
# Create an instance of the model
tf.keras.backend.clear_session()
model = TiedModel()

# Select Loss function, Optimizer and Metric
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

# Define training step
@tf.function
def train_step(input, output, d1_weight_ind, var_list):
      with tf.GradientTape() as tape:
        predictions = model(input, d1_weight_ind)
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


nb_epoch = 31 #number of epochs per weight vector
d1_weight_ind = 0
while d1_weight_ind <= encoding_dim-1:
    for epoch in range(nb_epoch):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()

        for num, (input, output) in enumerate(train_ds):
            var_list = []
            untrainable_weights = [i for ind, i in enumerate(model.d1.weights) if ind != d1_weight_ind]
            for v in model.trainable_variables:
                append = True
                for w in untrainable_weights:
                    if v is w:
                        append = False
                        break
                if append:
                    var_list.append(v)
            train_step(input, output, d1_weight_ind, var_list)

        for input, output in test_ds:
            test_step(input, output)

        template = 'Epoch {}, Loss: {}, Test Loss: {}'
        if epoch % 10 == 0:
            print(template.format(epoch,
                            train_loss.result(),
                            test_loss.result()))

    d1_weight_ind+=1

# %% [markdown]
# ## Comparing weights
# Let's compare weights/eigenvectors and projected data variance/eigenvalues.
#
# ### Orthorgonality
# The dot product of the eigenvector matrix `enc_weights` and its transpose is approximately the identity matrix i.e. the weights are orthogonal.
#
# ### Eigenvalues and Eigenvectors
# The 'eigenvalue' of the a weight vector can be obtained by projecting the data onto the vector and calculating the resulting variance. We shall call the variance the **projected variance**.
#
# The eigenvalues and eigenvectors for very close to the weights and projected variance.
#
# ### Eigenvector directions
# We can further test the direction of the eigenvector vs weights by taking the dot product of their unit vectors. The closer they are to 1, the smaller the angle between the vectors.

# %%
enc_weights = tf.concat(model.d1.weights,axis=1).numpy()
print('Check Orthogonality\n{}\n'.format(np.dot(enc_weights.T,enc_weights))) # Check for weight orthogonality
enc_values = np.dot(X_train_scaled, enc_weights).var(axis=0) # Get variance / eigenvalue
idx = np.argsort(enc_values*-1) # Sort the weights by data variance
values3, vectors3 = enc_values[idx], enc_weights[:,idx]

# Compare eigenvectors and eigenvalues
print('Original EigVal\n{}\nDerived EigVal\n{}\nAutoencoder DataVar\n{}\n'.format(values, values2, values3))
print('Original PC\n{}\nDerived PC\n{}\nAutoencoder Weights\n{}\n'.format(vectors.T, vectors2, vectors3.T))

# Compare direction of PCA eigenvectors and AE weights
def dp(a,b):
    #pseudo
    return np.dot(a, b)/(scipy.linalg.norm(a)*scipy.linalg.norm(b))

dot_prod = []
for i in range(vectors3.T.shape[0]):
    dot_prod.append(dp(vectors2[i], vectors3.T[i]))

print('Dot product of PCA eigenvectors and AE weights')
print(dot_prod)

# %% [markdown]
# ## Comparing loss between original autoencoder and PCA-like autoencoder
#
# Comparing the loss, it looks like the autoencoder performs similarly regardless of the PCA-inspired modifications. It suggests that the linear autoencoder is capable of finding the same latent space that best represents the data (i.e. the latent space that best reduces MSE) regardless of modifications. **The reason the weights are different is because the autoencoder choose to represent the same latent space using different weight vectors depending on the initialization.** By applying sequential training, the autoencoder selects meaningful vectors that are sorted by their power to reduce data variance. In fact, in `PCA_seqtrainonly.py`, sequential training alone of both encoder and decoder (i.e. training the first weight vector of encoder and decoder together followed by the second weight vector and so on) is sufficient for generating orthogonal weights, making the other modifications redundant!
#
# Another interesting feature is the redundancy of tied-weights and orthogonality regularization. Let's suppose for now that the unconstrained autoencoder returns orthogonal weights. If data dimension and encoding dimension are equal, the decoder simply takes the transpose (inverse) of the encoder weights to reconstruct the data perfectly. If encoding dimension is smaller, the transpose won't give us a perfect reconstruction, but one that minimizes MSE. The transpose of a non-square orthogonal matrix is also known as the pseudoinverse matrix. Conversely, if we tie the weights between encoder and decoder (via a transpose), the learned weights must be orthogonal since the loss is the same either way. Note that the redundancy only holds if the loss criteria is MSE. See `PCA_tiedonly.py` for more details.
#
# TL;DR Either one of these modifications, sequential training; tied-weights or orthogonality regularization, guarantees orthogonal weights when coupled with MSE loss. Only sequential training sorts the weights by 'eigenvalues' and matches the eigenvectors from PCA.
#
# ### Configuring the latent space
# Since the AE learns the same latent space except with different representation, we can learn the PCA representation by applying PCA to the latent space. The new eigenvectors will be in terms of the latent space representation, so to get back the depedence on the original input variables, apply a pseudo inverse.
#
# ### Pseudoinverse matrix and autoencoder
# Now let's suppose that the unconstrained autoencoder returns weights that are non-orthogonal. Since the autoencoder is representing the same optimal latent space with non-orthogonal weights, simply taking a transpose to decode the data will not work. Nevertheless, the decoder is able to find a set of weights that minimizes MSE. Turns out, the decoder weights match the **pseudoinverse matrix** calculated using SVD or `np.linalg.pinv`. The encoder projects data onto a latent space and transforms its coordinates, the decoder merely transforms the coordinates back but without 'unprojecting' the data, as shown in the cartoon below. An interesting property of pseudoinverse matrices is that multiplying a matrix by its pseudoinverse will yield an identity matrix (in the smaller dimension). See [link](https://bit.ly/2XNcYVC) for more details on pseudoinverse matrices.
#
# ![](figures/DL-02.png)
#
# ## Conclusion
# A linear autoencoder with MSE loss performs similarly to PCA. Given enough epochs, it'll find the same latent space as as the one found by PCA to minimize variance. The modifications proposed here do affect the latent space. Instead, they help the autoencoder look for vector representation of the latent space that are both interpretable and meaningful.

# %% [markdown]
# # Futher insights and future work
# 1. Given that MSE loss function is key to finding the right latent space that minimizes variance, **using a different the loss function** will produce a latent space that optimizes other characteristic in the data.
# 1. Biases matter if the data was not preprocessed. However, it complicates PCA <-> autoencoder analogy since biases don't have a simple inverse operation the same way weight matrices can be transposed. 
# 1. Non-linearity can be achieved by using non-linear activation, or by changing the number of layers?
# 1. The first layer projects the data onto an optimal latent space and determines the bulk of the compression. i.e. if the first encoding dimension is small, the subsequent layers won't be able to recover information that's lost.
# 1. The second/subsequent layers would look for trends in the latent space and alter it, such as prioritizing locality in the latent space for data sharing the same label. (See VAE)
# 1. Fourier compression essentially transforms signals to frequency components and removes high frequency ones. Analagously, autoencoder transforms data to a different latent space and projects them onto important vectors in a single step.
# 1. Comparing VAE and LDA, PCoA and tSNE
#
#
