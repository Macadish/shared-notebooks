# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] colab_type="text" id="rX8mhOLljYeM"
# ##### Copyright 2019 The TensorFlow Authors.

# %% [markdown] colab_type="text" id="3wF5wszaj97Y"
# # [TensorFlow 2 quickstart for experts](https://www.tensorflow.org/tutorials/quickstart/advanced)
#
# This guide demonstrates how to set up a simple NN model and to visualize the training steps in tensorboard.
#
# Import TensorFlow into your program:

# %% colab={} colab_type="code" id="0trJmd6DjqBZ"
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# Enable Tensorboard
%load_ext tensorboard
import datetime

# Remove previous log
#!rm -rf ./logs/ 

# %% [markdown] colab_type="text" id="7NAbSZiaoJ4z"
# Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

# %% colab={} colab_type="code" id="JqFRS6K07jJs"
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape)

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
print(x_train.shape)

# %% [markdown] colab_type="text" id="k1Evqx0S22r_"
# Use `tf.data` to batch and shuffle the dataset:

# %% colab={} colab_type="code" id="8Iu_quO024c2"
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
print(type(train_ds))

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# %% [markdown] colab_type="text" id="BPZ68wASog_I"
# Use this [link](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215) to understand how convolution NN works.
# Build the `tf.keras` model using the Keras [model subclassing API](https://www.tensorflow.org/guide/keras#model_subclassing):

# %% colab={} colab_type="code" id="h3IKyzTCDNGo"
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    # Define the layers
    self.conv1 = Conv2D(32, 3, activation='relu') #
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    # Connecting the layers
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()

# %% [markdown] colab_type="text" id="uGih-c2LgbJu"
# Choose an optimizer and loss function for training: 

# %% colab={} colab_type="code" id="u48C9WQ774n4"
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# %% [markdown] colab_type="text" id="JB6A1vcigsIe"
# Select metrics to measure the loss and the accuracy of the model. These metrics accumulate the values over epochs and then print the overall result. **Loss metric** used by the user to evaluate how well the model is doing whereas **loss function** is used by the model to learn. 

# %% colab={} colab_type="code" id="N0MqHFb4F_qn"
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# %% [markdown]
# Set up summary writers to write the summaries to disk in a different logs directory:

# %%
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "/home/jovyan/shared-notebooks/log/tf_advanced_tutorial"

# %%
# Remove previous log
!rm -rf {logdir}
!mkdir {logdir}

# %% [markdown]
# Setup log files for tensorboard

# %%
train_log_dir = logdir + '/gradient_tape/' + current_time + '/train'
test_log_dir = logdir + '/gradient_tape/' + current_time + '/test'
graph_log_dir = logdir + '/gradient_tape/' + current_time + '/graph'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
graph_summary_writer = tf.summary.create_file_writer(graph_log_dir)


# %% [markdown] colab_type="text" id="ix4mEL65on-w"
# Use `tf.GradientTape` to train the model:

# %% colab={} colab_type="code" id="OZACiVqA8KQV"
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  # Backpropagation
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


# %% [markdown] colab_type="text" id="Z8YT7UmFgpjV"
# Test the model:

# %% colab={} colab_type="code" id="xIKdEzHAJGt7"
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


# %% [markdown]
# Run Training and Testing

# %% colab={} colab_type="code" id="i-2pkctU_Ci7"
EPOCHS = 7

for epoch in range(EPOCHS):
# Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()
  tf.summary.trace_on(graph=True, profiler=True) # Trace Training
    
  for num, (images, labels) in enumerate(train_ds):
    train_step(images, labels)

  with graph_summary_writer.as_default():
    tf.summary.trace_export(
    name="my_func_trace",
    step=0,
    profiler_outdir=logdir)

  with train_summary_writer.as_default():
    tf.summary.scalar('loss', train_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)
    
  with test_summary_writer.as_default():
    tf.summary.scalar('loss', test_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
  
  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))

  

# %% [markdown] colab_type="text" id="T4JfEh7kvx6m"
# The image classifier is now trained to ~98% accuracy on this dataset. To learn more, read the [TensorFlow tutorials](https://www.tensorflow.org/tutorials).

# %% [markdown]
# Set logdir to the directory where the log files are generated
# View [tensorboard](https://tb.jonai.teojy.com/).
# Be sure to stop the process to open up port 6006 for other notebooks.

# %%
%tensorboard --logdir $logdir --port 6006 --host 0.0.0.0 #Go to https://tb.jonai.teojy.com/

# %%
#!ls {logdir}

# %%
