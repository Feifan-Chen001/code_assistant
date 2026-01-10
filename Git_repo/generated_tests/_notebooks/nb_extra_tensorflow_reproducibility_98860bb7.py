# --- notebook cell 4 ---
from IPython.display import IFrame
IFrame(src="https://www.youtube.com/embed/Ys8ofBeR2kA", width=560, height=315, frameborder="0", allowfullscreen=True)

# --- notebook cell 6 ---
from __future__ import division, print_function, unicode_literals

try:
    # %tensorflow_version only exists in Colab.
except Exception:
    pass

import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- notebook cell 11 ---
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

# --- notebook cell 14 ---
2. * 5. / 7.

# --- notebook cell 15 ---
2. / 7. * 5.

# --- notebook cell 17 ---
config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1)

with tf.Session(config=config) as sess:
    #... this will run single threaded
    pass

# --- notebook cell 19 ---
with tf.Session() as sess:
    #... also single-threaded!
    pass

# --- notebook cell 22 ---
print(set("Try restarting the kernel and running this again"))
print(set("Try restarting the kernel and running this again"))

# --- notebook cell 27 ---
if os.environ.get("PYTHONHASHSEED") != "0":
    raise Exception("You must set PYTHONHASHSEED=0 when starting the Jupyter server to get reproducible results.")

# --- notebook cell 29 ---
import random

random.seed(42)
print(random.random())
print(random.random())

print()

random.seed(42)
print(random.random())
print(random.random())

# --- notebook cell 31 ---
import numpy as np

np.random.seed(42)
print(np.random.rand())
print(np.random.rand())

print()

np.random.seed(42)
print(np.random.rand())
print(np.random.rand())

# --- notebook cell 34 ---
import tensorflow as tf

tf.set_random_seed(42)
rnd = tf.random_uniform(shape=[])

with tf.Session() as sess:
    print(rnd.eval())
    print(rnd.eval())

print()

with tf.Session() as sess:
    print(rnd.eval())
    print(rnd.eval())

# --- notebook cell 36 ---
tf.reset_default_graph()

tf.set_random_seed(42)
rnd = tf.random_uniform(shape=[])

with tf.Session() as sess:
    print(rnd.eval())
    print(rnd.eval())

print()

with tf.Session() as sess:
    print(rnd.eval())
    print(rnd.eval())

# --- notebook cell 38 ---
tf.reset_default_graph()
tf.set_random_seed(42)

graph = tf.Graph()
with graph.as_default():
    rnd = tf.random_uniform(shape=[])

with tf.Session(graph=graph):
    print(rnd.eval())
    print(rnd.eval())

print()

with tf.Session(graph=graph):
    print(rnd.eval())
    print(rnd.eval())

# --- notebook cell 40 ---
graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(42)
    rnd = tf.random_uniform(shape=[])

with tf.Session(graph=graph):
    print(rnd.eval())
    print(rnd.eval())

print()

with tf.Session(graph=graph):
    print(rnd.eval())
    print(rnd.eval())

# --- notebook cell 42 ---
tf.reset_default_graph()

rnd = tf.random_uniform(shape=[])

tf.set_random_seed(42) # BAD, NO EFFECT!
with tf.Session() as sess:
    print(rnd.eval())
    print(rnd.eval())

print()

tf.set_random_seed(42) # BAD, NO EFFECT!
with tf.Session() as sess:
    print(rnd.eval())
    print(rnd.eval())

# --- notebook cell 45 ---
tf.reset_default_graph()

rnd1 = tf.random_uniform(shape=[], seed=42)
rnd2 = tf.random_uniform(shape=[], seed=42)
rnd3 = tf.random_uniform(shape=[])

with tf.Session() as sess:
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())

print()

with tf.Session() as sess:
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())

# --- notebook cell 47 ---
tf.reset_default_graph()

tf.set_random_seed(42)

rnd1 = tf.random_uniform(shape=[], seed=42)
rnd2 = tf.random_uniform(shape=[], seed=42)
rnd3 = tf.random_uniform(shape=[])

with tf.Session() as sess:
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())

print()

with tf.Session() as sess:
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())

# --- notebook cell 50 ---
random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

# --- notebook cell 52 ---
my_config = tf.estimator.RunConfig(tf_random_seed=42)

feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                     feature_columns=feature_cols,
                                     config=my_config)

# --- notebook cell 54 ---
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)

# --- notebook cell 56 ---
indices = np.random.permutation(len(X_train))
X_train_shuffled = X_train[indices]
y_train_shuffled = y_train[indices]

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train_shuffled}, y=y_train_shuffled, num_epochs=10, batch_size=32, shuffle=False)
dnn_clf.train(input_fn=input_fn)

# --- notebook cell 59 ---
def create_dataset(X, y=None, n_epochs=1, batch_size=32,
                   buffer_size=1000, seed=None):
    dataset = tf.data.Dataset.from_tensor_slices(({"X": X}, y))
    dataset = dataset.repeat(n_epochs)
    dataset = dataset.shuffle(buffer_size, seed=seed)
    return dataset.batch(batch_size)

input_fn=lambda: create_dataset(X_train, y_train, seed=42)

# --- notebook cell 60 ---
random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

my_config = tf.estimator.RunConfig(tf_random_seed=42)

feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                     feature_columns=feature_cols,
                                     config=my_config)
dnn_clf.train(input_fn=input_fn)

# --- notebook cell 65 ---
keras.backend.clear_session()

random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10)

# --- notebook cell 69 ---
for i in range(10):
    with open("my_test_foo_{}".format(i), "w"):
        pass

[f for f in os.listdir() if f.startswith("my_test_foo_")]

# --- notebook cell 70 ---
for i in range(10):
    with open("my_test_bar_{}".format(i), "w"):
        pass

[f for f in os.listdir() if f.startswith("my_test_bar_")]

# --- notebook cell 72 ---
filenames = os.listdir()
filenames.sort()

# --- notebook cell 73 ---
[f for f in filenames if f.startswith("my_test_foo_")]

# --- notebook cell 74 ---
for f in os.listdir():
    if f.startswith("my_test_foo_") or f.startswith("my_test_bar_"):
        os.remove(f)