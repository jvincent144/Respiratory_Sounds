import os
import numpy as np
import tensorflow as tf
from datasets import loadTFRD
from models import LSTM

# Enable premature calculations
#Eager execution provides an imperative interface to TensorFlow.
# With eager execution enabled, TensorFlow functions execute operations immediately
# (as opposed to adding to a graph to be executed later in a tf.compat.v1.Session)
# and return concrete values (as opposed to symbolic references to a node in a computational graph).
tf.enable_eager_execution()

# Dataset paths
train_path = "audio_tfrd/train.tfrecords"
val_path = "audio_tfrd/val.tfrecords"
test_path = "audio_tfrd/test.tfrecords"

# load datasets
batch_size = 8
train_ds = loadTFRD(train_path, True, batch_size)
val_ds = loadTFRD(val_path, True, batch_size)
test_ds = loadTFRD(test_path, True, batch_size)

# Count elements in datasets
train_ct = 0
for audio, lbl in  train_ds:
    train_ct += 1

val_ct = 0
for audio, lbl in  val_ds:
    val_ct += 1

test_ct = 0
for audio, lbl in  test_ds:
    test_ct += 1

train_ds.repeat(None)
val_ds.repeat(None)
test_ds.repeat(None)

model = LSTM()
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

cp_path = "training_2/cp-{epoch:04d}.h5py"
cp_dir = os.path.dirname(cp_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = cp_path,
                                                 verbose = 1,
                                                 save_weights_only = True,
                                                 period = 1)
log_dir = "./runs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

print("Initiating Training.")
# model.fit(train_ds, epochs = 10, validation_data = val_ds, steps_per_epoch = train_ct, validation_steps = val_ct, shuffle = True, callbacks = [cp_callback])
model.fit(train_ds, epochs = 10, validation_data = val_ds, steps_per_epoch = train_ct, validation_steps = val_ct, shuffle = True, callbacks = [cp_callback, tensorboard_callback])
model.save_weights("./training_1/model.h5py")
print("Finished Training.")

# export PATH=/usr/local/cuda-10.0/bin:/usr/local/cuda-10.0/NsightCompute-1.0${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
