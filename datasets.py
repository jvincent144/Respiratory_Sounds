import numpy as np
import tensorflow as tf
from tensorflow.data import TFRecordDataset

####
# Create tf data record files

# convert to one of the three tensorflow data types : tf.train.ByteList, tf.train.FloatList, tf.train.Int64List
# Source : https://www.tensorflow.org/tutorials/load_data/tfrecord

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create tf.Example
def to_tf_example(bytestring, lbl):
    
    feature = {"audio_raw" : _bytes_feature(bytestring),
                "label" : _int64_feature(lbl)}
    
    tf_example = tf.train.Example(features = tf.train.Features(feature = feature))
    return tf_example

# function to create a TFRecordDataset
# Each dataset, such as the train dataset is stored in a file as a binary representation
# After further review of the documentation, I found that I can use multiple files to create the dataset
# I would need to pass a list of files into the TFRecordDataset() function, but that is easy
def createTFRD(target_file, fnames, lbls):
    # fnames & lbls must be the same length, i.e. one label for each file
    
    # (1) open TFRD writer, which writes to the specified file
    writer = tf.python_io.TFRecordWriter(target_file)
    
    # (2) iteratively read files and write to the target file
    for i in range(len(fnames)):
         # read a wavfile and return numpy array
        audio = wav.read_wav(fnames[i])
        # Should I cut the audio to a uniform size? Yes.
        # Cut to first 10 seconds b/c each recording is at least 10 seconds long
        # 44100 Hz = f_sample
        # 44100 Hz * 10 s = 441000 points
        f_sample = 44100
        # five second sample
        start = 2
        end = 3
        # duration = end*f_sample - start*f_sample
        audio = audio[start*f_sample:end*f_sample] # I love slicing!
        audio_raw = audio.tobytes() # Convert to bytestring
        # Load the corresponding label
        lbl = lbls[i]
        tf_example = to_tf_example(audio_raw, lbl) # Convert to tf.Example
        writer.write(tf_example.SerializeToString()) # SerializeToString() converts to a bytestring    
    writer.close()

# (4) split filepaths & labels into train, val, and test
def split(filepaths, lbls, val_prct, test_prct):
    # Note : filepaths must have the same length as lbls
    val_names = filepaths[:int(val_prct*len(filepaths))] # first, get the validation data
    test_names = filepaths[int(val_prct*len(filepaths)):int((test_prct*len(filepaths)))+int((val_prct*len(filepaths)))] # then, the test data
    train_names = filepaths[int(test_prct*len(filepaths))+int(val_prct*len(filepaths)):] # the rest is training data
    val_lbls = lbls[:int(val_prct*len(filepaths))]
    test_lbls = lbls[int(val_prct*len(filepaths)):int(test_prct*len(filepaths))+int(val_prct*len(filepaths))]
    train_lbls = lbls[int(test_prct*len(filepaths))+int(val_prct*len(filepaths)):]
    return train_names, val_names, test_names, train_lbls, val_lbls, test_lbls # Remember that these are just filepaths

####
# Load tf data record files

# (1) parser function/decoder function to read each individual audio sample and its respective label
def parser(serialized_example):
    f_sample = 44100
    start = 2
    end = 3
    
    features = {"audio_raw": tf.io.FixedLenFeature([], tf.string),
                "label": tf.io.FixedLenFeature([], tf.int64)}
    
    tf_example = tf.io.parse_single_example(serialized_example, features)
    
    audio_raw = tf_example["audio_raw"]
    audio = tf.decode_raw(audio_raw, tf.float64, True) # first argument : dtype to write as, second argument : output type, third argument : little-endian
    audio = tf.reshape(audio, ((end*f_sample) - (start*f_sample), 1)) # Reshape to original audio dimension
    audio = tf.cast(audio, tf.float32) # Cast to float32 because that is the standard for machine learning
    lbl = tf_example["label"]
    lbl = tf.cast(lbl, tf.int32)
    return audio, lbl # Should I return a dictionary?

# (2) read tf data record
# The filenames argument to the TFRecordDataset initializer can either be a string, a list of strings, or a tf.Tensor of strings. Therefore if you have two sets of files for training and validation purposes, you can create a factory method that produces the dataset, taking filenames as an input argument:
def loadTFRD(fnames, shuffle, batch_size):
    ds = tf.data.TFRecordDataset(filenames = fnames) # function defined above
    ds = ds.map(parser) # applies the parser to a sample once it extracts it from the .tfrecords file
    
    if(shuffle):
        ds = ds.shuffle(len(fnames)) # buffer_size = number of elements in the ds
        # I could also do the shuffling step with the filename lists, but that won't be necessary if this works
    ds = ds.batch(batch_size, drop_remainder = True) # batch the dataset
    # drop the last batch if it is smaller than the designated batch size
    # ds = ds.make_one_shot_iterator() # deprecated
    ds = ds.repeat(1) # Number of times you can iterate through the dataset before it runs out of elements
    
    return ds