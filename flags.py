#! /usr/bin/env python

import tensorflow as tf
# Parameters
# ==================================================
# Data loading params
#tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

tf.flags.DEFINE_string(
    "data_file",
    #"./data/aclimdb_data_file_test.npz",  
    "./data/aclimdb_data_file.npz",
    "data[x] data[y] data[embed_dict] vectors")

tf.flags.DEFINE_string(
    "data_file_test",
    "./data/aclimdb_data_file_test.npz",  
    "data[x] data[y] data[embed_dict] vectors")

# Model Parameters
tf.flags.DEFINE_string(
    "out_dir_name", "aclImdb",
    "Output model directory name (default: timestamp)")

# Model Hyperparameters
tf.flags.DEFINE_integer(
    "embedding_dim", 300,
    "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_integer(
    "max_document_length", 2800,
    "Max number of words in a sample (default: 2800)")

 