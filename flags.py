#! /usr/bin/env python

import tensorflow as tf
# Parameters
# ==================================================
# Data loading params
#tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string(
    "negative_data_dir",
    "./../data.arc/aclImdb/test/neg/*.txt",  #"./../data.arc/aclImdb/train/neg/*.txt",
    "Data source dir/mask for the negative data.")
tf.flags.DEFINE_string(
    "positive_data_dir",
    "./../data.arc/aclImdb/test/pos/*.txt",  #"./../data.arc/aclImdb/train/pos/*.txt",
    "Data source dir/mask for the positive data.")

# Calculation data storage params
tf.flags.DEFINE_string(
    "words_dic_file",
    "./data/aclimdb_words_dict_test.csv",  #"./data/aclimdb_words_dict.csv",
    "Ordered words dictionary")
tf.flags.DEFINE_string(
    "data_file",
    "./data/aclimdb_data_file_test.npz",  #"./data/aclimdb_data_file.npz",
    "data[x] data[y] data[embed_dict] vectors")

# Model Hyperparameters
tf.flags.DEFINE_integer(
    "embedding_dim", 300,
    "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_integer(
    "max_document_length", 2633,
    "Max number of words in a sample (default: 2633)")

 