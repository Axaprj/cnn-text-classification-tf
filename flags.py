#! /usr/bin/env python

import tensorflow as tf
# Parameters
# ==================================================
# Data loading params
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
# Calculation data storage params
tf.flags.DEFINE_string("data_file", "./data/data_file.npz", "data[x] data[y] vectors")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
