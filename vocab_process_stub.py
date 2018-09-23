#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import flags
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================
FLAGS = tf.flags.FLAGS

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    vocab_size = len(vocab_processor.vocabulary_)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    # Write vocabulary
    #vocab_processor.save(os.path.join(out_dir, "vocab"))
    print("Building Embedded presentation...")
    # Embedding layer
    W = tf.Variable(
          tf.random_uniform([vocab_size, FLAGS.embedding_dim], -1.0, 1.0), name="W")
    embedded_chars = tf.nn.embedding_lookup(W, x)
    with tf.Session() as session:
            session.run(tf.global_variables_initializer())  # reset values to wrong
            x = session.run(embedded_chars)
    print("Done.")
    return x, y

def main(argv=None):
    x, y = preprocess()
    np.savez(FLAGS.data_file, x=x, y=y)

if __name__ == '__main__':
    tf.app.run()