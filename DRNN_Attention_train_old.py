from __future__ import print_function, division
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
import json
import copy
import itertools
import time
import utils
from drnn_for_attention import drnn_classification
from attention_layer_for_drnn import attention

logging.getLogger().setLevel(logging.INFO)
path = '/Users/annikaschoene/Desktop/Code_Last_Statement/Data/CombinedData_no_german_with_DL.csv'

x_, y_, vocabulary,vocabulary_inv,dataframe ,labels = utils.load_data(path)
print("===> Loaded Data and Parameters")

# Tokenize tweets and map to one_hot labels
word_embeddings = utils.load_embeddings(vocabulary)

# split data
x, x_test, y, y_test = train_test_split(x_, y_, test_size=0.1, random_state=42)

# shuffle the train set and split the train set into train and dev sets
shuffle_indices = np.random.permutation(np.arange(len(y_)))
x_shuffled = x_[shuffle_indices]
y_shuffled = y_[shuffle_indices]
x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1)

# save the labels into labels.json since predict.py needs it
with open('./labels.json', 'w') as outfile:
    json.dump(labels, outfile, indent=4)

logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

# configurations
n_steps = x_train.shape[1]
n_classes = 3
vocabulary_size = len(vocabulary)
EMBEDDING_DIM = 200
ATTENTION_SIZE = 50
HIDDEN_SIZE = 20
MODEL_PATH = './model'
DELTA = 0.5
KEEP_PROB = 0.5

# model config
cell_type = "LSTM"
assert(cell_type in ["RNN", "LSTM", "GRU"])
hidden_structs = [20] *2  # hidden dimension is 20 and it is a 9  layer dilated RNN
dilations = [1, 2] # dilations don't have to start a 1 and dont have to increase by power of 2
assert(len(hidden_structs) == len(dilations)) # assert is a way of debugging

# build computation graph / create placeholders
# Different placeholders
with tf.name_scope('Inputs'):
    x = tf.placeholder(tf.int32, [None, n_steps], name='batch_ph')
    y = tf.placeholder(tf.float32, [None, n_classes], name='target_ph')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

with tf.name_scope('Embedding_layer'):
    embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0,1.0), trainable=True)
    embedded_chars = tf.nn.embedding_lookup(embeddings_var, x, max_norm=None, validate_indices=True, partition_strategy='mod')

global_step = tf.Variable(0, name='global_step', trainable=False)


pred = drnn_classification(embedded_chars, hidden_structs, dilations, n_steps, KEEP_PROB, EMBEDDING_DIM,
                           cell_type)

# Attention layer
with tf.name_scope('Attention_layer'):
    attention_output, alphas = attention(pred, ATTENTION_SIZE, return_alphas=True)
    tf.summary.histogram('alphas', alphas)
drop = tf.nn.dropout(attention_output, keep_prob_ph)

# Fully connected layer
with tf.name_scope('Fully_connected_layer'):
    W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, n_classes], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[n_classes]))
    y_hat = tf.nn.xw_plus_b(drop, W, b)
    predictions = tf.argmax(input=y_hat, axis=1, name='predictions')
    tf.summary.histogram('W', W)

with tf.name_scope('Loss'):
    # Cross-entropy loss and optimizer initialization
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=y))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

with tf.name_scope('Accuracy'):
    # Accuracy metric
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.nn.softmax(y_hat)), y), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('correct_predictions'):
    correct_predictions = tf.equal(tf.argmax(input=y, axis=1), predictions)
    num_correct = tf.reduce_sum(input_tensor=tf.cast(correct_predictions, 'float'),
                                     name='correct_predictions')

merged = tf.summary.merge_all()

def next_batchs(batch_size, data, labels):
    """
    Return a total of `num` random samples and labels.
    """
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# start training the model
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# model config
batch_size= 128
step = 0
train_results = []
validation_results = []
test_results = []
training_iters = batch_size * 200
display_step = 10
testing_step = 10

saver = tf.train.Saver()
#MODEL_PATH = './model'

while step * batch_size < training_iters:
    batch_x, batch_y = next_batchs(batch_size,x_train,y_train)

    feed_dict = {x: batch_x,
                 y: batch_y,
                 keep_prob_ph:KEEP_PROB}

    cost_, accuracy_, step_,  _ = sess.run([loss, accuracy, global_step, optimizer], feed_dict=feed_dict)
    train_results.append((step_, cost_, accuracy_))

    # Display logs per epoch step
    if (step + 1) % display_step == 0:
        print("Iter " + str(step + 1) + ", Minibatch Loss: " + "{:.6f}".format(cost_) + ", Training Accuracy: "
              + "{:.6f}".format(accuracy_))

    if (step + 1) % testing_step == 0:

        # validation performance
        batch_x = x_dev
        batch_y = y_dev

        feed_dict = {x : batch_x,
                    y : batch_y,
                     keep_prob_ph: KEEP_PROB}

        cost_, accuracy__, step_ = sess.run([loss, accuracy, global_step], feed_dict=feed_dict)
        validation_results.append((step_, cost_, accuracy__))

        # test performance
        batch_x = x_test
        batch_y = y_test
        feed_dict = {x : batch_x,
                     y : batch_y,
                     keep_prob_ph: KEEP_PROB}

        cost_, accuracy_, step_ = sess.run([loss, accuracy, global_step], feed_dict=feed_dict)
        test_results.append((step_, cost_, accuracy_))
        print("========> Validation Accuarcy: " + "{:.6f}".format(accuracy__)
        + ", Testing Accuarcy: " + "{:.6f}".format(accuracy_))
    step += 1
