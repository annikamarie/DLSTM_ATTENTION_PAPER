from __future__ import print_function, division

from sklearn.model_selection import train_test_split
from DLSTM_Attention_layers import *
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import logging
import utils
import copy
import json
import time
import sys
import os

logging.getLogger().setLevel(logging.INFO)
path = '/Users/annikaschoene/Desktop/Code_Last_Statement/Data/all_corpora_unequal.csv'
#h_parameters = json.loads(open('./parameters.json').read())

x_, y_, vocabulary,vocabulary_inv,dataframe ,labels = utils.load_data(path)
print("===> Loaded Data and Parameters")

# Tokenize tweets and map to one_hot labels
word_embeddings = utils.load_embeddings(vocabulary)

# split data
x_train, x_dev, y_train, y_dev = train_test_split(x_, y_, test_size=0.2, random_state=42, shuffle=True)

#x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.2, shuffle=True)

# save the labels into labels.json since predict.py needs it
with open('./labels.json', 'w') as outfile:
    json.dump(labels, outfile, indent=4)

#logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
#logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

# configurations
n_steps = x_train.shape[1]
n_classes = 3
vocabulary_size = len(vocabulary)
EMBEDDING_DIM = 200
ATTENTION_SIZE = 300
HIDDEN_SIZE = 300
MODEL_PATH = './model_equal/test'
DELTA = 0.5
KEEP_PROB = 0.5
BATCH_SIZE = 128
NUM_EPOCHS = 1
INDEX_FROM = 0

# model config
cell_type = "LSTM"
assert(cell_type in ["RNN", "LSTM", "GRU"])
hidden_structs = [300] *2
dilations = [1, 2]
assert(len(hidden_structs) == len(dilations))

with tf.name_scope('Inputs'):
    x = tf.placeholder(tf.int32, [None, n_steps], name='batch_ph')
    y = tf.placeholder(tf.float32, [None, n_classes], name='target_ph')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

with tf.name_scope('Embedding_layer'):
    embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0,1.0), trainable=True)
    embedded_chars = tf.nn.embedding_lookup(embeddings_var, x, max_norm=None, validate_indices=True, partition_strategy='mod')

pred = drnn_classification(embedded_chars, hidden_structs, dilations, n_steps, KEEP_PROB, EMBEDDING_DIM,cell_type)

with tf.name_scope('Attention_layer'):
    attention_output, alphas = attention(pred, ATTENTION_SIZE, return_alphas=True)
    tf.summary.histogram('alphas', alphas)
    drop = tf.nn.dropout(attention_output, keep_prob_ph)

with tf.name_scope('Fully_connected_layer'):
    W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, n_classes], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[n_classes]))
    y_hat = tf.nn.xw_plus_b(drop, W, b)
    predictions = tf.argmax(input=y_hat, axis=1, name='predictions')
    tf.summary.histogram('W', W)

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=y))
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.nn.softmax(y_hat)), y), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('correct_predictions'):
    correct_predictions = tf.equal(tf.argmax(input=y, axis=1), predictions)
    num_correct = tf.reduce_sum(input_tensor=tf.cast(correct_predictions, 'float'),name='correct_predictions')

merged = tf.summary.merge_all()

# Batch generators
train_batch_generator = utils.batch_generator(x_train, y_train, BATCH_SIZE)
test_batch_generator = utils.batch_generator(x_dev, y_dev, BATCH_SIZE)
#predict_generator = utils.batch_generator(x_test, y_test, BATCH_SIZE)

#session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    sess.run(tf.global_variables_initializer())
    print("Start learning...")
    for epoch in range(NUM_EPOCHS):
        loss_train = 0
        loss_test = 0
        accuracy_train = 0
        accuracy_test = 0

        print("epoch: {}\t".format(epoch), end="")

        # Training
        num_batches = x_train.shape[0] // BATCH_SIZE
        for b in tqdm(range(num_batches)):
            x_batch, y_batch = next(train_batch_generator)
            loss_tr, acc, _, summary = sess.run([loss, accuracy, optimizer, merged],
                                                feed_dict={x: x_batch,
                                                           y: y_batch,
                                                           keep_prob_ph: KEEP_PROB})
            accuracy_train += acc
            loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
        accuracy_train /= num_batches

        # Validation
        num_batches = x_dev.shape[0] // BATCH_SIZE
        for b in tqdm(range(num_batches)):
            x_batch, y_batch = next(test_batch_generator)
            loss_test_batch, acc, summary = sess.run([loss, accuracy, merged],
                                                     feed_dict={x: x_batch,
                                                                y: y_batch,
                                                                keep_prob_ph: 1.0})
            accuracy_test += acc
            loss_test += loss_test_batch
        accuracy_test /= num_batches
        loss_test /= num_batches

        print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
            loss_train, loss_test, accuracy_train, accuracy_test
        ))


    # predict x_test
    #num_batches = x_test.shape[0] // BATCH_SIZE
    #predict_correct = 0
    #for batch in tqdm(range(num_batches)):
        #x_batch, y_batch = next(predict_generator)
        #loss_pred, acc_pred, n_correct = sess.run([loss,accuracy,num_correct], feed_dict={
         #   x: x_batch,
         #   y: y_batch,
         #   keep_prob_ph : 0.5 })

        #predict_correct += n_correct

    #pred_accuracy = float(predict_correct)/len(y_test)
    #print(pred_accuracy)

    saver.save(sess, MODEL_PATH)
    print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")