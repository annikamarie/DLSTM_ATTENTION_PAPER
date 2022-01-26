from __future__ import print_function, division
import pandas as pd
import os
from numpy import array
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
import sys
from tensorflow.contrib.rnn import GRUCell, LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from DLSTM_Attention_layers import *
from utils import fit_in_vocabulary, zero_pad, batch_generator
import logging
from tensorflow.contrib import learn
import json
import time


logging.getLogger().setLevel(logging.INFO)
path = '/Users/annikaschoene/Desktop/Code_Last_Statement/Data/all_corpora_unequal.csv'
df = pd.read_csv(path,index_col=None, encoding='UTF-8', engine='python', dtype=str)
h_parameters = json.loads(open('./parameters.json').read())

df_text = df.text
df_label = df['label']

print("===> Loaded Data and Parameters")

# Map the actual labels to one hot labels
labels = sorted(list(set(df_label.tolist())))
one_hot = np.zeros((len(labels), len(labels)), int)
np.fill_diagonal(one_hot, 1)
label_dict = dict(zip(labels, one_hot))
print("===> Finished label encoding")

# Tokenize tweets and map to one_hot labels
def clean_str(s):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(s)
    return s.strip().lower()

x_raw = df_text.apply(lambda x: clean_str(x)).tolist()
ys = df_label.apply(lambda y: label_dict[y]) # is now a series
y_raw = ys.tolist()

""" pad each sentence to the same length and map each word to an id"""
max_tweet_length = max([len(x.split(' ')) for x in x_raw])
logging.info('The maximum length of all sentences: {}'.format(max_tweet_length))
vocab_processor = learn.preprocessing.VocabularyProcessor(max_tweet_length)
x = np.array(list(vocab_processor.fit_transform(x_raw)))
y = np.array(y_raw)

# split data
x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.1, random_state=42)


# shuffle the train set and split the train set into train and dev sets
shuffle_indices = np.random.permutation(np.arange(len(y_)))
x_shuffled = x_[shuffle_indices]
y_shuffled = y_[shuffle_indices]
x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.2)

# save the labels into labels.json since predict.py needs it
with open('./labels.json', 'w') as outfile:
    json.dump(labels, outfile, indent=4)

logging.info('x_train: {}, x_dev: {}'.format(len(x_train), len(x_dev)))
logging.info('y_train: {}, y_dev: {}'.format(len(y_train), len(y_dev)))


"""START GRAPH MODEL HERE"""
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session(config= tf.ConfigProto(allow_soft_placement=True, log_device_placement=False), graph=graph)
    with sess.as_default():
        vocab_size = len(vocab_processor.vocabulary_)
        epochs = h_parameters['num_epochs']
        Evaluate_every = 10
        n_steps = x_train.shape[1]
        n_classes = 3
        vocabulary_size = len(vocab_processor.vocabulary_)
        EMBEDDING_DIM = 20
        ATTENTION_SIZE = 10
        HIDDEN_SIZE = 10
        MODEL_PATH = './model'
        DELTA = 0.5
        KEEP_PROB = 0.5
        BATCH_SIZE = 128
        cell_type = "LSTM"
        assert (cell_type in ["RNN", "LSTM", "GRU"])
        hidden_structs = [10] * 2
        dilations = [1, 2]
        assert (len(hidden_structs) == len(dilations))

        with tf.name_scope('Inputs'):
            input_x = tf.placeholder(tf.int32, [None, n_steps], name='batch_ph')
            input_y = tf.placeholder(tf.float32, [None, n_classes], name='target_ph')
            dropout_keep_prob = tf.placeholder(tf.float32, name='keep_prob_ph')

        with tf.name_scope('Embedding_layer'):
            embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
            embedded_chars = tf.nn.embedding_lookup(embeddings_var, input_x, max_norm=None, validate_indices=True,
                                                    partition_strategy='mod')

        with tf.name_scope('DLSTM_layer'):
            pred = drnn_classification(embedded_chars, hidden_structs, dilations, n_steps, KEEP_PROB, EMBEDDING_DIM,cell_type)

        with tf.name_scope('Attention_layer'):
            attention_output, alphas = attention(pred, ATTENTION_SIZE, return_alphas=True)
            tf.summary.histogram('alphas', alphas)
            drop = tf.nn.dropout(attention_output, dropout_keep_prob)

        with tf.name_scope('Fully_connected_layer'):
            W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, n_classes], stddev=0.1))
            b = tf.Variable(tf.constant(0., shape=[n_classes]))
            y_hat = tf.nn.xw_plus_b(drop, W, b)
            predictions = tf.argmax(input=y_hat, axis=1, name='predictions')
            tf.summary.histogram('W', W)

        with tf.name_scope('Loss'):
            loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=input_y))
            tf.summary.scalar('loss', loss1)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss1, global_step=global_step)

        with tf.name_scope('Accuracy'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.nn.softmax(y_hat)), input_y), tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        with tf.name_scope('correct_predictions'):
            correct_predictions = tf.equal(tf.argmax(input=input_y, axis=1), predictions)
            num_correct = tf.reduce_sum(input_tensor=tf.cast(correct_predictions, 'float'), name='correct_predictions')

        merged = tf.summary.merge_all()

        # build the graph for learning
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver()

        def train_step(x_batch, y_batch):
            feed_dict = {
                input_x: x_batch,
                input_y: y_batch,
                dropout_keep_prob: KEEP_PROB}
            _, step, loss, acc = sess.run([optimizer, global_step, loss1, accuracy], feed_dict)

            # One evaluation step: evaluate the model with one batch

        def dev_step(x_batch, y_batch):
            feed_dict = {input_x: x_batch, input_y: y_batch, dropout_keep_prob: 1.0}
            step, loss, acc, correc_pred = sess.run([global_step, loss1, accuracy, num_correct], feed_dict)
            return correc_pred

        def batch_iter(data, batch_size, num_epochs, shuffle=True):
            """Iterate the data batch by batch"""
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int(data_size / batch_size) + 1

            for epoch in range(num_epochs):
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffled_data = data[shuffle_indices]
                else:
                    shuffled_data = data

                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    yield shuffled_data[start_index:end_index]

        # Save the word_to_id map since predict.py needs it
        vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
        sess.run(tf.global_variables_initializer())

        # Training starts here
        train_batches = batch_iter(list(zip(x_train, y_train)), BATCH_SIZE, epochs)
        best_accuracy, best_at_step = 0, 0

        # Train the model
        for train_batch in train_batches:
            x_train_batch, y_train_batch = zip(*train_batch)
            train_step(x_train_batch, y_train_batch)
            current_step = tf.train.global_step(sess, global_step)

            # evaluate the model with x_dev
            if current_step % Evaluate_every == 0:
                dev_batches = batch_iter(list(zip(x_dev, y_dev)), BATCH_SIZE, 1)
                total_dev_correct = 0
                for dev_batch in dev_batches:
                    x_dev_batch, y_dev_batch = zip(*dev_batch)
                    num_dev_correct = dev_step(x_dev_batch, y_dev_batch)
                    total_dev_correct += num_dev_correct

                dev_accuracy = float(total_dev_correct) / len(y_dev)
                logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))

                # save model based on best accruacy of dev set
                if dev_accuracy >= best_accuracy:
                    best_accuracy, best_at_step = dev_accuracy, current_step
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logging.critical('Saved model at {} at step {}'.format(path, best_at_step))
                    logging.critical('Best accuracy is {} at step {}'.format(best_accuracy, best_at_step))


        # predict x_test
        test_batches = batch_iter(list(zip(x_test, y_test)), BATCH_SIZE, 1)
        total_test_correct = 0
        for test_batch in test_batches:
            x_test_batch, y_test_batch = zip(*test_batch)
            num_test_correct = dev_step(x_test_batch, y_test_batch)
            total_test_correct += num_test_correct

        test_accuracy = float(total_test_correct) / len(y_test)
        logging.critical('Accuracy on test set is {} based on the best model {}'.format(test_accuracy, path))
        logging.critical('The training is complete')
        save_path = saver.save(sess, MODEL_PATH)