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
from utils import *
import logging
from tensorflow.contrib import learn
import json
import time


logging.getLogger().setLevel(logging.INFO)
path = '/Users/annikaschoene/Desktop/Code_Last_Statement/Data/all_corpora_unequal.csv'
df = pd.read_csv(path,index_col=None, encoding='UTF-8', engine='python', dtype=str)
df_text = df.text
text = df_text.tolist()
df_label = df['label']
label = df_label.values

# Tokenize tweets
all_tweets = text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_tweets)
print("====>Fitting is complete.")

text_seq = tokenizer.texts_to_sequences(text)
print("====>Text_to_sequence is completed")

# word to number mapping
word_index = tokenizer.word_index
print("Vocabulary: ",len(word_index))
reversed_dictionary = dict(zip(word_index.values(), word_index.keys()))


label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(label)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
one_hot_labels = onehot_encoder.fit_transform(integer_encoded)
print("====> Finished one-hot encoding")

flat_list = [item for sublist in integer_encoded for item in sublist]
flat_list = array(flat_list)

X = text_seq
Y = one_hot_labels

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


# Sequences pre-processing
vocabulary_size = get_vocabulary_size(X_train)
X_test = fit_in_vocabulary(X_test, vocabulary_size)
X_train = zero_pad(X_train, SEQUENCE_LENGTH)
X_test = zero_pad(X_test, SEQUENCE_LENGTH)

vocab_size = len(vocab_processor.vocabulary_)
epochs = h_parameters['num_epochs']
Evaluate_every = h_parameters['Evaluate_every']
n_steps = x_train.shape[1]
n_classes = 3
vocabulary_size = len(vocab_processor.vocabulary_)
EMBEDDING_DIM = 200
ATTENTION_SIZE = 300
HIDDEN_SIZE = 300
MODEL_PATH = './model'
DELTA = 0.5
KEEP_PROB = 0.5
BATCH_SIZE = 128
NUM_EPOCHS = 100
INDEX_FROM = 0
cell_type = "LSTM"
assert (cell_type in ["RNN", "LSTM", "GRU"])
hidden_structs = [300] * 2
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
    global_step = tf.Variable(0, name='global_step', trainable=False)
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
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss1)

with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.nn.softmax(y_hat)), input_y), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('correct_predictions'):
    correct_predictions = tf.equal(tf.argmax(input=input_y, axis=1), predictions)
    num_correct = tf.reduce_sum(input_tensor=tf.cast(correct_predictions, 'float'), name='correct_predictions')

    merged = tf.summary.merge_all()

# Batch generators
train_batch_generator = batch_generator(X_train, y_train, BATCH_SIZE)
test_batch_generator = batch_generator(X_test, y_test, BATCH_SIZE)

if __name__ == "__main__":
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        print("Start learning...")
        for epoch in range(NUM_EPOCHS):
            loss_train = 0
            loss_test = 0
            accuracy_train = 0
            accuracy_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training
            num_batches = X_train.shape[0] // BATCH_SIZE
            for b in tqdm(range(num_batches)):
                x_batch, y_batch = next(train_batch_generator)
                loss_tr, acc, _, summary = sess.run([loss1, accuracy, optimizer, merged],
                                                    feed_dict={input_x: x_batch,
                                                               input_y: y_batch,

                                                               dropout_keep_prob: KEEP_PROB
                                                               })
                #seq_len_ph: seq_len,
                accuracy_train += acc
                loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
                train_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_train /= num_batches

            # Testing
            num_batches = X_test.shape[0] // BATCH_SIZE
            for b in tqdm(range(num_batches)):
                x_batch, y_batch = next(test_batch_generator)
                # seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                loss_test_batch, acc, summary = sess.run([loss1, accuracy, merged],
                                                         feed_dict={input_x: x_batch,
                                                                    input_y: y_batch,

                                                                    dropout_keep_prob: 1.0}) #seq_len_ph: seq_len,
                accuracy_test += acc
                loss_test += loss_test_batch
                test_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_test /= num_batches
            loss_test /= num_batches

            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
                loss_train, loss_test, accuracy_train, accuracy_test
            ))
        train_writer.close()
        test_writer.close()
        saver.save(sess, MODEL_PATH)
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")