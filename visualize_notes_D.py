import pandas as pd
import utils
import json
import sys
import os


path = '/Users/annikaschoene/Desktop/Code_Last_Statement/Data/CombinedData_no_german_with_DL.csv'
df = pd.read_csv(path,index_col=None, encoding='UTF-8', engine='python', dtype=str)
#h_parameters = json.loads(open('./parameters.json').read())
df_text = df.text
text = df_text.tolist()

from DLSTM_Attention_training import *

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, MODEL_PATH)

    x_batch_test, y_batch_test = x_dev[1:], y_dev[1:]

    alphas_test = sess.run([alphas], feed_dict={x: x_batch_test, y: y_batch_test,
                                                keep_prob_ph: 1.0})
alphas_values = alphas_test[0][0]

x_, y_, vocabulary,vocabulary_inv,dataframe ,labels = utils.load_data(path)

word_index = {word: index + INDEX_FROM for word, index in vocabulary.items()}
word_index[":PAD:"] = 0
word_index[":START:"] = 1
word_index[":UNK:"] = 2
index_word = {value: key for key, value in vocabulary.items()}
words = list(map(index_word.get, x_batch_test[0]))

# Save visualization as HTML
with open("visualization.html", "w") as html_file:
    for word, alpha in zip(words, alphas_values / alphas_values.max()):
        if word == ":START:":
            continue
        elif word == ":PAD:":
            break
        html_file.write('<font style="background: rgba(255, 255, 0, %f)">%s</font>\n' % (alpha, word))

print('\nOpen visualization.html to checkout the attention coefficients visualization.')

