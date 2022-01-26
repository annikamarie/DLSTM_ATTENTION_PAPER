import pandas as pd

path = '/Users/annikaschoene/Desktop/Code_Last_Statement/Data/all_corpora_unequal.csv'
#path = "/Users/annikaschoene/Desktop/Dilated_RNN/0_Data_Preprocessing/Ekman_10K_no_pos.csv"
#path = "/Users/annikaschoene/Desktop/Attention_Zip/attention_test.csv"
df = pd.read_csv(path,index_col=None, encoding='UTF-8', engine='python', dtype=str)

df_text = df.text
text = df_text.tolist()

from DLSTM_ATT import *

saver = tf.train.Saver()

# Calculate alpha coefficients for the first test example
with tf.Session() as sess:
    saver.restore(sess, MODEL_PATH)

    x_batch_test, y_batch_test = X_test[6:], y_test[6:]
    seq_len_test = np.array([list(x).index(0) + 1 for x in x_batch_test])
    alphas_test = sess.run([alphas], feed_dict={input_x: x_batch_test, input_y: y_batch_test,
                                                dropout_keep_prob: 1.0})
alphas_values = alphas_test[0][0]

# Build correct mapping from word to index and inverse
all_tweets = text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_tweets)
print("====>Fitting is complete.")
# word to number mapping
word_index = tokenizer.word_index
print("Vocabulary: ",len(word_index))


word_index = {word: index + INDEX_FROM for word, index in word_index.items()}
word_index[":PAD:"] = 0
word_index[":START:"] = 1
word_index[":UNK:"] = 2
index_word = {value: key for key, value in word_index.items()}
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
