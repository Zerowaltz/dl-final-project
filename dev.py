import pandas as pd
import tensorflow.python.eager.context
from math import nan
from future.utils import iteritems
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from decoder import InferNER
from tensorflow import python as tf_python
import tensorflow as tf
import numpy as np
tf.config.run_functions_eagerly(True)
# load in data
# def load_data():
#     df = pd.DataFrame()
#     for filename in os.listdir("items"):
#         with open(os.path.join(os.cwd(), filename), 'r', encoding = 'utf-8') as f:
#             df = df.append(pd.read_json(f))
    
#     df.to_csv("processed_input.csv", encoding = 'utf-8', index = False)

# if __name__ == "main":
#     load_data()

class SentenceGetter(object):
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),
                                                     s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

################################################################################

df = pd.read_csv("ner.csv", encoding="ISO-8859-1", error_bad_lines=False)
dataset = df[["sentence_idx", "word", "tag"]]


getter = SentenceGetter(dataset)
sentences = getter.sentences

words = list(set(dataset["word"].values))
num_words = len(words)

tags = []
for tag in set(dataset["tag"].values):
    if tag is nan or isinstance(tag, float):
        tags.append('unk')
    else:
        tags.append(tag)
num_tags = len(tags)


word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {v: k for k, v in iteritems(tag2idx)}

maxlen = max([len(s) for s in sentences])

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=maxlen, sequences=X, padding='post', value=(num_words - 1))

Y = [[tag2idx[w[1]] for w in s] for s in sentences]
Y = pad_sequences(maxlen=maxlen, sequences=Y, padding='post', value=tag2idx["O"])
Y = [to_categorical(i, num_classes=num_tags) for i in Y]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
tf.data.experimental.enable_debug_mode()


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = tf.keras.models.Sequential()
model.add(InferNER(num_words, 128, 128))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss="sce", run_eagerly=True)
model.run_eagerly = True
tf.config.run_functions_eagerly(True)
print(tf.executing_eagerly())
print("checkpoint")
model.fit(X, np.array(Y), batch_size=256, epochs=10, validation_split=0.2, verbose=1, callbacks=[callback])
model.summary()