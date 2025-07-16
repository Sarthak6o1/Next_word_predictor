import kagglehub
import os
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential

# Download and load the Medium articles dataset
dorianlazar_medium_articles_dataset_path = kagglehub.dataset_download('dorianlazar/medium-articles-dataset')
csv_file_path = os.path.join(dorianlazar_medium_articles_dataset_path, 'medium_data.csv')
medium_data = pd.read_csv(csv_file_path)
print("Data source import complete.")
print("Records Total: ", medium_data.shape[0])
print("Fields Total: ", medium_data.shape[1])

# Clean and preprocess titles
medium_data['title'] = medium_data['title'].astype(str).apply(lambda x: x.replace(u'\xa0', u' ').replace('\u200a', ' '))

tokenizer = Tokenizer(oov_token='<oov>')
tokenizer.fit_on_texts(medium_data['title'])
total_words = len(tokenizer.word_index) + 1
print("Total number of words: ", total_words)

# Construct n-gram input sequences
input_sequences = []
for line in medium_data['title']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

print("Total input sequences: ", len(input_sequences))

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

x = input_sequences[:, :-1]
labels = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Define and train the GRU model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
model.add(GRU(150))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x, y, epochs=50, verbose=1)

import matplotlib.pyplot as plt
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

import time
text = input("Enter a text: ")

for _ in range(10):
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded_token_text = pad_sequences([token_text], maxlen=max_sequence_len - 1, padding='pre')
    pos = np.argmax(model.predict(padded_token_text, verbose=0))
    for word, index in tokenizer.word_index.items():
        if index == pos:
            text = text + " " + word
            print(text)
            time.sleep(2)

from IPython.display import Audio
from google.colab import output

def speak(word):
    output.eval_js(f'new Audio("https://dict.youdao.com/dictvoice?audio={word}").play()')

text = input("Enter a text: ")

for _ in range(9):
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded_token_text = pad_sequences([token_text], maxlen=max_sequence_len - 1, padding='pre')
    predicted_probabilities = model.predict(padded_token_text, verbose=0)[0]
    pos = np.argmax(predicted_probabilities)
    for word, index in tokenizer.word_index.items():
        if index == pos:
            text = text + " " + word
            print(text)
            speak(word)
            time.sleep(2)
