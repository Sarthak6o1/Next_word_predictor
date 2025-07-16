# Next-Word Prediction with RNN, GRU, and LSTM

This project demonstrates next-word prediction using different types of recurrent neural networks: **SimpleRNN**, **GRU**, and **LSTM**. Using the Medium Articles dataset, this project covers data preprocessing, model building, training, evaluation, and interactive text generation.

---

## Project Overview

- **Dataset**: Medium articles titles (`medium_data.csv`) from Kaggle (`dorianlazar/medium-articles-dataset`).
- **Task**: Predict the next word in a sequence from input text.
- **Models**: 
  - SimpleRNN (baseline)
  - GRU (gated recurrent unit for better memory)
  - LSTM (long short-term memory for long-range dependencies)
- **Outputs**: Model accuracy and loss plots, and word-by-word text generation from user seed input.

---

## Dataset Preparation

- Load dataset titles.
- Clean special Unicode characters.
- Use Keras Tokenizer to create word indices.
- Generate n-gram sequences for supervised training.
- Pad sequences to uniform length.
- One-hot encode the next word labels.

---

## Model Architectures

### SimpleRNN

A single recurrent layer with 150 units.


![SimpleRNN Architecture] SimpleRNN processes sequences step-by-step with a hidden state passed forward.*

---

### GRU (Gated Recurrent Unit)

Improves memory retention with gating mechanisms, reducing vanishing gradients.


![GRU Architecture]
GRU uses update and reset gates to regulate information flow.*

---

### LSTM (Long Short-Term Memory)

Uses input, forget, and output gates to model complex, long-term dependencies.

![LSTM Architecture]
LSTM cell structure with gating mechanisms and memory cell.*

---

## Training Details

- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- Metrics: Accuracy
- Epochs: 50
- Batch size: Configurable

---

## Usage

After running training:

1. View accuracy and loss plots.
2. Input a text seed at prompt.
3. Model outputs predicted next word iteratively to generate text.

Example:

Enter a text: deep learning is
deep learning is powerful
deep learning is powerful and
deep learning is powerful and flexible


---

## Plotting Training Metrics

Use this helper function:


import matplotlib.pyplot as plt

def plot_graphs(history, metric):
plt.plot(history.history[metric])
plt.xlabel('Epochs')
plt.ylabel(metric.capitalize())
plt.show()

Usage
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

---

## Requirements

- pandas
- numpy
- tensorflow
- matplotlib
- kagglehub

Install via pip:

pip install pandas numpy tensorflow matplotlib kagglehub

text

---

## References

- [Keras Documentation](https://keras.io/api/layers/recurrent_layers/)
- [Medium Articles Dataset on Kaggle](https://www.kaggle.com/dorianlazar/medium-articles-dataset)

---

*Replace image placeholders with your architecture diagrams to visualize the model components.*

---

Future idea training Transformer architecture for more enhanced outputs.
