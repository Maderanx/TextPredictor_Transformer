# Transformer-Based Text Prediction Model

This repository contains the implementation of a transformer-based model for text prediction using Keras. The model predicts the next word in a sequence given an input sentence. 

## Overview

The project demonstrates a transformer-based text prediction model. The model leverages multi-head attention and positional encoding to process input sentences and predict the next word in the sequence.

## Features

- Multi-head attention mechanism
- Positional encoding for sequence processing
- Custom text preprocessing
- Trained on a dataset of scientific facts
- Predicts the next word in a given sentence

## Technologies Used

- Python
- TensorFlow
- Keras
- Numpy

## Installation

To run this project, you need to have Python installed along with the required libraries. You can install the necessary libraries using pip:

```bash
pip install numpy tensorflow keras
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/Maderanx/transformer-text-prediction.git
cd transformer-text-prediction
```

2. Open and run the code in Google Colaboratory or your local Python environment.

3. Ensure you have the following files in your working directory:
   - `tf_preprocess.py`: Contains the `text_preprocess` function.
   - `Positional_encoding.py`: Contains functions for positional encoding.

4. Run the script to preprocess the data, build the transformer model, train it, and make predictions.

## Model Architecture

The model uses the following architecture:
- Embedding Layer
- Positional Encoding
- Multiple layers of Multi-Head Attention and Feed-Forward Networks
- Output Dense Layer with Softmax activation

## Training

The model is trained on a dataset of scientific facts. The dataset is preprocessed to convert sentences into sequences of indices. The training parameters are as follows:
- Epochs: 300
- Batch size: 100
- Loss: Sparse Categorical Crossentropy
- Optimizer: Adam

## Prediction

To make predictions with the trained model, input a new sentence, preprocess it to convert it into a sequence of indices, and pass it to the model. The model will output the predicted next word in the sequence.

Example:
```python
# Convert a new sentence to its corresponding sequence of indices
new_sentence = "Zinc is a"
new_sequence = [dictionary[word.lower()] for word in new_sentence.split()]

# Ensure the input is of the correct length
input_sequence = np.array(new_sequence).reshape(1, -1)

# Make prediction
prediction = model.predict(input_sequence)
predicted_index = np.argmax(prediction, axis=-1)

# Convert predicted indices back to words
reverse_dictionary = {index: word for word, index in dictionary.items()}
predicted_words = [reverse_dictionary[idx] for idx in predicted_index[0]]

print("Input sequence: ", new_sentence)
print("Predicted next tokens: ", predicted_words)
```
