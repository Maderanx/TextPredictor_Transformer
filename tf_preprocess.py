from nltk.stem import WordNetLemmatizer
import numpy as np
sequence_len=4

def text_preprocess(Sentence_Data):
    # Tokenizing Data
    tokenized_data = [[word for word in sentence[0].split()] for sentence in Sentence_Data]
    print("Tokenized Data:")
    print(tokenized_data)
    print('\n')

    # Making all lower case
    lower_Sentence_Data = [[word.lower() for word in sentence] for sentence in tokenized_data]
    print("Lower Case Data:")
    print(lower_Sentence_Data)
    print('\n')

    # Lemmatizing 
    Lemmatizer = WordNetLemmatizer()
    Lemmatized_data = [[Lemmatizer.lemmatize(word) for word in sentence] for sentence in lower_Sentence_Data]
    print("Lemmatized Data:")
    print(Lemmatized_data)
    print('\n')

    # Dictionary
    dictionary = list(set([word for sentence in Lemmatized_data for word in sentence]))
    dictionary.sort()
    dict_size = len(dictionary)
    dictionary = {word: i for i, word in enumerate(dictionary)}  # Start indices from 1
    print("Dictionary:")
    print(dictionary)
    print('\n')
    
    # Sequencing
    # text to sequence no
    output_vector = [[dictionary[word] for word in sentence] for sentence in Lemmatized_data]
    print("Output Vector (Sequences):")
    for sequence in output_vector:
        print(sequence)
    print('\n')
    
    # Flatten 
    flat_output_vector = [word for sentence in output_vector for word in sentence]
    print(flat_output_vector)

    sequences = []
    for i in range(len(flat_output_vector) - sequence_len + 1):
        seq = flat_output_vector[i:i + sequence_len]
        sequences.append(seq)

    sequences=np.array(sequences)
    x=sequences[:,:-1]
    y=sequences[:,1:]
    print(x.shape)
    print(y.shape)
    return x, y, dict_size, dictionary, sequence_len
