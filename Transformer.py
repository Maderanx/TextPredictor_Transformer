import numpy as np
from keras.layers import Embedding, Input, Add, MultiHeadAttention, Dense, LayerNormalization, Dropout
from keras.models import Model

# Define the text_preprocess function
from tf_preprocess import text_preprocess
from Positional_encoding import get_positional_encoding, add_positional_encoding

# Example sentence data for demonstration purposes
Sentence_Data = [
    ["The speed of light is approximately 299,792 kilometers per second."],
    ["Water is made up of two hydrogen atoms and one oxygen atom."],
    ["The Earth revolves around the Sun once every 365.25 days."],
    ["The human body contains 206 bones."],
    ["DNA stands for Deoxyribonucleic Acid."],
    ["The Milky Way galaxy is about 100,000 light-years in diameter."],
    ["The average human body temperature is about 37 degrees Celsius."],
    ["Photosynthesis is the process by which plants make their own food using sunlight."],
    ["There are eight planets in our solar system."],
    ["The Great Barrier Reef is the largest coral reef system in the world."],
    ["The human brain contains approximately 86 billion neurons."],
    ["The Pacific Ocean is the largest ocean on Earth."],
    ["The chemical symbol for gold is Au."],
    ["A light-year is the distance light travels in one year."],
    ["The Hubble Space Telescope was launched in 1990."],
    ["The mitochondria is the powerhouse of the cell."],
    ["A proton is a subatomic particle with a positive charge."],
    ["The Earth's atmosphere is composed of 78% nitrogen."],
    ["The largest planet in our solar system is Jupiter."],
    ["Albert Einstein developed the theory of relativity."],
    ["A neutron is a subatomic particle with no charge."],
    ["The chemical formula for table salt is NaCl."],
    ["The Andromeda Galaxy is the closest spiral galaxy to the Milky Way."],
    ["An atom is the smallest unit of matter that retains the properties of an element."],
    ["The human heart has four chambers."],
    ["The boiling point of water is 100 degrees Celsius at standard atmospheric pressure."],
    ["The chemical symbol for oxygen is O."],
    ["The universe is approximately 13.8 billion years old."],
    ["A molecule is a group of atoms bonded together."],
    ["The Earth's core is composed primarily of iron and nickel."],
    ["The smallest bone in the human body is the stapes, located in the ear."],
    ["The chemical symbol for carbon is C."],
    ["Black holes are regions of space where gravity is so strong that not even light can escape."],
    ["The atomic number of hydrogen is 1."],
    ["The Amazon Rainforest produces about 20% of the world's oxygen."],
    ["The human body is about 60% water."],
    ["The speed of sound in air is approximately 343 meters per second."],
    ["The chemical symbol for helium is He."],
    ["Mitosis is the process by which a cell divides to form two daughter cells."],
    ["The sun is composed primarily of hydrogen and helium."],
    ["The human genome contains approximately 3 billion base pairs."],
    ["The Great Wall of China is the longest man-made structure in the world."],
    ["The chemical symbol for nitrogen is N."],
    ["The freezing point of water is 0 degrees Celsius."],
    ["The Earth's crust is divided into tectonic plates."],
    ["The average lifespan of a red blood cell is about 120 days."],
    ["The chemical symbol for potassium is K."],
    ["The North Star, also known as Polaris, is located almost directly above the North Pole."],
    ["The average distance from the Earth to the Moon is about 384,400 kilometers."],
    ["The chemical symbol for calcium is Ca."],
    ["The human skeleton is divided into two parts: the axial skeleton and the appendicular skeleton."],
    ["The highest mountain on Earth is Mount Everest."],
    ["The chemical symbol for sulfur is S."],
    ["The Earth's atmosphere consists of five layers: troposphere, stratosphere, mesosphere, thermosphere, and exosphere."],
    ["The smallest unit of life is the cell."],
    ["The chemical symbol for iron is Fe."],
    ["The Earth's magnetic field is generated by the movement of molten iron in its outer core."],
    ["The largest organ in the human body is the skin."],
    ["The chemical symbol for sodium is Na."],
    ["The human eye can distinguish about 10 million different colors."],
    ["The longest river in the world is the Nile River."],
    ["The chemical symbol for chlorine is Cl."],
    ["The human brain uses approximately 20% of the body's energy."],
    ["The largest desert in the world is the Sahara Desert."],
    ["The chemical symbol for aluminum is Al."],
    ["The human ear can detect sounds in the frequency range of 20 Hz to 20,000 Hz."],
    ["The deepest part of the ocean is the Mariana Trench."],
    ["The chemical symbol for copper is Cu."],
    ["The most abundant gas in the Earth's atmosphere is nitrogen."],
    ["The longest bone in the human body is the femur."],
    ["The chemical symbol for mercury is Hg."],
    ["The Earth is about 4.5 billion years old."],
    ["The chemical symbol for zinc is Zn."],
    ["The second most abundant element in the Earth's crust is silicon."],
    ["The chemical symbol for lead is Pb."],
    ["The most common element in the universe is hydrogen."],
    ["The chemical symbol for tin is Sn."],
    ["The Earth's surface is about 71% water."],
    ["The chemical symbol for gold is Au."],
    ["The average human brain weighs about 1.4 kilograms."],
    ["The chemical symbol for silver is Ag."],
    ["The hottest planet in our solar system is Venus."],
    ["The chemical symbol for platinum is Pt."],
    ["The closest star to Earth is the Sun."],
    ["The chemical symbol for nickel is Ni."],
    ["The human body has five senses: sight, hearing, taste, touch, and smell."],
    ["The chemical symbol for silicon is Si."],
    ["The Earth's gravity is approximately 9.8 meters per second squared."],
    ["The chemical symbol for fluorine is F."],
    ["The human body has about 640 muscles."],
    ["The chemical symbol for phosphorus is P."],
    ["The Earth's circumference is approximately 40,075 kilometers."],
    ["The chemical symbol for magnesium is Mg."],
    ["The average pH of human blood is about 7.4."],
    ["The chemical symbol for barium is Ba."],
    ["The largest land animal is the African elephant."],
    ["The chemical symbol for cobalt is Co."],
    ["The tallest tree in the world is a coast redwood named Hyperion."],
    ["The chemical symbol for iodine is I."],
    ["The Great Barrier Reef is located in the Coral Sea."],
    ["The chemical symbol for lithium is Li."],
    ["The average human heart beats about 100,000 times a day."],
    ["The chemical symbol for boron is B."],
    ["The Earth's axial tilt is about 23.5 degrees."]
]

# Preprocess the data
x, y, vocab_size, dictionary, sequence_len = text_preprocess(Sentence_Data)

def build_transformer_model(vocab_size, max_len, d_model, num_heads, ff_dim, num_layers):
    inputs = Input(shape=(max_len,))
    embedding_layer = Embedding(vocab_size, d_model)(inputs)
    positional_encoding = add_positional_encoding(embedding_layer, max_len, d_model)
    
    x = positional_encoding
    for _ in range(num_layers):
        # Multi-head attention
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attention_output = Dropout(0.1)(attention_output)
        attention_output = LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed-forward network
        ffn_output = Dense(ff_dim, activation='relu')(attention_output)
        ffn_output = Dropout(0.1)(ffn_output)
        ffn_output = Dense(d_model)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
    
    outputs = Dense(vocab_size, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


max_len = sequence_len - 1  
d_model = 16
num_heads = 10
ff_dim = 2
num_layers = 12

model = build_transformer_model(vocab_size, max_len, d_model, num_heads, ff_dim, num_layers)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=300, batch_size=100, verbose=1)
model.summary()

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
