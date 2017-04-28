'''Train a Bidirectional LSTM on the IMDB sentiment classification task.
Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

# Python 2 -> Compatibility
from __future__ import print_function
# Numpy!
import numpy as np

# looks like uses numpy to generate fancy arrays, and generates some n-gram models
from keras.preprocessing import sequence
# The Sequential model is a linear stack of neural network
from keras.models import Sequential
# These are just layers to be stacked up on the network
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
# Obviousliy imports imbdb stuff
from keras.datasets import imdb

# what are features???
max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
# What is a batch???
batch_size = 32

print('Loading data...')

#Not sure what x and y are, but puts x and y data into training and test data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print("Pad sequences (samples x time)")
# Makes arrays of shape specified below
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Making a basic, linear model
model = Sequential()
# Adding random layers so the machine can guess stuff???
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
# Turns all those layers into one, holdable thing that can be used with backpropagation to generate a model
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
# Tunes the weights of the layers to produce the most accurate model possible
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=4,
          validation_data=[x_test, y_test])


# Waht do with model???
# How speed up???
# How do layers work???
