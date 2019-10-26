from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000
max_len = 500
batch_size = 32

print('loading data....')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words = max_features)
