from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import SimpleRNN
import matplotlib.pyplot as plt
from tensorflow import *

max_features = 10000
maxlen = 500
batch_size = 32

