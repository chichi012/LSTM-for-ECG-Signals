import numpy as np

'''dimensionality of input,output feature spaces
number of timesteps in input sequence'''

timesteps = 100
input_features = 32
output_features = 64

#initial state is all-zero vector
inputs = np.random.random((timesteps, input_features))
state_t = np.zeros(output_features)

#create random weight matrices
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features))

updated_outputs = []

for i in inputs:
    output = np.tanh(np.dot(W, i) + np.dot(U, state_t) + b)    #combines input with the current state (previous output) to obtain current output

    updated_outputs.append(output)

    state_t = output  #stores output in a list

final_output_sequence = np.concatenate(updated_outputs,axis=0)       #final o/p is 2D tensor of shape (timesteps, output_features) ie timesteps x output_features
print(output)
print(len(final_output_sequence))