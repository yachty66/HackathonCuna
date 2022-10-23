from keras.layers import Dense, Input, Flatten, concatenate
from keras.models import Model

inputSize = 8
inputLength = 10
outputSize = 3
outputLength = 10
l = 32

input_1 = Input(shape=(inputSize, ))
dense1 = Dense(l, activation = 'relu')(input_1)

# input_2 = Input(shape=(inputSize, ))
# dense1_2 = Dense(l, activation = 'relu')(input_2)

# concat = concatenate([dense1,dense1_2])

dense2 = Dense(l, activation = 'relu')(dense1)
dense3 = Dense(4*l, activation = 'relu')(dense2)
dense4 = Dense(16*l, activation = 'relu')(dense3)
dense5 = Dense(16*l, activation = 'relu')(dense4)
dense6 = Dense(4*l, activation = 'relu')(dense5)
dense7 = Dense(l, activation = 'relu')(dense6)
dense8 = Dense(3, activation = 'relu')(dense7)

output = Dense(outputSize, activation = 'linear')(dense8)

inputList = input_1
outputList = output

model = Model(inputList, outputList)

name = "struc_15"

model.save(f'code/model/model_{name}.h5')

print("model_saved")