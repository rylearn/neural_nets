
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

def split_data(x_train, y_train, validation_fraction = 0.3):
	num_examples = len(x_train)
	size_validation = int(validation_fraction * num_examples)
	random_indices = np.random.choice(num_examples, size_validation)
	train_indices = [i for i in range(num_examples) if i not in random_indices]
	x_validation = x_train[random_indices]
	y_validation = y_train[random_indices]
	x_train = x_train[train_indices]
	y_train = y_train[train_indices]
	return x_train, y_train, x_validation, y_validation
	
num_classes = 10
img_rows, img_cols = 28, 28 	# input image dimensions

batch_size = 128
epochs = 20

dropout_amount = 0.3
hidden_nodes = 128

conv1_filters = 32
conv1_dims = (2, 2)

conv2_filters = 64
conv2_dims = (3, 3)

maxpool = (2, 2)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train, y_train, x_validation, y_validation = split_data(x_train, y_train)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_validation.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_validation = keras.utils.to_categorical(y_validation, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(conv1_filters, kernel_size = conv1_dims,
                activation = 'relu',
                input_shape = input_shape))
model.add(Conv2D(conv2_filters, kernel_size = conv2_dims, 
				activation = 'relu'))
model.add(MaxPooling2D(pool_size = maxpool))
model.add(Dropout(dropout_amount))
model.add(Flatten())
model.add(Dense(hidden_nodes, activation = 'relu'))
model.add(Dropout(dropout_amount))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])

model.fit(x_train, y_train,
          batch_size = batch_size,
          epochs = epochs,
          verbose = True,
          validation_data = (x_validation, y_validation))
score = model.evaluate(x_test, y_test, verbose = False)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
