import keras
from keras.layers import Convolution2D, MaxPool2D, Dense, Flatten


class QModel:
    def __init__(self, input_shape, output_shape):
        self.model = self.build_model(input_shape, output_shape)
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build_model(self, input_shape, outputs):
        model = keras.Sequential()
        model.add(Convolution2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding='same', input_shape=input_shape,
                                activation='relu'))
        # model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
        # Second
        model.add(Convolution2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu'))
        # model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
        # Third
        model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        # First fully connected
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(outputs))
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse', metrics=['accuracy'])
        # print(model.summary())
        return model

    def save(self):
        self.model.save('supermodel')

    def clone(self):
        new_model = QModel(self.input_shape, self.output_shape)
        new_model.model = keras.models.clone_model(self.model)
        return new_model
