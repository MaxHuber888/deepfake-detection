from keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D, TimeDistributed, GRU, Dense, Dropout
from keras.src.layers import InputLayer
from keras import layers, models


def get_cnn_rnn_hybrid(img_size=256):
    shape = (img_size, img_size, 3)
    # Create our convnet with (112, 112, 3) input shape
    convnet = build_convnet()

    # then create our final model
    model = models.Sequential()
    # add the convnet
    model.add(TimeDistributed(convnet))
    # here, you can also use GRU or LSTM
    model.add(GRU(64))
    # and finally, we make a decision network
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


def build_convnet(img_size=256):
    shape = (img_size, img_size, 3)
    momentum = .9
    model = models.Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=shape,
                     padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    # flatten...
    model.add(GlobalMaxPool2D())
    return model
