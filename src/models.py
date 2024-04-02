from keras import layers, models
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.src.layers import InputLayer


def get_mesonet_model(img_size=256):
    model = models.Sequential()
    model.add(InputLayer(shape=(img_size, img_size, 3)))
    model.add(layers.Conv2D(8, (3, 3), padding='same', activation='relu', data_format='channels_last'))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(8, (5, 5), padding='same', activation='relu', data_format='channels_last'))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(16, (5, 5), padding='same', activation='relu', data_format='channels_last'))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(16, (5, 5), padding='same', activation='relu', data_format='channels_last'))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.MaxPooling2D(pool_size=(4, 4)))

    model.add(layers.Reshape((1024,)))
    # model.add(layers.Flatten())

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='linear', kernel_initializer='random_uniform', bias_initializer='zeros'))

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros'))

    model.summary()

    return model


def get_mouthnet_model(img_size=256):
    base_model = InceptionResNetV2(include_top=False, weights=None, input_shape=(img_size, img_size, 3))

    model = models.Sequential()
    model.add(InputLayer(shape=(img_size, img_size, 3)))
    model.add(base_model)

    model.add(layers.Reshape((55296,)))
    # model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros'))

    model.summary()

    return model


def get_cnn_rnn_hybrid(img_size=256):
    feature_extractor = InceptionV3(weights="imagenet", include_top=False, pooling="avg", input_shape=(img_size, img_size, 3))

    model = models.Sequential()
    model.add(InputLayer(shape=(img_size, img_size, 3)))
    model.add(feature_extractor)
    model.add(layers.Dense(128, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros'))
    #model.add(layers.Reshape((img_size*img_size, 3)))
    #model.add(layers.GRU(16, return_sequences=True))
    #model.add(layers.GRU(8))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(8, activation="relu"))
    model.add(layers.Dense(1, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros'))

    model.summary()

    return model