from keras import layers, models
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.src.layers import InputLayer


def get_mesonet_model(img_size=256):
    model = models.Sequential()
    model.add(InputLayer(shape=(img_size, img_size, 3)))
    model.add(
        layers.Conv2D(
            8, (3, 3), padding="same", activation="relu", data_format="channels_last"
        )
    )
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.MaxPooling2D())

    model.add(
        layers.Conv2D(
            8, (5, 5), padding="same", activation="relu", data_format="channels_last"
        )
    )
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.MaxPooling2D())

    model.add(
        layers.Conv2D(
            16, (5, 5), padding="same", activation="relu", data_format="channels_last"
        )
    )
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.MaxPooling2D())

    model.add(
        layers.Conv2D(
            16, (5, 5), padding="same", activation="relu", data_format="channels_last"
        )
    )
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.MaxPooling2D(pool_size=(4, 4)))

    model.add(layers.Reshape((1024,)))

    model.add(layers.Dropout(0.5))
    model.add(
        layers.Dense(
            16,
            activation="linear",
            kernel_initializer="random_uniform",
            bias_initializer="zeros",
        )
    )

    model.add(layers.Dropout(0.5))
    model.add(
        layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer="random_uniform",
            bias_initializer="zeros",
        )
    )

    return model


def get_mouthnet_model(img_size=256):
    base_model = InceptionResNetV2(
        include_top=False, weights=None, input_shape=(img_size, img_size, 3)
    )

    model = models.Sequential()
    model.add(InputLayer(shape=(img_size, img_size, 3)))
    model.add(base_model)

    model.add(layers.Reshape((55296,)))

    model.add(
        layers.Dense(
            128,
            activation="softmax",
            kernel_initializer="random_uniform",
            bias_initializer="zeros",
        )
    )
    model.add(layers.Dropout(0.5))
    model.add(
        layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer="random_uniform",
            bias_initializer="zeros",
        )
    )

    return model


def get_optical_model(img_size=256, num_frames=100):
    model = models.Sequential(
        [
            layers.TimeDistributed(
                layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
                input_shape=(
                    num_frames,
                    img_size,
                    img_size,
                    3,
                ),
            ),
            layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
            layers.TimeDistributed(
                layers.Conv2D(16, (3, 3), activation="relu", padding="same")
            ),
            layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
            layers.TimeDistributed(layers.Flatten()),
            layers.Conv1D(16, 3, activation="relu", padding="same"),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model
