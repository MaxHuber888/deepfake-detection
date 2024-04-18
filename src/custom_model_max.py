from keras.layers import TimeDistributed, Dense, Dropout, LSTM
from keras import models
from keras.models import load_model



def get_cnn_rnn_hybrid(img_size=256):
    # Pretrained Model
    mesonet = load_model("saved_models\MESONET_RegularFrame.keras")

    # Create our final model
    model = models.Sequential()
    # Add the convnet
    model.add(TimeDistributed(mesonet))
    # Use LSTM
    model.add(LSTM(64))
    # Make a decision network
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model
