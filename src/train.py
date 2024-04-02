import os

from keras.optimizers import Adam, SGD
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def prep_and_train_model(model, optim, train_data, test_data, epochs=1, batch_size=10, model_name="temp"):
    # Initialize optimizer
    if optim == "adam":
        optimizer = Adam()
    elif optim == "sgd":
        lr = 0.1
        optimizer = SGD(learning_rate=lr, momentum=0.9, weight_decay=lr / epochs)
    else:
        optimizer = Adam()

    # Compile the model for training
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    # loss="sparse_categorical_crossentropy"

    # Saves the model whenever a new max val accuracy is reached
    model_checkpoint_callback = ModelCheckpoint(
        filepath=f"saved_models/{model_name}.keras",
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Train the model
    history = model.fit(
        train_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=test_data,
        shuffle=True,
        callbacks=[model_checkpoint_callback]
    )

    return history, model


def save_history(history, model_name):
    hist_df = pd.DataFrame(history.history)
    # If previous history exists, concatenate histories
    if os.path.exists("saved_models/" + model_name + "_history.csv"):
        previous_hist_df = pd.read_csv("saved_models/" + model_name + "_history.csv")
        hist_df = pd.concat([previous_hist_df, hist_df], axis=0, ignore_index=True)
    hist_df.to_csv("saved_models/" + model_name + "_history.csv")


def load_model_from_path(model_path):

    print(f"Model loaded from {model_path}.")
    return model
