from keras.optimizers import Adam, SGD
import pickle
from keras.models import load_model


def prep_and_train_model(model, optim, train_data, test_data, epochs=1, batch_size=10):
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

    # TODO: Add parallel workers

    # Train the model
    history = model.fit(
        train_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=test_data,
        workers=4,
        use_multiprocessing=True,
        shuffle=True,
    )

    return history, model


def save_model(history, model, out_path):
    model.save(out_path)
    history_path = out_path + "_history.pkl"
    with open(history_path, "w") as f:
        json.dump(history.history, f)

    print(
        f"Model and training history are saved at {out_path} and {history_path} respectively."
    )


def load_model(model_path):
    model = load_model(model_path)
    print(f"Model loaded from {model_path}.")
    return model
