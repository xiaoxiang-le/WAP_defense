from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Embedding, GRU
from tensorflow.keras.models import Sequential

from artifacts import atomic_keras_save
from evaluate import evaluate_predictions


MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "gru.keras"


class GRUModel:
    def __init__(
        self,
        train_data,
        train_labels,
        validation_data,
        validation_labels,
        test_data,
        test_labels,
        vocab_size,
        sequence_length,
        epochs=3,
    ):
        self.train_data = train_data
        self.train_labels = np.asarray(train_labels)
        self.validation_data = validation_data
        self.validation_labels = np.asarray(validation_labels)
        self.test_data = test_data
        self.test_labels = np.asarray(test_labels)
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.epochs = epochs

    def train(self):
        model = Sequential(
            [
                Embedding(
                    input_dim=self.vocab_size,
                    output_dim=128,
                    input_length=self.sequence_length,
                    mask_zero=True,
                ),
                GRU(96, dropout=0.2),
                BatchNormalization(),
                Dropout(0.3),
                Dense(3, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print("[GRU] 开始训练")
        model.fit(
            self.train_data,
            self.train_labels,
            validation_data=(self.validation_data, self.validation_labels),
            batch_size=128,
            epochs=self.epochs,
            callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
            verbose=2,
        )
        atomic_keras_save(model, MODEL_PATH)
        predictions = np.argmax(model.predict(self.test_data, verbose=0), axis=1)
        return evaluate_predictions("GRU", self.test_labels, predictions)
