"""
model_init.py
-------------
Defines the LSTMTimeSeriesAutoencoder architecture.

IMPORTANT CONTRACT:
This module ONLY defines and returns a compiled model.
It does NOT save or load weights.

Trainer saves weights.
Calibration and Export load weights.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed
from tensorflow.keras.models import Model

from config import logging_config

logger = logging_config.get_logger(__name__)


# ---------------------------------------------------------------------
# Model Definition (Copied faithfully from notebook)
# ---------------------------------------------------------------------

class LSTMTimeSeriesAutoencoder:
    def __init__(self, sequence_length, num_dimensions, dim_weights):
        self.sequence_length = sequence_length
        self.num_dimensions = num_dimensions
        self.dim_weights = dim_weights
        self.model = self.build_model()

    def build_model(self):
        # Define input shape
        input_shape = (self.sequence_length, self.num_dimensions)

        # Encoder
        encoder_inputs = Input(shape=input_shape)
        encoder = LSTM(128, return_sequences=True)(encoder_inputs)
        encoder = LSTM(64, return_sequences=True)(encoder)
        encoder = LSTM(32, return_sequences=True)(encoder)
        encoder = LSTM(16, return_sequences=True)(encoder)
        encoder = LSTM(8, return_sequences=False)(encoder)

        # Repeat the latent vector for each time step
        decoder_inputs = RepeatVector(self.sequence_length)(encoder)

        # Decoder
        decoder = LSTM(8, return_sequences=True)(decoder_inputs)
        decoder = LSTM(16, return_sequences=True)(decoder)
        decoder = LSTM(32, return_sequences=True)(decoder)
        decoder = LSTM(64, return_sequences=True)(decoder)
        decoder = LSTM(128, return_sequences=True)(decoder)

        # Output layer
        output = TimeDistributed(Dense(self.num_dimensions))(decoder)

        # Define the model
        model = Model(inputs=encoder_inputs, outputs=output)

        return model

    def custom_loss(self, y_true, y_pred):
        # Define weights for each dimension
        weights = tf.constant(self.dim_weights, dtype=tf.float32)

        # Calculate mean squared error (MSE) for each dimension
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=0)

        # Multiply MSE by weights and sum across dimensions
        weighted_loss = tf.reduce_sum(mse * weights)

        return weighted_loss

    def compile_model(self):
        self.model.compile(optimizer='adam', loss=self.custom_loss)

    def summary(self):
        self.model.summary()

    def train(self, x_train, epochs, batch_size, callbacks):
        history = self.model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1,  callbacks=callbacks, shuffle=False)
        return history

    def predict(self, x):
        return self.model.predict(x)

    def save_weights(self, filepath):
        # Save model weights
        self.model.save_weights(filepath)

    def load_weights(self, filepath):
        # Load model weights
        self.model.load_weights(filepath)


# ---------------------------------------------------------------------
# Public Factory Function
# ---------------------------------------------------------------------

def initialize_model(sequence_length, num_features, dim_weights):
    """
    Build and compile the model.

    Returns
    -------
    tf.keras.Model
        Compiled model ready for training or inference
    """

    logger.info(
        "Initializing LSTM Autoencoder | seq_len=%d | features=%d | dim_weights=%s",
        sequence_length,
        num_features, 
        dim_weights
    )

    model = LSTMTimeSeriesAutoencoder(sequence_length, num_features, dim_weights)

    return model
