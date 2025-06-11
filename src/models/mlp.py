# ==============================================================================
# MODULE: models/mlp.py
# PURPOSE: Contains the definition for the Multi-Layer Perceptron (MLP) used
#          for link prediction in the evaluation pipeline.
# ==============================================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from typing import Dict, Any


def build_mlp_model(input_shape: int, mlp_params: Dict[str, Any], learning_rate: float) -> tf.keras.Model:
    """
    Builds and compiles the MLP model for link prediction.
    (Adapted from evaluater.py)

    Args:
        input_shape (int): The dimension of the input edge features.
        mlp_params (Dict[str, Any]): A dictionary containing the MLP architecture parameters.
        learning_rate (float): The learning rate for the Adam optimizer.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    model = Sequential([InputLayer(input_shape=(input_shape,)), Dense(mlp_params['dense1_units'], activation='relu', kernel_regularizer=l2(mlp_params['l2_reg'])), Dropout(mlp_params['dropout1_rate']),
                        Dense(mlp_params['dense2_units'], activation='relu', kernel_regularizer=l2(mlp_params['l2_reg'])), Dropout(mlp_params['dropout2_rate']), Dense(1, activation='sigmoid')])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model
