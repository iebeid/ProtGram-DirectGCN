# ==============================================================================
# MODULE: models/mlp.py
# PURPOSE: Contains the definition for the Multi-Layer Perceptron (MLP) used
#          for link prediction in the evaluation pipeline.
# VERSION: 2.0 (Refactored into MLPModelBuilder class)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import Dict, Any

import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam  # Adam is used directly
from tensorflow.keras.regularizers import l2


class MLP:
    """
    A class responsible for building and compiling MLP models
    for link prediction.
    """

    def __init__(self, input_shape: int, mlp_params: Dict[str, Any], learning_rate: float):
        """
        Initializes the MLPModelBuilder.

        Args:
            input_shape (int): The dimension of the input edge features.
            mlp_params (Dict[str, Any]): A dictionary containing the MLP architecture parameters.
                                         Expected keys: 'dense1_units', 'dropout1_rate',
                                                        'dense2_units', 'dropout2_rate', 'l2_reg'.
            learning_rate (float): The learning rate for the Adam optimizer.
        """
        self.input_shape = input_shape
        self.mlp_params = mlp_params
        self.learning_rate = learning_rate

    def build(self) -> tf.keras.Model:
        """
        Builds and compiles the MLP model.

        Returns:
            tf.keras.Model: The compiled Keras model.
        """
        model = Sequential([
            InputLayer(input_shape=(self.input_shape,)),
            Dense(
                self.mlp_params['dense1_units'],
                activation='relu',
                kernel_regularizer=l2(self.mlp_params['l2_reg'])
            ),
            Dropout(self.mlp_params['dropout1_rate']),
            Dense(
                self.mlp_params['dense2_units'],
                activation='relu',
                kernel_regularizer=l2(self.mlp_params['l2_reg'])
            ),
            Dropout(self.mlp_params['dropout2_rate']),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        return model