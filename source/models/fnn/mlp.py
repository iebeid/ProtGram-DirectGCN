# ==============================================================================
# MODULE: models/fnn/mlp.py
# PURPOSE: Contains the definition for the Multi-Layer Perceptron (MLP) used
#          for link prediction in the evaluation trainers.
# VERSION: 2.1 (Corrected docstrings and path)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

from typing import Dict, Any

import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class MLP:
    """
    A class responsible for building and compiling MLP models
    for link prediction.
    """

    def __init__(self, input_shape: int, mlp_params: Dict[str, Any], learning_rate: float):
        """
        Initializes the MLP.

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
            # --- DEFINITIVE FIX: Use .get() for robust parameter access ---
            # This prevents KeyErrors if a parameter is missing from the config
            # and makes the model builder more self-contained and resilient.
            Dense(
                self.mlp_params.get('dense1_units', 128),
                activation='relu',
                kernel_regularizer=l2(self.mlp_params.get('l2_reg', 1e-5))
            ),
            Dropout(self.mlp_params.get('dropout1_rate', 0.5)),
            Dense(
                self.mlp_params.get('dense2_units', 64),
                activation='relu',
                kernel_regularizer=l2(self.mlp_params.get('l2_reg', 1e-5))
            ),
            Dropout(self.mlp_params.get('dropout2_rate', 0.5)),
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