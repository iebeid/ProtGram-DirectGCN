# ==============================================================================
# MODULE: source/models/fnn/mlp.py
# PURPOSE: Defines a Multi-Layer Perceptron (MLP) for link prediction.
# VERSION: 2.1 (Updated Keras imports for TF 2.15+)
# AUTHOR: Islam Ebeid
# ==============================================================================

# --- DEFINITIVE FIX: Use the standalone tf_keras package for consistency ---
# The setup script explicitly installs `tf-keras`. To avoid namespace conflicts
# with the Keras bundled in TensorFlow, we will consistently use the `tf_keras` package.
from tf_keras.layers import InputLayer, Dense, Dropout
from tf_keras.models import Sequential
from tf_keras.optimizers import Adam
from tf_keras.regularizers import l2


class MLP:
    """A Multi-Layer Perceptron for link prediction."""

    @staticmethod
    def build(input_dim: int, config) -> Sequential:
        """
        Builds and compiles the MLP model.

        Args:
            input_dim: The dimension of the input layer.
            config: The configuration object with MLP parameters.

        Returns:
            A compiled Keras Sequential model.
        """
        model = Sequential([
            InputLayer(input_shape=(input_dim,)),

            Dense(
                config.EVAL_MLP_DENSE1_UNITS,
                activation='relu',
                kernel_regularizer=l2(config.EVAL_MLP_L2_REG)
            ),
            Dropout(config.EVAL_MLP_DROPOUT1_RATE),

            Dense(
                config.EVAL_MLP_DENSE2_UNITS,
                activation='relu',
                kernel_regularizer=l2(config.EVAL_MLP_L2_REG)
            ),
            Dropout(config.EVAL_MLP_DROPOUT2_RATE),

            Dense(1, activation='sigmoid')
        ])

        optimizer = Adam(learning_rate=config.EVAL_MLP_LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model