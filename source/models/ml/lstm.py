# ==============================================================================
# MODULE: models/ml/lstm.py
# PURPOSE: Defines the PyTorch LSTM model for next-character prediction.
# VERSION: 1.1 (Clarified docstrings and method purposes)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    A standard LSTM model designed to predict the next amino acid in a sequence.
    The final hidden state of the LSTM is used as the protein's embedding.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int):
        """
        Initializes the LSTM model layers.

        Args:
            vocab_size (int): The number of unique characters in the vocabulary.
            embedding_dim (int): The dimensionality of the character embeddings.
            hidden_dim (int): The number of features in the LSTM's hidden state.
            num_layers (int): The number of recurrent layers in the LSTM.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the next-character prediction task during training.

        Args:
            x (torch.Tensor): A batch of input sequences.

        Returns:
            torch.Tensor: The output logits for predicting the next character.
        """
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # For next-character prediction, we only need the output of the last time step.
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return out

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gets the final protein embedding after processing the full sequence.
        This is used for inference after the model has been trained.

        Args:
            x (torch.Tensor): A single, full-length protein sequence.

        Returns:
            torch.Tensor: The final hidden state of the last LSTM layer, which
                          serves as the embedding for the protein.
        """
        with torch.no_grad():
            x = self.embedding(x)
            # The second element of the LSTM output tuple is (hidden_state, cell_state)
            _, (hidden, _) = self.lstm(x)
            # We return the hidden state of the last layer.
            # Shape: (num_layers, batch_size, hidden_dim) -> we take the last layer.
            return hidden[-1]