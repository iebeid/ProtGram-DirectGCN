# ==============================================================================
# MODULE: models/rnn/lstm.py
# PURPOSE: Defines the PyTorch LSTM model for next-character prediction.
# VERSION: 2.0 (Refactored to a single, efficient forward pass)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

from typing import Tuple
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    A standard LSTM model designed to predict the next amino acid in a sequence.
    The final hidden state of the LSTM is used as the protein's embedding.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout_rate: float = 0.5):
        """
        Initializes the LSTM model layers.

        Args:
            vocab_size (int): The number of unique characters in the vocabulary.
            embedding_dim (int): The dimensionality of the character embeddings.
            hidden_dim (int): The number of features in the LSTM's hidden state.
            num_layers (int): The number of recurrent layers in the LSTM.
            dropout_rate (float): Dropout probability. Applied after the LSTM layer.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # The built-in dropout is applied between layers of a multi-layer LSTM.
        lstm_dropout = dropout_rate if num_layers > 1 else 0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the forward pass. This single method provides both the logits for
        training and the final sequence embedding for inference, which is more
        efficient than having two separate methods.

        Args:
            x (torch.Tensor): A batch of input sequences.

        Returns:
            A tuple containing:
            - logits (torch.Tensor): The output logits for the next-character prediction task.
            - sequence_embedding (torch.Tensor): The final hidden state of the LSTM,
                                                 which serves as the embedding for the sequence.
        """
        x = self.embedding(x)
        # The lstm layer returns all hidden states (lstm_out) and the final hidden/cell states
        lstm_out, (hidden, _) = self.lstm(x)

        # --- 1. Logits for Training ---
        # We take the output of the last time step for next-character prediction.
        last_time_step_out = lstm_out[:, -1, :]
        # Apply dropout before the final linear layer for regularization.
        last_time_step_out = self.dropout(last_time_step_out)
        logits = self.fc(last_time_step_out)

        # --- 2. Embedding for Inference ---
        # The final hidden state of the last layer serves as the sequence embedding.
        sequence_embedding = hidden[-1]

        return logits, sequence_embedding