import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=3):
        """
        Args:
            input_dim (int): Dimensionality of the input features.
            output_dim (int): Dimensionality of the output features.
            hidden_dim (int): Number of features in the hidden state of the LSTM.
            num_layers (int): Number of stacked LSTM layers.
        """
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, output_dim)
        """
        out, _ = self.lstm(x)  # out shape: (batch_size, seq_length, hidden_dim)
        out = self.fc(out)     # projecting hidden states to output dimension
        return self.activation(out)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3):
        """
        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Number of features in the hidden state of the LSTM.
            num_layers (int): Number of stacked LSTM layers.
        """
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # The final hidden state will be a concatenation of forward and backward states.
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
        Returns:
            Tensor: A probability score of shape (batch_size, 1)
        """
        # Get output and hidden states from the bidirectional LSTM
        _, (h_n, _) = self.lstm(x)
        # h_n has shape: (num_layers*2, batch_size, hidden_dim)
        # Extract the last layer's forward and backward hidden states:

        h_forward = h_n[-2]  # Forward direction from the last layer
        h_backward = h_n[-1] # Backward direction from the last layer
        h = torch.cat((h_forward, h_backward), dim=1)  # shape: (batch_size, hidden_dim*2)
        out = self.fc(h)
        return self.activation(out)
