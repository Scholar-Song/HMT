import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class DeepAR(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(DeepAR, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoder = nn.GRU(input_size, hidden_size)
        self.decoder = nn.GRU(input_size + hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        # y: [batch_size, seq_len, output_size]
        x= x.unsqueeze(-1)
        z_obs = x.permute(1, 0, 2)

        encoder_input = x[:, :, :self.input_size]
        encoder_output, encoder_hidden = self.encoder(encoder_input.permute(1, 0, 2))

        decoder_input = encoder_input[-1, :, :]
        decoder_hidden = encoder_hidden

        output = []
        for i in range(z_obs.size(0)):
            decoder_input = torch.cat([decoder_input, decoder_hidden[-1, :, :]], dim=-1)
            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(0), decoder_hidden)
            y_pred = self.fc(decoder_output.squeeze(0))
            output.append(y_pred)

        output = torch.stack(output).permute(1, 0, 2)
        return output