import torch
import casual_conv_layer


class LSTM_Time_Series(torch.nn.Module):
    def __init__(self, input_size=2, embedding_size=128, kernel_width=4, hidden_size=512):
        super(LSTM_Time_Series, self).__init__()

        self.input_embedding = casual_conv_layer.context_embedding(input_size, embedding_size, kernel_width)

        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, batch_first=True)

        self.fc1 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, y):
        """
        x: the time covariate
        y: the observed target
        """
        # concatenate observed points and time covariate
        # (B,input size + covariate size,sequence length)
        z_obs = torch.cat((y.unsqueeze(1), x.unsqueeze(1)), 1)

        # input_embedding returns shape (B,embedding size,sequence length)
        z_obs_embedding = self.input_embedding(z_obs)

        # permute axes (B,sequence length, embedding size)
        z_obs_embedding = self.input_embedding(z_obs).permute(0, 2, 1)

        # all hidden states from lstm
        # (B,sequence length,num_directions * hidden size)
        lstm_out, _ = self.lstm(z_obs_embedding)

        # input to nn.Linear: (N,*,Hin)
        # output (N,*,Hout)
        return self.fc1(lstm_out)