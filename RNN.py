import torch
import casual_conv_layer


class RNN_series(torch.nn.Module):
    def __init__(self, input_size=2, embedding_size=128, kernel_width=4, hidden_size=512):
        super(RNN_series, self).__init__()
        self.input_embedding = casual_conv_layer.context_embedding(input_size, embedding_size, kernel_width)

        self.rnn = torch.nn.RNN(embedding_size, hidden_size, batch_first=True)

        self.fc1 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, y):
        """
        x: the time covariate
        y: the observed target
        """
        # concatenate observed points and time covariate
        # (B,input size + covariate size,sequence length)
        z_obs = torch.cat((y.unsqueeze(1), x.unsqueeze(1)), 1)
        z_obs_embedding = self.input_embedding(z_obs)

        # permute axes (B,sequence length, embedding size)
        z_obs_embedding = self.input_embedding(z_obs).permute(0, 2, 1)

        RNN_out, _ = self.rnn(z_obs_embedding)

        # input to nn.Linear: (N,*,Hin)
        # output (N,*,Hout)
        return self.fc1(RNN_out)
