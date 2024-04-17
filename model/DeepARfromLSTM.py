import torch
import casual_conv_layer


class DeepAR_Time_Series(torch.nn.Module):
    def __init__(self, input_size=2, embedding_size=2, kernel_width=4, hidden_size=128,output_size=1):
        super(DeepAR_Time_Series, self).__init__()
        self.output_dim = output_size

        self.input_embedding = casual_conv_layer.context_embedding(input_size, embedding_size, kernel_width)
        self.encoder = torch.nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.relu = torch.nn.ReLU()
        self.decoder = torch.nn.GRU(hidden_size+hidden_size, hidden_size, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size, output_size)

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
        inputs,(hidden,cell)=self.encoder(z_obs_embedding)
        inputs = self.relu(inputs)
        hidden = hidden.repeat(inputs.size(0), inputs.size(1), 1)
        inputs_hidden = torch.cat([inputs, hidden[:, 0:inputs.size(1), :]], dim=-1)

        decoder_outputs, _ = self.decoder(inputs_hidden)

        out= self.fc1(decoder_outputs)
        # loc = out[:, :, :self.output_dim]
        # scale = torch.exp(out[:, :, self.output_dim:])
        # weight = torch.nn.Softmax(dim=-1)(out[:, :, self.output_dim:])
        # return loc, scale, weight


        # all hidden states from lstm
        # (B,sequence length,num_directions * hidden size)
        #deeparout, _ = self.decoder(z_encoder)

        # input to nn.Linear: (N,*,Hin)
        # output (N,*,Hout)
        return out