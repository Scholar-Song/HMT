import numpy as np
import torch
import matplotlib.pyplot as plt
import casual_conv_layer

class InformerTimeSeries(torch.nn.Module):
    """
    Time Series application of transformers based on paper

    causal_convolution_layer parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel

    nn.Transformer parameters:
        d_model: the size of the embedding vector (input)

    PositionalEncoding parameters:
        d_model: the size of the embedding vector (positional vector)
        dropout: the dropout to be used on the sum of positional+embedding vector

    """

    def __init__(self):
        super(InformerTimeSeries, self).__init__()
        self.input_embedding = casual_conv_layer.context_embedding(2, 128, 4)
        self.positional_embedding = torch.nn.Embedding(256, 128)
        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=3)

        self.fc1 = torch.nn.Linear(128, 1)

    def forward(self, x, y, attention_masks):
        # concatenate observed points and time covariate
        # (B*feature_size*n_time_points) -》 对应的xyz坐标
        z = torch.cat((y.unsqueeze(1), x.unsqueeze(1)), 1)  # cat 第二个参数是拼接方向，0是竖着拼，1是横着拼

        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        z_embedding = self.input_embedding(z).permute(2, 0, 1)  # permute 实现任意维度转置，0-x，1-y，2-z

        # get my positional embeddings (Batch size, sequence_len, embedding_size) -> need (sequence len,Batch size,embedding_size)
        positional_embeddings = self.positional_embedding(x.type(torch.long)).permute(1, 0, 2)

        input_embedding = z_embedding + positional_embeddings  # 叠加

        transformer_embedding = self.transformer_decoder(input_embedding, attention_masks)

        output = self.fc1(transformer_embedding.permute(1, 0, 2))

        return output
