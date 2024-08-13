import torch
import torch.nn as nn

class GeneExpressionEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(GeneExpressionEncoder, self).__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_encoder_layers
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        src = self.input_layer(src)  # Embed the input features
        src = self.dropout(src)
        encoded_output = self.transformer_encoder(src)

        print(encoded_output.size())
        return encoded_output



class GeneExpressionDecoder(nn.Module):
    def __init__(self, d_model, output_dim=1, num_decoder_layers=1, dim_feedforward=2048, dropout=0.1):
        super(GeneExpressionDecoder, self).__init__()
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=1, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_decoder_layers
        )
        self.output_layer = nn.Linear(d_model, output_dim)  # Output a single value for regression (age)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, memory):
        decoded_output = self.transformer_decoder(tgt, memory)
        output = self.output_layer(decoded_output[-1])  # Use the last output to predict the age
        return output


class AgePredictionTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(AgePredictionTransformer, self).__init__()
        self.encoder = GeneExpressionEncoder(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.decoder = GeneExpressionDecoder(d_model, output_dim=1, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output



# 示例使用
input_dim = 10  # 每个人有 10 个基因特征
d_model = 64  # Transformer 模型的隐藏层维度
nhead = 4  # 多头注意力的头数
num_encoder_layers = 3  # 编码器的层数
num_decoder_layers = 3  # 解码器的层数
dim_feedforward = 256  # 前馈神经网络的维度
dropout = 0.1  # Dropout 的概率

model = AgePredictionTransformer(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

# 随机生成一些示例数据
src = torch.rand((5, 32, input_dim))  # 输入序列：5 个时间步，32 个样本，每个样本 10 个基因特征
tgt = torch.rand((5, 32, d_model))  # 解码器输入：同样 5 个时间步，32 个样本，d_model 维度的隐向量

output = model(src, tgt)
print("Predicted ages:", output)
