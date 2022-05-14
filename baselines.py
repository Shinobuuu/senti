import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn
from deploy import D, VOCAB_SIZE, device

RNN_LAYERS = 2


def get_mask(sizes):
    mask = torch.arange(sizes.max().item())[None, :] >= sizes[:, None]
    mask = mask.to(device)
    return mask


class BaseModule(nn.Module):
    def __init__(self, dropout):
        super(BaseModule, self).__init__()
        self.name = ""
        self.relu = nn.ReLU()
        self.filter_sizes = [1, 2]
        self.filter_nums = int(D / 2)
        self.lstm_layers = RNN_LAYERS
        self.emb_encoder = nn.Embedding(VOCAB_SIZE, D, padding_idx=0)

    def embed(self, x):
        return self.emb_encoder(x)

    def to_rnn(self, rnn_in, sizes):
        rnn_in = rnn.pack_padded_sequence(rnn_in, sizes, batch_first=True)
        rnn_hidden, rnn_final = self.rnn_layer(rnn_in)
        rnn_hidden = rnn.pad_packed_sequence(rnn_hidden, batch_first=True)
        rnn_hidden = rnn_hidden[0]
        rnn_final = rnn_final[0].transpose(0, 1)
        rnn_final = rnn_final.reshape(-1, self.lstm_layers, D).transpose(0, 1)[-1]
        return rnn_hidden, rnn_final

    def to_cnn(self, cnn_in, max_size):
        cnn_in = cnn_in.unsqueeze(1)
        pooled_list = []
        for i, conv in enumerate(self.conv_list):
            mp = nn.MaxPool2d((max_size - self.filter_sizes[i] + 1, 1))
            cnn_hidden = self.relu(conv(cnn_in))
            pooled = mp(cnn_hidden).squeeze()
            pooled_list.append(pooled)
        cnn_out = torch.cat(pooled_list, dim=1)
        return cnn_out


class TextCNN(BaseModule):
    def __init__(self, dropout):
        super(TextCNN, self).__init__(dropout)
        self.name = "TextCNN"
        self.conv_list = nn.ModuleList([nn.Conv2d(1, self.filter_nums, (s, D))
                                        for s in self.filter_sizes])
        self.fc = nn.Sequential(
            # nn.Dropout(p=0.4),
            nn.Linear(D, int(D * 1.5)),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(int(D * 1.5), 2)
        )

    def forward(self, x, sizes):
        emb = self.embed(x)
        cnn_out = self.to_cnn(emb, sizes[0])
        out = self.fc(cnn_out)
        return out


class TextLSTM(BaseModule):
    def __init__(self, dropout):
        super(TextLSTM, self).__init__(dropout)
        self.name = "TextLSTM"
        self.rnn_layer = nn.LSTM(D, int(D / 2), num_layers=self.lstm_layers, dropout=dropout,
                                 bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(D, int(D * 1.5)),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(int(D * 1.5), 2)
        )

    def forward(self, x, sizes):
        emb = self.embed(x)
        _, lstm_final = self.to_rnn(emb, sizes)
        out = self.fc(lstm_final)
        return out


class TextGRU(BaseModule):
    def __init__(self, dropout):
        super(TextGRU, self).__init__(dropout)
        self.name = "TextGRU"
        self.rnn_layer = nn.GRU(D, int(D / 2), num_layers=self.lstm_layers, dropout=dropout,
                                bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(D, int(D * 1.5)),
                                nn.Dropout(p=dropout),
                                nn.ReLU(),
                                nn.Linear(int(D * 1.5), 2))

    def forward(self, x, sizes):
        emb = self.embed(x)
        _, lstm_final = self.to_rnn(emb, sizes)
        out = self.fc(lstm_final)
        return out


class TextLCNN(BaseModule):
    def __init__(self, dropout):
        super(TextLCNN, self).__init__(dropout)
        self.name = "TextLCNN"
        self.conv_list = nn.ModuleList([nn.Conv2d(1, self.filter_nums, (s, D))
                                        for s in self.filter_sizes])
        self.rnn_layer = nn.LSTM(D, int(D / 2), num_layers=self.lstm_layers,
                                 bidirectional=True, batch_first=True)
        self.att_layer = nn.MultiheadAttention(D, num_heads=6, dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(
            # nn.Dropout(p=0.4),
            nn.Linear(D, int(D * 1.5)),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(int(D * 1.5), 2)
        )

    def forward(self, x, sizes):
        mask = get_mask(sizes)
        emb = self.embed(x)
        lstm_hidden, _ = self.to_rnn(emb, sizes)
        att_out = self.att_layer(lstm_hidden, lstm_hidden, lstm_hidden, mask)[0]
        cnn_out = self.to_cnn(att_out, sizes[0])
        out = self.fc(cnn_out)
        return out
