import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn
from baselines import BaseModule
from deploy import D, VOCAB_SIZE, device

MAX_LENGTH = 128
DEFAULT_LAYERS = 4
SENTI_LAYERS = 1
CNN_DROPOUT = 0.4
a = 0.5
n_heads = 8


def reverse(texts, sizes):
    tgt = [torch.flip(texts[i][:sizes[i]], dims=[0]) for i in range(len(sizes))]
    tgt = rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
    return tgt


def clear_feature(clear_in, sizes):
    clear_in = rnn.pack_padded_sequence(clear_in, sizes, batch_first=True)
    clear_out = rnn.pad_packed_sequence(clear_in, batch_first=True)[0]
    return clear_out


def get_mask(sizes):
    mask = torch.arange(sizes.max().item())[None, :] >= sizes[:, None]
    mask = mask.to(device)
    return mask


class PositionalEncoder(nn.Module):
    def __init__(self, d_hidden=D, max_len=MAX_LENGTH):
        super(PositionalEncoder, self).__init__()
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, d_hidden))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(0, d_hidden, 2, dtype=torch.float32) / d_hidden)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        self.requires_grad_(False)

    def forward(self, x):
        x = x + self.P[:, :x.shape[1], :].to(device)
        return x


class FeatureCater(nn.Module):
    def __init__(self):
        super(FeatureCater, self).__init__()
        self.P = nn.Parameter(torch.randn(D, requires_grad=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, s, r):
        weights = self.sigmoid(self.P).to(device)
        out = torch.mul(s, weights) + torch.mul(r, 1 - weights)
        return out


class TransformerBased(BaseModule):
    def __init__(self, dropout, num_layers=DEFAULT_LAYERS):
        super(TransformerBased, self).__init__(dropout)
        self.num_layers = num_layers
        self.pos_encoder = PositionalEncoder()

    def encode(self, x):
        word_embedding = self.emb_encoder(x)
        pos_embedding = self.pos_encoder(word_embedding)
        return pos_embedding


class TE2CNN(TransformerBased):
    def __init__(self, dropout, num_layers=DEFAULT_LAYERS):
        super(TE2CNN, self).__init__(dropout, num_layers)
        self.name = "TE{:.0f}-CNN".format(self.num_layers)
        self.conv_list = nn.ModuleList([nn.Conv2d(1, self.filter_nums, (s, D))
                                        for s in self.filter_sizes])
        self.te = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, nhead=n_heads, batch_first=True, dropout=dropout),
                                        num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Dropout(p=CNN_DROPOUT),
            nn.Linear(D, int(D * 1.5)),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(int(D * 1.5), 2)
        )

    def forward(self, x, sizes):
        encoding = self.encode(x)
        mask = get_mask(sizes)
        te_out = self.te(encoding, src_key_padding_mask=mask)
        te_out = clear_feature(te_out, sizes)
        cnn_out = self.to_cnn(te_out, sizes[0])
        out = self.fc(cnn_out)
        return out


class ABiTE2CNN(TransformerBased):
    def __init__(self, dropout, num_layers=DEFAULT_LAYERS):
        super(ABiTE2CNN, self).__init__(dropout, num_layers)
        self.name = "ABiTE{:.0f}-CNN".format(self.num_layers)
        self.conv_list = nn.ModuleList([nn.Conv2d(1, self.filter_nums, (s, D))
                                        for s in self.filter_sizes])
        self.te = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, nhead=n_heads, batch_first=True, dropout=dropout),
                                        num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Dropout(p=CNN_DROPOUT),
            nn.Linear(D, int(D * 1.5)),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(int(D * 1.5), 2)
        )

    def forward(self, x, sizes):
        encoding = self.encode(x)
        te_s, te_r = self.get_bite_out(encoding, sizes)
        te_s = clear_feature(te_s, sizes)
        te_r = clear_feature(te_r, sizes)
        cnn_s = self.to_cnn(te_s, sizes[0])
        cnn_r = self.to_cnn(te_r, sizes[0])
        cnn_out = 0.5 * cnn_s + 0.5 * cnn_r
        out = self.fc(cnn_out)
        return out

    def get_bite_out(self, te_in, sizes):
        te_re = reverse(te_in, sizes)
        mask = get_mask(sizes)
        src_out = self.te(te_in, src_key_padding_mask=mask)
        rev_out = self.te(te_re, src_key_padding_mask=mask)
        return src_out, rev_out


class WBiTE2CNN(ABiTE2CNN):
    def __init__(self, dropout, num_layers=DEFAULT_LAYERS):
        super(WBiTE2CNN, self).__init__(dropout, num_layers)
        self.name = "WBiTE{:.0f}-CNN".format(self.num_layers)
        self.cater = FeatureCater()
        self.conv_list = nn.ModuleList([nn.Conv2d(1, self.filter_nums, (s, D))
                                        for s in self.filter_sizes])
        self.te = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, nhead=n_heads, batch_first=True, dropout=dropout),
                                        num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Dropout(p=CNN_DROPOUT),
            nn.Linear(D, int(D * 1.5)),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(int(D * 1.5), 2)
        )

    def forward(self, x, sizes):
        encoding = self.encode(x)
        te_s, te_r = self.get_bite_out(encoding, sizes)
        te_s = clear_feature(te_s, sizes)
        te_r = clear_feature(te_r, sizes)
        cnn_s = self.to_cnn(te_s, sizes[0])
        cnn_r = self.to_cnn(te_r, sizes[0])
        cnn_out = self.cater(cnn_s, cnn_r)
        out = self.fc(cnn_out)
        return out


class MBiTE2CNN(ABiTE2CNN):
    def __init__(self, dropout, num_layers=DEFAULT_LAYERS):
        super(MBiTE2CNN, self).__init__(dropout, num_layers)
        self.name = "MBiTE{:.0f}-CNN".format(self.num_layers)
        self.conv_list = nn.ModuleList([nn.Conv2d(1, self.filter_nums, (s, D))
                                        for s in self.filter_sizes])
        self.te = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, nhead=n_heads, batch_first=True, dropout=dropout),
                                        num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Dropout(p=CNN_DROPOUT),
            nn.Linear(D, int(D * 1.5)),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(int(D * 1.5), 2)
        )

    def forward(self, x, sizes):
        encoding = self.encode(x)
        te_s, te_r = self.get_bite_out(encoding, sizes)
        te_s = clear_feature(te_s, sizes)
        te_r = clear_feature(te_r, sizes)
        cnn_s = self.to_cnn(te_s, sizes[0]).unsqueeze(1)
        cnn_r = self.to_cnn(te_r, sizes[0]).unsqueeze(1)
        cnn_out = torch.cat([cnn_s, cnn_r], dim=1).max(1).values
        out = self.fc(cnn_out)
        return out


class DualTE2CNN(TransformerBased):
    def __init__(self, dropout, num_layers=DEFAULT_LAYERS):
        super(DualTE2CNN, self).__init__(dropout, num_layers)
        self.name = "Dual{:.0f}TE{:.0f}-CNN".format(self.num_layers, SENTI_LAYERS)
        self.tes = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            D, nhead=n_heads, batch_first=True, dropout=dropout), num_layers=SENTI_LAYERS)
        self.conv_list = nn.ModuleList([nn.Conv2d(1, self.filter_nums, (s, D))
                                        for s in self.filter_sizes])
        self.te = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, nhead=n_heads, batch_first=True, dropout=dropout),
                                        num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Dropout(p=CNN_DROPOUT),
            nn.Linear(D, int(D * 1.5)),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(int(D * 1.5), 2)
        )

    def forward(self, x, sizes):
        encoding = self.encode(x)
        mask = get_mask(sizes)
        te_out = self.te(encoding, src_key_padding_mask=mask)
        tes_out = self.tes(te_out, src_key_padding_mask=mask)
        tes_out = clear_feature(tes_out, sizes)
        cnn_out = self.to_cnn(tes_out, sizes[0])
        out = self.fc(cnn_out)
        return out


class TS2S(TransformerBased):
    def __init__(self, dropout, num_layers):
        super(TS2S, self).__init__(dropout, num_layers)
        self.name = "TS2S{:.0f}".format(self.num_layers)
        self.te = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, nhead=n_heads, batch_first=True, dropout=dropout),
                                        num_layers=num_layers)
        self.td = nn.TransformerDecoder(nn.TransformerDecoderLayer(D, nhead=n_heads, batch_first=True, dropout=dropout),
                                        num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(D, int(1.25 * D)),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(int(1.25 * D), VOCAB_SIZE)
        )

    def forward(self, src, tgt, sizes):
        extend = torch.tensor([VOCAB_SIZE - 2] * len(tgt)).unsqueeze(1).to(device)
        tgt = torch.cat([extend, tgt[:, :-1]], dim=1)
        src_encoding = self.encode(src)
        tgt_encoding = self.encode(tgt)
        out = self.encode_decode(src_encoding, tgt_encoding, sizes)
        out = self.fc(out)
        return out

    def encode_decode(self, src, tgt, sizes):
        mask = get_mask(sizes)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(sizes[0]).to(device)
        memory = self.te(src, src_key_padding_mask=mask)
        td_out = self.td(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=mask,
                         memory_key_padding_mask=mask)
        td_out = rnn.pack_padded_sequence(td_out, sizes, batch_first=True).data
        return td_out


if __name__ == "__main__":
    model = ABiTE2CNN(0.6, 4).to(device)
    model.name = "{}S".format(model.name)
    pretrained_dict = torch.load("mods3/{}.pt".format(model.name))
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict)
    torch.save(model_dict, "mods3/{}.pt".format(model.name))
