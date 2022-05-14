import torch
import torch.utils.data as dt
import numpy as np
import matplotlib.pyplot as plt
import jieba
import torch.nn.utils.rnn as rnn
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

VOCAB_SIZE = 25000
TEST_BATCH_SIZE = 80
D = 256
save_path = "mods3"
word_list_path = "{}/word_list.txt".format(save_path)
stop_words_path = "corps/en_stop_list.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SDataset(dt.Dataset):
    def __init__(self, test_data):
        super(SDataset, self).__init__()
        self.test_data = test_data

    def __getitem__(self, item):
        return self.test_data[item]

    def __len__(self):
        return len(self.test_data)


class LDataset(dt.Dataset):
    def __init__(self, dataframe, is_self=False):
        super(LDataset, self).__init__()
        self.word_list = read_word_list(word_list_path)
        self.train_data = pack_datas(dataframe, self.word_list, is_self)
        if not is_self:
            self.count_datas()
        else:
            logger.info("Generated {:.0f} self supervised datas".format(len(self.train_data)))

    def __getitem__(self, item):
        return self.train_data[item]

    def __len__(self):
        return len(self.train_data)

    def count_datas(self):
        total, pos, neg = 0, 0, 0
        for sentence in self.train_data:
            total += 1
            if sentence[-1]:
                pos += 1
            else:
                neg += 1
        logger.info("Loaded {:.0f} datas, positive: {:.0f} negative: {:.0f}".format(total, pos, neg))


def tokenize(review):
    words = []
    cut_list = jieba.cut(review)
    for word in cut_list:
        if "\u4e00" <= word <= "\u9fff":
            words.append(word)
    return words


def read_word_list(path):
    w_list = []
    file_read = open(path, "r", encoding='utf-8')
    for line in file_read:
        w_list.append(line.strip())
    return w_list


def pack_datas(dataframe, word_list, is_self):  # 将数据集中的数据和标签打包
    word_to_index = {word: i for i, word in enumerate(word_list)}
    inputs = []
    for i, record in dataframe.iterrows():
        if is_self:
            shuffle_sequence = [word_to_index.get(w, VOCAB_SIZE - 1) for w in record.shuffle.lower().split()]
            correct_sequence = [word_to_index.get(w, VOCAB_SIZE - 1) for w in record.correct.lower().split()]
            sequence = shuffle_sequence + correct_sequence
        else:
            sequence = [word_to_index.get(w, VOCAB_SIZE - 1) for w in record.comment.lower().split()]
            sequence.append(record.label)
        sequence = torch.tensor(sequence)
        inputs.append(sequence)
    return inputs


def remove_stop_words(sentence, stop_word_list):  # 去除句子中的停用词
    result = []
    for word in sentence:
        if word not in stop_word_list:
            result.append(word)
    return result


def get_predict_loss(v_set, model):  # 获得分类模型的输出
    model.eval()
    loss_f = torch.nn.CrossEntropyLoss().to(device)
    losses = []
    dl = dt.DataLoader(v_set, batch_size=TEST_BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    v_outputs = torch.empty(0, 2)
    v_outputs = v_outputs.to(device)
    v_labels = torch.empty(0)
    v_labels = v_labels.to(device)
    for v_datas in dl:
        v_inputs, labels, v_sizes = v_datas
        v_inputs, labels = v_inputs.to(device), labels.to(device)
        with torch.no_grad():
            out = model(v_inputs, v_sizes)
        loss = loss_f(out, labels)
        losses.append(loss.item())
        v_outputs = torch.cat([v_outputs, out], dim=0)
        v_labels = torch.cat([v_labels, labels])
    v_outputs = v_outputs.cpu()
    v_labels = v_labels.cpu()
    avg_loss = np.average(losses)
    return v_outputs, v_labels, avg_loss.item()


def calculate(v_set, model):
    v_outputs, v_labels, _ = get_predict_loss(v_set, model)
    predicts = v_outputs.argmax(dim=1)
    ac = accuracy_score(v_labels, predicts)
    p = precision_score(v_labels, predicts)
    r = recall_score(v_labels, predicts)
    f1 = f1_score(v_labels, predicts)
    return round(ac, 4), round(p, 4), round(r, 4), round(f1, 4)


def collate_fn(train_data):
    train_data.sort(key=lambda data: len(data), reverse=True)
    texts = []
    labels = []
    for seq in train_data:
        texts.append(seq[:-1])
        labels.append(seq[-1].item())
    labels = torch.tensor(labels)
    sizes = torch.tensor([len(s) for s in texts])
    texts = rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    return texts, labels, sizes


def collate_as_self(train_data):
    train_data.sort(key=lambda data: len(data), reverse=True)
    src, tgt = [], []
    for seq in train_data:
        length = int(len(seq) / 2)
        src.append(seq[:length])
        tgt.append(seq[length:])
    data_sizes = torch.tensor([len(s) for s in src])
    src = rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt = rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
    return src, tgt, data_sizes


def train(model, loss_f, optimizer, dataset, bs):
    model.train()
    dataloader = dt.DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn, num_workers=8)
    t_loss = []
    for training_datas in dataloader:
        optimizer.zero_grad()
        x, y, sizes = training_datas
        x, y = x.to(device), y.to(device)
        y_ = model(x, sizes)
        loss = loss_f(y_, y)
        loss = loss.cpu()
        t_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    avg_loss = np.average(t_loss)
    return avg_loss.item()


def get_self_loss(model, dataset):
    model.eval()
    loss_f = torch.nn.CrossEntropyLoss().to(device)
    dataloader = dt.DataLoader(dataset, batch_size=TEST_BATCH_SIZE, collate_fn=collate_as_self, num_workers=8)
    losses = []
    for testing_datas in dataloader:
        src, tgt, sizes = testing_datas
        src, tgt = src.to(device), tgt.to(device)
        labels = rnn.pack_padded_sequence(tgt, sizes, batch_first=True).data
        with torch.no_grad():
            y_ = model(src, tgt, sizes)
        loss = loss_f(y_, labels)
        losses.append(loss.item())
    avg_loss = np.average(losses)
    return avg_loss.item()


def train_as_self(model, loss_f, optimizer, dataset, bs):
    model.train()
    dataloader = dt.DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collate_as_self, num_workers=8)
    t_loss = []
    for training_datas in dataloader:
        optimizer.zero_grad()
        src, tgt, sizes = training_datas
        src, tgt = src.to(device), tgt.to(device)
        labels = rnn.pack_padded_sequence(tgt, sizes, batch_first=True).data
        y_ = model(src, tgt, sizes)
        loss = loss_f(y_, labels).cpu()
        t_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    avg_loss = np.average(t_loss)
    return avg_loss.item()


def p_d(s_path):  # 用上述生成的1000个xy值对生成1000个点
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # 删除右边框设为无
    ax.spines['top'].set_color('none')  # 删除上边框设为无
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))  # 调整x轴位置
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))  # 调整y轴位置
    plt.savefig(s_path)
    plt.show()
    plt.close(0)


def make_sigmoid():
    x = np.linspace(-6, 6, 1000)  # 这个表示在-5到5之间生成1000个x值
    y = [1 / (1 + np.exp(-i)) for i in x]  # 对上述生成的1000个数循环用sigmoid公式求对应的y
    plt.xlim((-6, 6))
    plt.ylim((0.00, 1.00))
    plt.yticks([0.5, 1.0], [0.5, 1.0])  # 设置y轴显示的刻度
    plt.plot(x, y, color='black')
    p_d("image/sigmoid.png")


def make_tanh():
    x = np.linspace(-3, 3, 1000)  # 这个表示在-5到5之间生成1000个x值
    y = [(np.exp(i) - np.exp(-i)) / (np.exp(i) + np.exp(-i)) for i in x]  # 对上述生成的1000个数循环用sigmoid公式求对应的y
    plt.xlim((-3, 3))
    plt.ylim((0.00, 1.00))
    plt.yticks([-1, -0.5, 0.5, 1.0], [-1, -0.5, 0.5, 1.0])  # 设置y轴显示的刻度
    plt.plot(x, y, color='black')
    p_d("image/tanh.png")


def make_relu():
    x = np.linspace(-1, 1, 1000)  # 这个表示在-5到5之间生成1000个x值
    y = [i if i >= 0 else 0 for i in x]  # 对上述生成的1000个数循环用sigmoid公式求对应的y
    plt.xlim((-1, 1))
    plt.xticks([-1.0, -0.5, 0, 0.5, 1.0])
    plt.ylim((-1.00, 1.00))
    plt.yticks([-0.5, 0.5, 1.0], [-0.5, 0.5, 1.0])  # 设置y轴显示的刻度
    plt.plot(x, y, color='black')
    p_d("image/relu.png")


if __name__ == "__main__":
    make_relu()
