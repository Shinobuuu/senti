import pandas as pd
import random
from sklearn.model_selection import train_test_split
import logging
import re
from collections import Counter
from deploy import VOCAB_SIZE, tokenize, read_word_list, remove_stop_words, save_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
corp_path = "corps/Electronics_5.json"
stop_list_path = "corps/en_stop_list.txt"
word_list_path = "{}/word_list.txt".format(save_path)
remove_chars = '[0-9!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
self_seq = [[0, 1, 2, 3], [2, 3, 1, 0], [1, 0, 3, 2], [3, 2, 0, 1]]
DATA_SIZE = 160000


def long_cut(length):
    indexes = [int(length / 4)] * 4
    a = length % 4
    for i in range(a):
        indexes[i] += 1
    for i in range(1, 4):
        indexes[i] += indexes[i - 1]
    return [0] + indexes


def short_cut(length):
    indexes = [4] * int(length / 4)
    indexes.append(length % 4)
    for i in range(1, len(indexes)):
        indexes[i] += indexes[i - 1]
    return [0] + indexes


def count_words(dataframe):
    sl = read_word_list(stop_list_path)
    sequence = []
    for sentence in dataframe.comment:
        sen = sentence.lower().split()
        sequence.extend(remove_stop_words(sen, sl))
    word_dictionary = dict(Counter(sequence).most_common(VOCAB_SIZE - 3))
    index_to_word = [word for word in word_dictionary.keys()]
    index_to_word.append("<s>")
    index_to_word.append("<unk>")
    index_to_word.insert(0, "<pad>")
    return index_to_word


def screen_data(path):
    logger.info("Loading corps...")
    dataframe_json = pd.read_json(path, lines=True)
    dataframe = pd.DataFrame(dataframe_json)
    dataframe = dataframe[(dataframe.overall == 5) | (dataframe.overall == 1)]
    sl = read_word_list(stop_list_path)
    list_neg, list_pos = [], []
    for i, series in dataframe.iterrows():
        a = re.sub(remove_chars, '', series.reviewText.lower()).split()
        a = remove_stop_words(a, sl)
        if 16 <= len(a) <= 128:
            if series.overall < 2:
                list_neg.append(' '.join(a))
            if series.overall > 4:
                list_pos.append(' '.join(a))
    logger.info("Positive: {:.0f} Negative: {:.0f}".format(len(list_pos), len(list_neg)))
    pos_l = DATA_SIZE - len(list_neg)
    dirt_ = {"comment": list_pos[:pos_l] + list_neg,
             "label": [1] * pos_l + [0] * len(list_neg)}
    df = pd.DataFrame(dirt_)
    df_train, df_ = train_test_split(df, test_size=0.2, shuffle=True, random_state=5)
    df_verify, df_test = train_test_split(df_, test_size=0.5, shuffle=True, random_state=5)

    logger.info(
        "Train set: {:.0f} Verify set: {:.0f} Test set: {:.0f}".format(len(df_train), len(df_verify), len(df_test)))
    df_train.to_csv("{}/train.csv".format(save_path))
    df_verify.to_csv("{}/validation.csv".format(save_path))
    df_test.to_csv("{}/test.csv".format(save_path))


def get_long(cut_sent):
    pos = long_cut(len(cut_sent))
    s = [cut_sent[pos[i]:pos[i + 1]] for i in range(4)]
    res = []
    for seq in self_seq:
        new_seq = []
        for i in seq:
            new_seq += s[i]
        res.append(' '.join(new_seq))
    return res


def get_short(cut_sent):
    pos = short_cut(len(cut_sent))
    s = [cut_sent[pos[i]:pos[i + 1]] for i in range(len(pos) - 1)]
    res = []
    for i in range(4):
        new_seq = []
        for j in range(len(s)):
            if j % 4 == i:
                a = list(s[j])
                random.shuffle(a)
                new_seq += a
            else:
                new_seq += s[j]
        res.append(' '.join(new_seq))
    return res


def make_self_supervised_dataset(set_path, self_path):
    logger.info("Loading training set...")
    self_dir = {"shuffle": [], "correct": []}
    dataframe = pd.read_csv(set_path)
    for sent in dataframe.comment:
        cut_sent = sent.split()
        if len(cut_sent) >= 16:
            long_seqs = get_long(cut_sent)
            short_seqs = get_short(cut_sent)
            new_seqs = list(set(long_seqs + short_seqs))
            self_dir['shuffle'] += new_seqs
            self_dir['correct'] += [sent] * len(new_seqs)
    df_self = pd.DataFrame(self_dir)
    df_self.to_csv(self_path)
    logger.info("Generate {:.0f} self-supervised datas".format(len(df_self)))


def test():
    dataframe = pd.read_csv("mods3/train.csv")
    length = [0] * 500
    for i, series in dataframe.iterrows():
        sen = series.comment.split()
        length[len(sen) - 12] += 1
    print(length[:64])


if __name__ == "__main__":
    df = pd.read_csv("{}/train.csv".format(save_path))
    wl = count_words(df)
    f = open(word_list_path, 'w+')
    for w in wl:
        f.write(w + '\n')
    # print(df.reviewText[0])
    # test()
    # screen_data(corp_path)
    # make_self_supervised_dataset("{}/validation.csv".format(save_path), "{}/self4v.csv".format(save_path))
    # make_self_supervised_dataset("{}/train.csv".format(save_path), "{}/self4t.csv".format(save_path))
    # print()
