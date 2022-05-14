import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from baselines import TextCNN, TextLSTM, TextLCNN, TextGRU
from transformers import TS2S, ABiTE2CNN, MBiTE2CNN, WBiTE2CNN, TE2CNN, DualTE2CNN
from deploy import train, LDataset, device, logger, train_as_self, get_self_loss, get_predict_loss, save_path, calculate
from pytorchtools import EarlyStopping

LR = 0.0001
RS = 5
EPOCH = 400
PATIENCE = 6
SELF_LAYERS = 4
BATCH_SIZE = 256
DROPOUT = 0.6  # 0.4-0.6

train_set_path = "{}/train.csv".format(save_path)
v_set_path = "{}/validation.csv".format(save_path)
self_set_path = "{}/self4t.csv".format(save_path)
self_verify_path = "{}/self4v.csv".format(save_path)
testing_set_path = "{}/test.csv".format(save_path)

loss_F = nn.CrossEntropyLoss().to(device)


def init_mod(model):
    logger.info("Initializing model: {}".format(model.name))
    for mo in model.modules():
        if isinstance(mo, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(mo.weight, gain=nn.init.calculate_gain('relu'))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def train_model(model, optimizer, epoch, ds_t, ds_v, bs, from_check=False):
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)
    if from_check:
        model.load_state_dict(torch.load("{}/{}.pt".format(save_path, model.name)))
        _, _, zero_loss = get_predict_loss(ds_v, model)
        early_stopping(zero_loss, model)
    logger.info("Start training model: {}, batch size = {:.0f}".format(model.name, bs))
    print(optimizer)
    for e in range(epoch):
        avg_loss = train(model, loss_F, optimizer, ds_t, bs)
        _, _, v_loss = get_predict_loss(ds_v, model)
        logger.info("epoch{:.0f}, train loss: {:.6f}, valid loss: {:.6f}".format(e + 1, avg_loss, v_loss))
        early_stopping(v_loss, model)
        if early_stopping.early_stop:
            logger.info("Early Stopping!")
            break
        torch.cuda.empty_cache()
    model.load_state_dict(torch.load("checkpoint.pt"))
    a, p, r, f1 = calculate(ds_v, model)
    logger.info("Valid set: Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(a, p, r, f1))
    torch.save(model.state_dict(), "{}/{}.pt".format(save_path, model.name))
    print()


def self_task(from_checkpoint=False):
    df_st = pd.read_csv(self_set_path)
    df_sv = pd.read_csv(self_verify_path)
    ds_st = LDataset(df_st, True)
    ds_sv = LDataset(df_sv, True)
    logger.info("Datasets ready\n")
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path="self_checkpoint.pt")
    model_s = TS2S(DROPOUT, SELF_LAYERS).to(device)
    if from_checkpoint:
        logger.info("from checkpoint...")
        model_s.load_state_dict(torch.load("self_checkpoint.pt"))
        loss_zero = get_self_loss(model_s, ds_sv)
        early_stopping(loss_zero, model_s)
    else:
        init_mod(model_s)
    optimizer_adam = optim.RAdam(model_s.parameters(), lr=LR)
    print(optimizer_adam)
    print("Start Self_supervised training for model: {}".format(model_s.name))
    for e in range(EPOCH):
        avg_loss = train_as_self(model_s, loss_F, optimizer_adam, ds_st, BATCH_SIZE)
        v_loss = get_self_loss(model_s, ds_sv)
        logger.info("epoch{:.0f}, train loss: {:.6f}, valid loss: {:.6f}".format(e + 1, avg_loss, v_loss))
        early_stopping(v_loss, model_s)
        if early_stopping.early_stop:
            logger.info("Early Stopping!")
            break
        torch.cuda.empty_cache()
    torch.save(torch.load("self_checkpoint.pt"), "{}/{}.pt".format(save_path, model_s.name))


def finetune(model_name, ds_t, ds_v, bs, dr=DROPOUT):
    model = model_name(dr, SELF_LAYERS).to(device)
    model.name = "{}S".format(model.name)
    pretrained_dict = torch.load("{}/TS2S{:.0f}.pt".format(save_path, model.num_layers))
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and "fc" not in k)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.te.requires_grad_(False)
    model.emb_encoder.requires_grad_(False)
    opt = optim.Adam(model.parameters(), lr=LR)
    train_model(model, opt, EPOCH, ds_t, ds_v, bs)


def directly(model_name, ds_t, ds_v, bs, dr=DROPOUT):
    model = model_name(dr).to(device)
    init_mod(model)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_model(model, optimizer, EPOCH, ds_t, ds_v, bs)


def senti_task():
    baseline_mods = [
        TextCNN,
        TextLSTM,
        # TextGRU,
        TextLCNN,
    ]

    tec_mods = [
        TE2CNN,
        ABiTE2CNN,
        MBiTE2CNN,
        WBiTE2CNN,
    ]

    self_tec_mods = [
        DualTE2CNN,
        # TE2CNN,  # 108
        # ABiTE2CNN,  # 108
        # MBiTE2CNN,
        # WBiTE2CNN,
    ]

    df_t = pd.read_csv(train_set_path)
    df_v = pd.read_csv(v_set_path)
    ds_t = LDataset(df_t)
    ds_v = LDataset(df_v)
    logger.info("Datasets ready\n")

    # for model_n in self_tec_mods:
    #     finetune(model_n, ds_t, ds_v, bs=BATCH_SIZE, dr=DROPOUT)

    for model_n in baseline_mods:
        directly(model_n, ds_t, ds_v, bs=BATCH_SIZE, dr=0.5)
    #
    # for model_n in tec_mods:
    #     directly(model_n, ds_t, ds_v, bs=BATCH_SIZE, dr=DROPOUT)


if __name__ == "__main__":
    setup_seed(RS)
    # self_task()
    senti_task()
