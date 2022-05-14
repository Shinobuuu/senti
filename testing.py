import torch
import pandas as pd
from deploy import calculate, device, LDataset, save_path, get_predict_loss
from baselines import TextCNN, TextLSTM, TextLCNN
from transformers import TE2CNN, WBiTE2CNN, DualTE2CNN, ABiTE2CNN, MBiTE2CNN
from training import SELF_LAYERS, DROPOUT
import torch.nn.functional as f

testing_set_path = "{}/test.csv".format(save_path)
makers = ["o", "v"]
feature_map_size = 4


def draw_points(fm, mk):
    x = fm[:, 0]
    y = fm[:, 0]
    plt.scatter(x, y, marker=mk)


def make_feature_map(model, ds_t):
    fe, labels, _ = get_predict_loss(ds_t, model)
    fe = f.softmax(fe, dim=1).float()
    print("{}:".format(model.name))
    torch.set_printoptions(precision=4)
    print(fe, labels)


def test_model(model, ds):
    for param in model.parameters():
        param.requires_grad = False
    ac, p, r, f1 = calculate(ds, model)
    print("Name: {} Accuracy: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(model.name, ac, p, r, f1))


def feature_pca():
    df_t = pd.read_csv(testing_set_path)[359:365]
    ds_t = LDataset(df_t)
    models = [
        TextLCNN
    ]

    selfs = [
        ABiTE2CNN,
        DualTE2CNN
    ]
    for model_n in models:
        model = model_n(DROPOUT).to(device)
        model.load_state_dict(torch.load("{}/{}.pt".format(save_path, model.name)))
        make_feature_map(model, ds_t)

    for model_n in selfs:
        model = model_n(DROPOUT).to(device)
        model.name = "{}S".format(model.name)
        model.load_state_dict(torch.load("{}/{}.pt".format(save_path, model.name)))
        make_feature_map(model, ds_t)


def test():
    df_t = pd.read_csv(testing_set_path)
    ds_t = LDataset(df_t)
    models = [
        TextCNN,
        TextLSTM,
        TextLCNN,
        # TE2CNN,
        # ABiTE2CNN,
        # MBiTE2CNN,
        # WBiTE2CNN,
    ]
    self_models = [
        # TE2CNN,
        # ABiTE2CNN,
        # MBiTE2CNN,
        # WBiTE2CNN,
        DualTE2CNN,
    ]
    # for model_n in models:
    #     model = model_n(DROPOUT).to(device)
    #     model.load_state_dict(torch.load("{}/{}.pt".format(save_path, model.name)))
    #     test_model(model, ds_t)
    for model_n in self_models:
        model = model_n(DROPOUT).to(device)
        model.name = "{}S".format(model.name)
        model.load_state_dict(torch.load("{}/{}.pt".format(save_path, model.name), map_location=torch.device("cpu")))
        test_model(model, ds_t)


if __name__ == "__main__":
    # feature_pca()
    test()
