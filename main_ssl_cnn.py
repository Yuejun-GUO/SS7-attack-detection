import argparse
import gc
import os
import glob
import pdb

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import pandas as pd
from torch import optim
from torch.autograd import Variable
import json

device = "cpu"
hidden_dim = 15  # number of features in hidden layer
batch_size = 49
input_size = 31
num_epoches = 200
used_features = ['f_same_cggt_is_hlr_oc', 'f_same_cggt_is_hlr_ossn', 'f_velocity_greater_than_1000',
                 'f_count_unloop_country_last_x_hours_ul', 'f_count_gap_ok_sai_and_all_lu',
                 'f_count_ok_cl_between2lu',
                 'f_count_ok_fwsm_mo_between2lu', 'f_count_ok_fwsm_mt_between2lu',
                 'f_count_ok_fwsm_report_between2lu',
                 'f_count_ok_prn_between2lu', 'f_count_ok_psi_between2lu', 'f_count_ok_sai_between2lu',
                 'f_count_ok_si_between2lu', 'f_count_ok_sri_between2lu', 'f_count_ok_srism_between2lu',
                 'f_count_ok_ul_between2lu', 'f_count_ok_ulgprs_between2lu', 'f_count_ok_ussd_between2lu',
                 'f_frequent_ok_cl_between2lu', 'f_frequent_ok_fwsm_mo_between2lu',
                 'f_frequent_ok_fwsm_mt_between2lu',
                 'f_frequent_ok_fwsm_report_between2lu', 'f_frequent_ok_prn_between2lu',
                 'f_frequent_ok_psi_between2lu',
                 'f_frequent_ok_sai_between2lu', 'f_frequent_ok_si_between2lu', 'f_frequent_ok_sri_between2lu',
                 'f_frequent_ok_srism_between2lu', 'f_frequent_ok_ul_between2lu', 'f_frequent_ok_ulgprs_between2lu',
                 'f_frequent_ok_ussd_between2lu', 'label']


def obtain_data(args):
    if args.data_name == 'Simulated_SS7':
        args.data_dir = f"datasets/{args.data_name}/post_20subs_vip_full.csv"
    else:
        args.data_dir = f"datasets/{args.data_name}/encrypted_hp_tuning_data/encrypted_tuning_ul_with_label.csv"
    whole_data = pd.read_csv(args.data_dir)
    used_data = whole_data[used_features]
    if args.data_name == 'POST_SS7':
        train_data, test_data = train_test_split(used_data, test_size=17, train_size=45, random_state=args.exp_id)
    else:
        train_data, test_data = train_test_split(used_data, test_size=17, train_size=68, random_state=args.exp_id)
    return train_data, test_data


def evaluate(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    val_acc = accuracy_score(y_true, y_pred)
    val_pre = precision_score(y_true, y_pred)
    val_rec = recall_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred)
    result = {"accuracy": val_acc,
              "precision": val_pre,
              "recall": val_rec,
              "F1": val_f1}
    return result


class unsuper_Dataset(torch.utils.data.Dataset):
    def __init__(self, unsuper_trainData):
        self.train_data = unsuper_trainData.to_numpy()
        self.train_labels = np.full(len(self.train_data), 0)

    def __getitem__(self, index):
        data, target = self.train_data[index, :], self.train_labels[index]

        return data, target

    def __len__(self):
        return len(self.train_labels)


class super_Dataset(torch.utils.data.Dataset):
    def __init__(self, super_data):
        self.data = super_data.to_numpy()[:, :-1]
        self.labels = super_data["label"].to_numpy()

    def __getitem__(self, index):
        data, target = self.data[index, :], self.labels[index]

        return data, target

    def __len__(self):
        return len(self.labels)


def obtain_ssl_data_POST():
    file_name = f"datasets/POST_SS7/encrypted_train_data_all/*.csv"
    ssl_features = used_features[:-1]
    count = 0
    for file_path in glob.iglob(file_name, recursive=True):
        file_data = pd.read_csv(file_path)
        used_data = file_data[ssl_features]
        if count == 0:
            whole_data = used_data
            break
        else:
            whole_data = pd.concat([whole_data, used_data], ignore_index=True, axis=0)
        count += 1
    return whole_data


def obtain_ssl_data_Simulated():
    ssl_features = used_features[:-1]
    whole_data = pd.read_csv("datasets/Simulated_SS7/post_20subs_normal_full.csv")
    used_data = whole_data[ssl_features]

    return used_data


class CNNEncoder(nn.Module):

    def __init__(self, hidden_size, output_size, channels=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cnnseq = nn.Sequential(
            nn.Conv1d(1, 31, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm1d(31),
            nn.ReLU(inplace=True),
            nn.Conv1d(31, 31, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm1d(31),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.BatchNorm1d(31),
            nn.ReLU(inplace=True),
            nn.Conv1d(31, 62, kernel_size=3, padding=2, bias=False),
            nn.BatchNorm1d(62),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(62),  # 128
        )
        self.reggresor = nn.Sequential(  # 30:64 #45:128, #60:
            nn.Linear(62, 124, bias=False),
            nn.BatchNorm1d(124),
            nn.ReLU(inplace=True),
            #            nn.Dropout(0.5),
            nn.Linear(124, self.output_size)
        )

    def forward(self, images):
        code = self.cnnseq(images)
        code = code.view([images.size(0), -1])
        code = self.reggresor(code)
        code = code.view([code.size(0), self.output_size])
        return code


def run_unsupervised_training(args):
    learning_rate = 0.001
    batch_size = 100
    Mode = 0
    if args.data_name == 'POST_SS7':
        ssl_train_data = obtain_ssl_data_POST()
    else:
        ssl_train_data = obtain_ssl_data_Simulated()
    ssl_train_dataset = unsuper_Dataset(ssl_train_data)
    ssl_train_loader = torch.utils.data.DataLoader(ssl_train_dataset, batch_size=batch_size, shuffle=True,
                                                   drop_last=True)
    if Mode == 0:
        encoder = CNNEncoder(hidden_dim, 2)
        encoder = encoder.to(device=device)
        # loss_function = nn.MSELoss()
        loss_function = torch.nn.CrossEntropyLoss()  # change the loss function to cross entropy
        optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        for epoch in range(num_epoches):
            for i, data in enumerate(ssl_train_loader, 1):
                seq, target = data
                seq, target = seq.to(device), target.to(device)
                seq = Variable(seq).float()
                target = Variable(target).long()
                seq = seq.reshape(batch_size, 1, 31)
                out = encoder(seq)
                loss = loss_function(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        torch.save(encoder, args.pretrain_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="Simulated_SS7", type=str, help="name of the dataset", choices=['Simulated_SS7', 'POST_SS7'])
    parser.add_argument("--output_dir", default="results", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs for training")
    parser.add_argument("--model", default="SSL_CNN", type=str, help="Name of the model for training",
                        choices=["SSL_CNN"])

    args = parser.parse_args()
    args.pretrain_model = f"{args.output_dir}/{args.data_name}/saved_models/ssl_pretrained-model.pth"
    os.makedirs(f"{args.output_dir}/{args.data_name}/logs/{args.model}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/{args.data_name}/saved_models/", exist_ok=True)
    if not os.path.isfile(args.pretrain_model):
        print("run unsupervised")
        run_unsupervised_training(args)
    for args.exp_id in np.arange(100):
        learning_rate = 0.0001
        epsilon = 1e-08
        weight_decay = 0
        num_classes = 2  # for supervised learning
        encoder = torch.load(args.pretrain_model, map_location=device)
        logger_name = f"{args.output_dir}/{args.data_name}/logs/{args.model}/{args.model}_{args.exp_id}.json"
        train_data, test_data = obtain_data(args)
        train_dataset = super_Dataset(train_data)
        test_dataset = super_Dataset(test_data)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        ct = 0
        for child in encoder.children():
            if ct < -1:
                for param in child.parameters():
                    param.requires_grad = False
            ct += 1
        encoder.reggresor = nn.Sequential(
            nn.Linear(62, 124, bias=False),
            nn.BatchNorm1d(124),
            nn.ReLU(inplace=True),
            nn.Linear(124, 62)
        )
        encoder.output_size = 62
        linear = nn.Sequential(nn.Linear(62, 62), nn.ReLU(inplace=True), nn.Linear(62, 62), nn.ReLU(inplace=True),
                               nn.Linear(62, num_classes), nn.Softmax(dim=1))
        transferedModel = nn.Sequential(encoder, linear)
        transferedModel = transferedModel.to(device=device)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, transferedModel.parameters()), lr=learning_rate,
                                     eps=epsilon, weight_decay=weight_decay)
        min_loss_val = 1000
        for epoch in range(num_epoches):
            transferedModel.train()
            for data in train_loader:
                seq, target = data
                seq, target = seq.to(device), target.to(device)
                seq = Variable(seq).float()
                target = Variable(target).long()
                seq = seq.reshape(len(target), 1, 31)
                out = transferedModel(seq)
                target.reshape(1, len(target))
                loss = loss_function(out, target)
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            transferedModel.eval()
            loss_val = 0
            with torch.no_grad():
                for item in test_loader:
                    val_seq, val_target = item
                    val_seq, val_target = val_seq.to(device), val_target.to(device)
                    val_seq = Variable(val_seq).float().reshape(len(val_seq), 1, 31)
                    val_target = Variable(val_target).long()
                    out = transferedModel(val_seq)
                    predict_label = out.max(1)[1]
                    loss_val += loss_function(out, val_target)
            loss_val = loss_val / len(test_loader)
            if loss_val < min_loss_val:
                min_loss_val = loss_val
                result = evaluate(test_data['label'], predict_label.cpu().detach().numpy())
        with open(logger_name, "w") as outfile:
            outfile.write(json.dumps(result))
        del encoder
        del transferedModel
        gc.collect()


if __name__ == "__main__":
    main()