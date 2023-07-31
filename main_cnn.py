import argparse
import gc
import os
import pdb

import torch
import numpy as np
from main_supervised import obtain_data, evaluate
import torch.nn as nn
from torch.autograd import Variable
import json


hidden_dim = 15  # number of features in hidden layer
batch_size = 45
input_size = 31
num_epoches = 200


class super_Dataset(torch.utils.data.Dataset):
    def __init__(self, super_data):
        self.data = super_data.to_numpy()[:, :-1]
        self.labels = super_data["label"].to_numpy()

    def __getitem__(self, index):
        data, target = self.data[index, :], self.labels[index]

        return data, target

    def __len__(self):
        return len(self.labels)


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
            nn.Linear(124, self.output_size)
        )

    def forward(self, images):
        code = self.cnnseq(images)
        code = code.view([images.size(0), -1])
        code = self.reggresor(code)
        code = code.view([code.size(0), self.output_size])
        return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="Simulated_SS7", type=str, help="name of the dataset", choices=['Simulated_SS7', 'POST_SS7'])
    parser.add_argument("--output_dir", default="results", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs for training")
    parser.add_argument("--model", default="CNN", type=str, help="Name of the model for training",
                        choices=["CNN"])

    args = parser.parse_args()
    pdb.set_trace()
    os.makedirs(f"{args.output_dir}/{args.data_name}/{args.model}/logs/{args.model}", exist_ok=True)
    for args.exp_id in np.arange(100):
        learning_rate = 0.0001
        epsilon = 1e-08
        weight_decay = 0
        num_classes = 2  # for supervised learning
        logger_name = f"{args.output_dir}/{args.data_name}/logs/{args.model}/{args.model}_{args.exp_id}.json"
        train_data, test_data = obtain_data(args)
        train_dataset = super_Dataset(train_data)
        test_dataset = super_Dataset(test_data)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        encoder = CNNEncoder(hidden_dim, 62)
        linear = nn.Sequential(nn.Linear(62, 62), nn.ReLU(inplace=True), nn.Linear(62, 62), nn.ReLU(inplace=True),
                               nn.Linear(62, num_classes), nn.Softmax(dim=1))
        transferedModel = nn.Sequential(encoder, linear)
        pdb.set_trace()
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, transferedModel.parameters()), lr=learning_rate,
                                     eps=epsilon, weight_decay=weight_decay)
        min_loss_val = 1000
        for epoch in range(num_epoches):
            transferedModel.train()
            running_loss = 0.0
            total_target = 0

            for data in train_loader:
                seq, target = data
                seq = Variable(seq).float()
                target = Variable(target).long()
                seq = seq.reshape(len(target), 1, 31)
                out = transferedModel(seq)
                target.reshape(1, len(target))
                loss = loss_function(out, target)
                _, pred = torch.max(out, 1)
                total_target += len(pred)
                running_loss += loss.item() * target.size(0)
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            transferedModel.eval()
            loss_val = 0
            with torch.no_grad():
                for item in test_loader:
                    val_seq, val_target = item
                    val_seq = Variable(val_seq).float().reshape(len(val_seq), 1, 31)
                    val_target = Variable(val_target).long()
                    out = transferedModel(val_seq)
                    predict_label = out.max(1)[1]
                    loss_val += loss_function(out, val_target)
            loss_val = loss_val / len(test_loader)
            if loss_val < min_loss_val:
                min_loss_val = loss_val
                result = evaluate(test_data['label'], predict_label)
        with open(logger_name, "w") as outfile:
            outfile.write(json.dumps(result))
        del predict_label
        del encoder
        del transferedModel
        gc.collect()


if __name__ == "__main__":
    main()