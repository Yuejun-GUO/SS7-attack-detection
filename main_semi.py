import argparse
import os
import json
import numpy as np
import torch
import glob
import pandas as pd
import importlib as imp
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def obtain_un_POST():
    file_name = f"datasets/POST_SS7/encrypted_train_data_all/*.csv"
    ssl_features = used_features[:-1]
    count = 0
    for file_path in glob.iglob(file_name, recursive=True):
        file_data = pd.read_csv(file_path)
        used_data = file_data[ssl_features]
        if count == 0:
            whole_data = used_data
        else:
            whole_data = pd.concat([whole_data, used_data], ignore_index=True, axis=0)
        count += 1
    whole_data.to_csv(f"datasets/POST_SS7/encrypted_train_data_all/unlabeled.csv")
    return


def obtain_ssl_data_POST(train_data, test_data):
    whole_data = pd.read_csv(f"datasets/POST_SS7/encrypted_train_data_all/unlabeled.csv")
    X_train = np.concatenate((train_data.to_numpy()[:, :-1], whole_data.to_numpy()[:, 1:]))
    y_train = np.zeros(len(X_train))
    y_train[:len(train_data.to_numpy())] = train_data.to_numpy()[:, -1]
    X_test = test_data.to_numpy()[:, :-1]
    y_test = test_data.to_numpy()[:, -1]
    print(len(y_train))
    return X_train, y_train, X_test, y_test


def obtain_ssl_data_Simulated(train_data, test_data):
    ssl_features = used_features[:-1]
    whole_data = pd.read_csv("datasets/Simulated_SS7/post_20subs_normal_full.csv")
    used_data = whole_data[ssl_features]
    X_train = np.concatenate((train_data.to_numpy()[:, :-1], used_data.to_numpy()))
    y_train = np.zeros(len(X_train))
    y_train[:len(train_data.to_numpy())] = train_data.to_numpy()[:, -1]
    X_test = test_data.to_numpy()[:, :-1]
    y_test = test_data.to_numpy()[:, -1]
    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="Simulated_SS7", type=str, choices=['Simulated_SS7', 'POST_SS7'])
    parser.add_argument("--output_dir", default="results", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs for training")

    args = parser.parse_args()
    for args.exp_id in np.arange(100):
        train_data, test_data = obtain_data(args)
        if args.data_name == "POST_SS7":
            X_train, y_train, X_test, y_test = obtain_ssl_data_POST(train_data, test_data)
        else:
            X_train, y_train, X_test, y_test = obtain_ssl_data_Simulated(train_data, test_data)
        for args.model in ["DevNet", "PReNet", "DeepSAD"]:
            logger_name = f"{args.output_dir}/{args.data_name}/logs/{args.model}/{args.model}_{args.exp_id}.json"
            if os.path.isfile(logger_name):
                continue
            os.makedirs(f"{args.output_dir}/{args.data_name}/logs/{args.model}", exist_ok=True)
            module = imp.import_module('deepod.models')
            model_class = getattr(module, args.model)
            model = model_class(epochs=1, hidden_dims=20, batch_size=100, device='cuda', random_state=42, verbose=0)
            best_loss = 1000
            best_f1 = -1
            for epoch_id in range(args.epochs):
                print(f"{args.model}, {epoch_id}")
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)
                if args.model in ["DeepSAD"]:
                    result = evaluate(y_test, prediction)
                    if result['accuracy'] > best_f1:
                        best_f1 = result['accuracy']
                        best_result = result
                else:
                    loss = torch.mean(model.criterion(torch.Tensor(y_test), torch.Tensor(prediction)))
                    if loss < best_loss:
                        result = evaluate(y_test, prediction)
                        best_result = result
                        best_loss = loss
            with open(logger_name, "w") as outfile:
                outfile.write(json.dumps(best_result))
            del logger_name
            del model
            del result
            gc.collect()


if __name__ == "__main__":
    main()
