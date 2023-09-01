import argparse
import os
import gc
import json
import glob
import sys
import torch
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.ssl_models import DenoisingAutoEncoderConfig
from main_supervised import obtain_data, evaluate, used_features, categorical_features, numerical_features
import pdb


def obtain_ssl_data_POST(args):
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
    ssl_train_data, ssl_test_data = train_test_split(whole_data, test_size=0.2, random_state=args.exp_id)
    return ssl_train_data, ssl_test_data, whole_data


def obtain_ssl_data_Simulated(args):
    ssl_features = used_features[:-1]
    whole_data = pd.read_csv("datasets/Simulated_SS7/post_20subs_normal_full.csv")
    used_data = whole_data[ssl_features]
    ssl_train_data, ssl_test_data = train_test_split(used_data, test_size=0.2, random_state=args.exp_id)
    return ssl_train_data, ssl_test_data, used_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="Simulated_SS7", type=str, help="name of the dataset", choices=['Simulated_SS7', 'POST_SS7'])
    parser.add_argument("--output_dir", default="results", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs for training")
    parser.add_argument("--model", default="DAE", type=str, help="Name of the model for training",
                        choices=["DAE"])

    args = parser.parse_args()
    os.makedirs(f"{args.output_dir}/{args.data_name}/logs", exist_ok=True)
    os.makedirs(f"{args.output_dir}/{args.data_name}/saved_models", exist_ok=True)
    for args.exp_id in np.arange(100):
        obtain_ssl_data_POST(args)
        logger_name = f"{args.output_dir}/{args.data_name}/logs/{args.model}_{args.exp_id}.json"
        train_data, test_data = obtain_data(args)
        optimizer_config = OptimizerConfig()
        if args.data_name == 'POST_SS7':
            ssl_train_data, ssl_test_data, _ = obtain_ssl_data_POST(args)
        else:
            ssl_train_data, ssl_test_data, _ = obtain_ssl_data_Simulated(args)
        ssl_data_config = DataConfig(
                target=None,
                continuous_cols=numerical_features,
                categorical_cols=categorical_features,
                normalize_continuous_features=True,
                handle_unknown_categories=False,
                handle_missing_values=False,
            )
        ssl_trainer_config = TrainerConfig(
                auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
                batch_size=100,
                max_epochs=args.epochs,
                early_stopping=None,
                checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
                checkpoints_path=f"{args.output_dir}/{args.data_name}/saved_models",
                checkpoints_name=f"ssl_{args.model}_{args.exp_id}",
                load_best=True,  # After training, load the best checkpoint
                accelerator="gpu"
            )
        encoder_config = CategoryEmbeddingModelConfig(
                task="backbone",
                head=None
            )
        decoder_config = CategoryEmbeddingModelConfig(
                task="backbone",
                head=None
            )
        ssl_model_config = DenoisingAutoEncoderConfig(
                noise_strategy="zero",
                default_noise_probability=0.7,
                encoder_config=encoder_config,
                decoder_config=decoder_config
            )
        ssl_tabular_model = TabularModel(
            data_config=ssl_data_config,
            model_config=ssl_model_config,
            optimizer_config=optimizer_config,
            trainer_config=ssl_trainer_config
        )
        ssl_tabular_model.pretrain(train=ssl_train_data, validation=ssl_test_data)
        ft_trainer_config = TrainerConfig(
                auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
                batch_size=100,
                max_epochs=args.epochs,
                early_stopping=None,
                checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
                checkpoints_path=f"{args.output_dir}/{args.data_name}/saved_models",
                checkpoints_name=f"ft_{args.model}_{args.exp_id}",
                load_best=True,  # After training, load the best checkpoint
                accelerator="gpu"
            )
        tabular_model = ssl_tabular_model.create_finetune_model(
                task="classification",
                target=['label'],
                head="LinearHead",
                trainer_config=ft_trainer_config,
                optimizer_config=optimizer_config,
                head_config={
                    "layers": "512-256-512-64",
                    "activation": "ReLU",
                },
            )
        # Check if the new model has the pretrained weights
        assert torch.equal(ssl_tabular_model.model.encoder.linear_layers[0].weight,
                               tabular_model.model._backbone.encoder.linear_layers[0].weight)
        tabular_model.finetune(train=train_data, validation=test_data, freeze_backbone=True)
        pred_df = tabular_model.predict(test_data)
        np.save(f"{args.output_dir}/{args.data_name}/logs/true_{args.exp_id}.npy", test_data['label'])
        np.save(f"{args.output_dir}/{args.data_name}/logs/prediction_{args.exp_id}.npy", pred_df["prediction"])
        result = evaluate(test_data['label'], pred_df["prediction"])
        with open(logger_name, "w") as outfile:
            outfile.write(json.dumps(result))
        del logger_name
        del ssl_tabular_model
        del tabular_model
        del pred_df
        del result
        gc.collect()


if __name__ == "__main__":
    main()
