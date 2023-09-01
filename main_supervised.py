import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pytorch_tabular import TabularModel
import time
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import (AutoIntConfig, CategoryEmbeddingModelConfig,
                                    FTTransformerConfig, GatedAdditiveTreeEnsembleConfig,
                                    NodeConfig, TabNetModelConfig, TabTransformerConfig)
import gc

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
categorical_features = ['f_same_cggt_is_hlr_oc', 'f_same_cggt_is_hlr_ossn', 'f_velocity_greater_than_1000']
numerical_features = ['f_count_unloop_country_last_x_hours_ul', 'f_count_gap_ok_sai_and_all_lu',
                      'f_count_ok_cl_between2lu', 'f_count_ok_fwsm_mo_between2lu', 'f_count_ok_fwsm_mt_between2lu',
                      'f_count_ok_fwsm_report_between2lu', 'f_count_ok_prn_between2lu', 'f_count_ok_psi_between2lu',
                      'f_count_ok_sai_between2lu', 'f_count_ok_si_between2lu', 'f_count_ok_sri_between2lu',
                      'f_count_ok_srism_between2lu', 'f_count_ok_ul_between2lu', 'f_count_ok_ulgprs_between2lu',
                      'f_count_ok_ussd_between2lu', 'f_frequent_ok_cl_between2lu',
                      'f_frequent_ok_fwsm_mo_between2lu', 'f_frequent_ok_fwsm_mt_between2lu',
                      'f_frequent_ok_fwsm_report_between2lu', 'f_frequent_ok_prn_between2lu',
                      'f_frequent_ok_psi_between2lu', 'f_frequent_ok_sai_between2lu', 'f_frequent_ok_si_between2lu',
                      'f_frequent_ok_sri_between2lu', 'f_frequent_ok_srism_between2lu',
                      'f_frequent_ok_ul_between2lu', 'f_frequent_ok_ulgprs_between2lu',
                      'f_frequent_ok_ussd_between2lu']


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


def obtain_tabular_model(args):
    data_config = DataConfig(
        target=['label'],
        continuous_cols=numerical_features,
        categorical_cols=categorical_features,
    )
    trainer_config = TrainerConfig(
        auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
        batch_size=100,
        max_epochs=args.epochs,
        early_stopping=None,
        checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
        checkpoints_path=f"{args.output_dir}/{args.data_name}/saved_models",
        checkpoints_name=f"{args.model}_{args.exp_id}",
        load_best=True,  # After training, load the best checkpoint
        progress_bar='none',
        accelerator="gpu"
    )
    optimizer_config = OptimizerConfig()
    if args.model == "AutoInt":
        model_config = AutoIntConfig(
            attn_dropouts=0.1,
            dropout=0.1,
            task="classification",
            batch_norm_continuous_input=True
        )
    elif args.model == "Catemb":
        model_config = CategoryEmbeddingModelConfig(
            layers='32-64-16',
            dropout=0.1,
            task="classification",
            batch_norm_continuous_input=True
        )
    elif args.model == "FTT":
        model_config = FTTransformerConfig(
            input_embed_dim=16,
            attn_feature_importance=False,
            task="classification",
            batch_norm_continuous_input=True
        )
    elif args.model == "Gate":
        model_config = GatedAdditiveTreeEnsembleConfig(
            gflu_stages=3,
            num_trees=10,
            task="classification",
            batch_norm_continuous_input=True
        )
    elif args.model == "Node":
        model_config = NodeConfig(
            num_layers=3,
            num_trees=10, #128
            input_dropout=0.1,
            initialize_selection_logits='normal',
            task="classification",
            batch_norm_continuous_input=True
        )
    elif args.model == "TabNet":
        model_config = TabNetModelConfig(
            task="classification",
            virtual_batch_size=32,
            batch_norm_continuous_input=True
        )
    elif args.model == "TabTrans":
        model_config = TabTransformerConfig(
            task="classification",
            out_ff_layers='64-32-16',
            batch_norm_continuous_input=True
        )
    else:
        sys.exit("This model does not exist!!")
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    return tabular_model


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="Simulated_SS7", type=str, choices=['Simulated_SS7', 'POST_SS7'])
    parser.add_argument("--output_dir", default="results", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs for training")

    args = parser.parse_args()
    os.makedirs(f"{args.output_dir}/{args.data_name}/logs", exist_ok=True)

    for args.exp_id in np.arange(100):
        train_data, test_data = obtain_data(args)
        for args.model in ["AutoInt", "Catemb", "FTT", "Gate", "Node", "TabNet", "TabTrans"]:
            os.makedirs(f"{args.output_dir}/{args.data_name}/saved_models/{args.model}", exist_ok=True)
            logger_name = f"{args.output_dir}/{args.data_name}/logs/{args.model}/{args.model}_{args.exp_id}.json"
            tabular_model = obtain_tabular_model(args)
            tabular_model.fit(train=train_data, validation=test_data)
            pred_df = tabular_model.predict(test_data)
            result = evaluate(test_data['label'], pred_df["prediction"])
            with open(logger_name, "w") as outfile:
                outfile.write(json.dumps(result))
            del logger_name
            del tabular_model
            del pred_df
            del result
            gc.collect()
        if args.exp_id == 0:
            sys.exit()


if __name__ == "__main__":
    main()
