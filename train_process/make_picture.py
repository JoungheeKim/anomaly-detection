import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import random
from argparse import ArgumentParser
from train import (
    build_model,
    load_saved_config,
    load_saved_model,
    set_seed,
)
from data_utils import TagDataset, save_pickle
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.nn import functional as F
from torch import nn

def build_parser():
    parser = ArgumentParser()

    ##GPU 사용할지 여부
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    ## 데이터 위치
    parser.add_argument("--test_file", type=str)

    ## 저장폴더 위치
    parser.add_argument("--pre_trained_dir", type=str, required=True)

    ## 저장폴더 위치
    parser.add_argument("--picture_dir", default="pictures", type=str)

    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=1024,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )

    config = parser.parse_args()
    return config

def evaluate(args, eval_dataset, model):

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, batch_size=args.eval_batch_size
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    label_list = []

    sum_mse_list = []
    mse_list = []

    sum_mae_list=[]
    mae_list = []

    mse_fn = nn.MSELoss(reduction='none')
    mae_fn = nn.L1Loss(reduction='none')

    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        inputs, labels = batch
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            predict_values = model(inputs)
            loss = model.loss_function(*predict_values, M_N=len(inputs))
            eval_loss += loss.mean().item()
        nb_eval_steps += 1

        labels = labels.detach().cpu()

        recons = predict_values[0].cpu()
        inputs = predict_values[1].cpu()

        if len(recons.size())>2:
            temp_loss = mse_fn(recons, inputs)
            mse = torch.sum(temp_loss, dim=1)
            sum_mse = torch.sum(mse, dim=1)

            temp_loss = mae_fn(recons, inputs)
            mae = torch.sum(temp_loss, dim=1)
            sum_mae = torch.sum(mae, dim=1)

            labels = labels[:,-1]
        else:
            mse = mse_fn(recons, inputs)
            sum_mse = torch.sum(mse, dim=1)

            mae = mae_fn(recons, inputs)
            sum_mae = torch.sum(mae, dim=1)

        label_list.append(labels)

        ## MSE
        sum_mse_list.append(sum_mse)
        mse_list.append(mse)
        ## MAE
        sum_mae_list.append(sum_mae)
        mae_list.append(mae)

    mse_list = np.concatenate(mse_list, axis=0)
    sum_mse_list = np.concatenate(sum_mse_list, axis=0)

    mae_list = np.concatenate(mae_list, axis=0)
    sum_mae_list = np.concatenate(sum_mae_list, axis=0)
    
    ## 저장하기
    data_df = eval_dataset.df
    selected_column = eval_dataset.selected_column

    data_df['mse'] = sum_mse_list
    data_df['mae'] = sum_mae_list

    data_df[['mse_{}'.format(item) for item in selected_column]] = mse_list
    data_df[['mae_{}'.format(item) for item in selected_column]] = mae_list

    save_path = os.path.join(args.picture_dir, 'summary.p')
    save_pickle(save_path, data_df)

    eval_loss = eval_loss / nb_eval_steps
    result = {
        "loss": eval_loss,
    }

    return result

def main():
    def _print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))

    ## 설정 불러오기.
    args = build_parser()
    _print_config(args)

    ##저장 폴더 만들기
    if not os.path.isdir(args.picture_dir):
        os.mkdir(args.picture_dir)

    config = load_saved_config(args.pre_trained_dir)

    ## Device setting
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()
    config.device = device
    config.n_gpu = torch.cuda.device_count()
    config.picture_dir = args.picture_dir
    config.per_gpu_eval_batch_size = args.per_gpu_eval_batch_size

    ## set seed
    set_seed(config)

    ## Load model
    model = build_model(config)
    model = load_saved_model(model, args.pre_trained_dir)
    model.to(device)

    eval_dataset = TagDataset(file_path=args.test_file if args.test_file is not None else config.test_file,
                              input_dim=config.input_dim,
                              window_size=config.window_size,
                              norm_path=config.norm_file)

    results = evaluate(config, eval_dataset, model)
























if __name__ == "__main__":
    main()


