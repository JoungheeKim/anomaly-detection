from argparse import ArgumentParser
from torch.utils.data import DataLoader
import logging
import os
import torch
import random
import numpy as np
from torch import optim
from data_utils import TagDataset, ResultWriter
from torch.nn import functional as F
import torch as t
from tqdm import tqdm, trange
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
logger = logging.getLogger(__name__)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from shutil import copyfile

CHECKPOINT_NAME = 'pytorch_model.bin'
CONFIG_NAME = "training_args.bin"


## Json 파일 읽기.
def load_json(path):
    with open(path) as handle:
        return json.load(handle)

## SEED 설정
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

## 모델 구성
def build_model(args):
    model_type = args.model_type.lower()
    model = None
    if model_type == 'vae':
        from vae_model import VAutoEncoder
        model = VAutoEncoder(
                 input_dim=args.input_dim,
                 latent_dim=args.latent_dim,
                 window_size=args.window_size,
                 hidden_dims=load_json(args.hidden_json) if args.hidden_json else None)
    if model_type == 'ae':
        from ae_model import AutoEncoder
        model = AutoEncoder(
                 input_dim=args.input_dim,
                 latent_dim=args.latent_dim,
                 window_size=args.window_size,
                 hidden_dims=load_json(args.hidden_json) if args.hidden_json else None)
    if model_type == 'lstmae':
        from lstm_model import LSTMAutoEncoder
        model = LSTMAutoEncoder(
                 input_dim=args.input_dim,
                 latent_dim=args.latent_dim,
                 window_size=args.window_size,
                 num_layers=args.num_layers)
    return model

## 모델 불러오기
def load_saved_model(model, path):
    checkpoint = torch.load(os.path.join(path, CHECKPOINT_NAME))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
    return model

## 모델 저장
def save_model(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, os.path.join(path, CHECKPOINT_NAME))

def save_config(config, path):
    torch.save(config, os.path.join(path, CONFIG_NAME))

def load_saved_config(path):
    return torch.load(os.path.join(path, CONFIG_NAME))

def build_parser():
    parser = ArgumentParser()

    ## 모델 type
    parser.add_argument("--model_type", default="vae", type=str, help="vae, ae, 등 모델의 이름을 입력하세요.")
    
    ## 모델 설정
    parser.add_argument("--window_size", default=1, type=int, help="확장을 고려한 변수입니다.(현재 사용중이지 않음)")
    parser.add_argument("--input_dim", default=31, type=int, help="공정 Tag 갯수(변수갯수)를 의미합니다.")
    parser.add_argument("--latent_dim", default=4, type=int, help="최종 압축 차원 크기를 의미합니다.")
    parser.add_argument("--hidden_json", default=None, type=str, help="중간 압축 차원 크기를 의미하며 json 형태의 .txt 파일을 입력하세요.")
    parser.add_argument("--num_layers", default=2, type=int, help="LSTM AutoEncoder Layer 수를 의미하며 json 형태의 .txt 파일을 입력하세요.")

    ##GPU 사용할지 여부
    parser.add_argument("--no_cuda", action="store_true", help="GPU를 사용하지 않고 모델을 학습&테스트 하려면 true로 변경하세요")

    ## 데이터 위치
    parser.add_argument("--train_file", default="201907_202006/train.p", type=str, help='train에 사용할 pickle 데이터를 선택해 주세요.')
    parser.add_argument("--valid_file", default="201907_202006/val.p", type=str, help='validation에 사용할 pickle 데이터를 선택해 주세요')
    parser.add_argument("--test_file", default="201907_202006/test.p", type=str, help='test에 사용할 pickle 데이터를 선택해 주세요.')
    parser.add_argument("--norm_file", default=None, type=str, help='정규화를 사용하려면 각 tag의 mean, std를 포함한 .csv 파일을 입력하세요.')
    parser.add_argument("--do_train", action="store_true", help='학습하려면 true로 변경하세요')
    parser.add_argument("--do_eval", action="store_true", help='평가하려면 true로 변경하세요')

    ## 평가표
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default='experiments/experiment.csv',
        help="학습 및 평가한 내용에 대해 Summary를 제공하고 있습니다. Summary파일의 이름을 입력하세요.",
    )

    ## 저장폴더 위치
    parser.add_argument("--pre_trained_dir", type=str, help="이전에 학습한 모델이 저장되어 있는 폴더를 입력하세요")

    ## 저장폴더 위치
    parser.add_argument("--output_dir", default="results", type=str, help="학습 후 최종 결과물이 저장될 위치를 입력하세요.")
    ## 저장폴더에 새로운 파일 overwirte 여부
    parser.add_argument("--overwrite_output_dir", action="store_true", help="학습 후 최종 결과물이 저장될 위치에 파일이 있을 때 덮어쓰기를 사용하시려면 true로 변경하세요")
    
    ## 학습 설정
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=1024,
        type=int,
        help="평가에 사용하는 Batch 사이즈를 입력하세요.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=1024,
        type=int,
        help="학습에 사용하는 Batch 사이즈를 입력하세요.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument(
        "--weight_decay", default=0.01, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )

    config = parser.parse_args()
    return config


## 학습과정
def train(args, train_dataset, valid_dataset, model):
    """
    :param args: 모든 설정값이 포함되어 있는 객체
    :param train_dataset: 학습용 데이터(torch.dataset)
    :param valid_dataset: 평가용 데이터(torch.dataset)
    :param model: 학습할 모델
    :return:
    """


    ## Summay 객체 생성
    tb_writer = SummaryWriter()
    
    ## GPU 갯수에 따라 batch 조정
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    ## 데이터 불러오기
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.train_batch_size
    )
    
    ## 학습 step 조정
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    # Optimizer를 조정하기.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)

    ## apex를 사용시 설정
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running Sequence Classification training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    logger.info("  Starting Training.")

    tr_loss, logging_loss = 0.0, 0.0
    best_loss = 1e10
    best_mse = 1e10

    model.zero_grad()
    
    ## Train epoch 설정
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
    )
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration",
        )
        for step, batch in enumerate(epoch_iterator):
            
            ## batch 단위로 데이터 불러오기
            inputs, real_label = batch
            inputs = inputs.to(args.device)
            real_label = real_label.to(args.device)
            model.train()
            predict_values = model(inputs)
            
            ## loss 계산
            loss = model.loss_function(*predict_values, M_N=len(inputs))
            epoch_iterator.set_postfix_str('train_loss=%.4e' % (loss))

            ## backward & upgrade
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if (
                    args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics 저장
                    if (
                        args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, valid_dataset, model)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar(
                        "loss", (tr_loss - logging_loss) / args.logging_steps, global_step
                    )
                    logging_loss = tr_loss
                    
                    ## 모델 저장조건(loss가 가장 작은 것을 선택하여 최종 모델로 선정)
                    if results["loss"] < best_loss:
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        save_model(model_to_save, args.output_dir)
                        save_config(args, args.output_dir)

                        best_loss = results["loss"]
                        best_mse = results["mse"]

                    logger.info("***** best_mse : %.4f *****", best_mse)
                    logger.info("***** best_loss : %.4f *****", best_loss)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step, best_loss, best_mse

## 평가 단계
def evaluate(args, eval_dataset, model):

    ## GPU 갯수에 따른 Batch 조정
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    ## 평가용 데이터 불러오기
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, batch_size=args.eval_batch_size
    )

    # GPU가 여러개이면 DataParallel을 이용하기.
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    mse_list=[]
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        
        ## Batch 형태로 평가 데이터 불러오기
        inputs, real_values = batch
        inputs = inputs.to(args.device)
        real_values = real_values.to(args.device)

        with torch.no_grad():
            predict_values = model(inputs)
            loss = model.loss_function(*predict_values, M_N=len(inputs))
            eval_loss += loss.mean().item()
        nb_eval_steps += 1
        
        ## 전체 Loss 계산
        mse_list.append(float(loss))

    if eval_loss > 0:
        eval_loss = eval_loss / nb_eval_steps
    result = {
        "loss": eval_loss,
        "mse": np.mean(mse_list),
    }

    ## 결과를 로그로 저장하기.
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result



## MAIN 실행
def main():
    def _print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))

    ## 설정 불러오기.
    args = build_parser()
    _print_config(args)

    ## CPU setting
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    ## 유효성 검사
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "결과를 저장할 폴더에 ({}) 이미 다른 파일이 있습니다. 이 폴더에 저장하려면 overwrite_output_dir 설정을 true로 변경하시기 바랍니다.".format(
                args.output_dir
            )
        )
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if args.do_train:
        if args.train_file is None or args.valid_file is None:
            raise ValueError(
                "학습할 파일을 입력하세요."
            )

    if args.do_eval:
        if args.test_file is None:
            raise ValueError(
                "평가할 파일을 입력하시오."
            )

    if args.do_train is None and args.do_eval:
        if args.pre_trained_dir is None:
            raise ValueError(
                "평가만을 하려면 학습된 모델을 넣으세요."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        device,
        args.n_gpu,
        args.fp16,
    )

    # Set seed
    set_seed(args)

    ########################### Load Model ##########################
    logger.info("Load model %s", args.model_type)
    model = build_model(args)
    model.to(device)

    best_mse=None
    
    ## 학습 설정
    if args.do_train:
        ## 학습용 데이터 불러오기
        train_dataset = TagDataset(file_path=args.train_file,
                                   input_dim=args.input_dim,
                                   window_size=args.window_size,
                                   norm_path=args.norm_file)
        valid_dataset = TagDataset(file_path=args.valid_file,
                                   input_dim=args.input_dim,
                                   window_size=args.window_size,
                                   norm_path=args.norm_file)
        ## 학습하기
        global_step, tr_loss, best_loss, best_mse = train(args, train_dataset, valid_dataset, model)

        args.pre_trained_dir = args.output_dir
        logger.info(
            " global_step = %s, average loss = %s, best_loss  = %s, best_mse = %s",
            global_step,
            tr_loss,
            best_loss,
            best_mse,
        )

        ## json 파일 저장
        if args.hidden_json is not None:
            temp_path = os.path.join(args.output_dir, 'hidden_json.txt')
            copyfile(args.hidden_json, temp_path)

        ## json 파일 저장
        if args.norm_file is not None:
            temp_path = os.path.join(args.output_dir, 'norm.csv')
            copyfile(args.norm_file, temp_path)

    # 평가 설정
    if args.do_eval:

        ## 최종 학습된 모델 설정 불러오기
        saved_config = load_saved_config(args.pre_trained_dir)

        ## 학습된 모델 불러오기
        model = build_model(saved_config)
        model = load_saved_model(model, args.pre_trained_dir)
        model.to(device)

        ## 평가용 데이터 불러오기
        eval_dataset = TagDataset(file_path=args.test_file,
                                  input_dim=args.input_dim,
                                  window_size=args.window_size,
                                  norm_path=args.norm_file)
        
        ## 평가하기
        results = evaluate(args, eval_dataset, model)

        train_action = "Only Test"
        if args.do_train:
            train_action = "Train and Test"
        else:
            saved_config.experiments_dir = args.experiments_dir
            saved_config.test_file = args.test_file
        
        ## Summay 파일에 정보를 저장하기
        results.update(
            {
                'action': train_action,
                'valid_loss' : best_mse,
            }
        )

        writer = ResultWriter(args.experiments_dir)
        writer.update(saved_config, **results)



if __name__ == "__main__":
    main()