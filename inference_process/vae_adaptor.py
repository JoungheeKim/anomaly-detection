from base import Inference_skeleton
import os
import torch
import json
import pandas as pd
import pickle
import numpy as np
from torch import nn

class AutoEncoder_Inference(Inference_skeleton):

    def __init__(self, args:dict) -> None:
        super(AutoEncoder_Inference, self).__init__(args)

        ## dict를 object로 변경(편의상)
        args = self.convert_to_object(args)

        ## gpu 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

        ## 최종 학습된 모델 설정 불러오기
        saved_config = load_saved_config(args.model_path)
        self.saved_config = saved_config

        # Hidden dim 불러오기
        hidden = load_hidden(args.model_path)

        ## 학습된 모델 불러오기
        model = build_model(saved_config, hidden)
        model = load_saved_model(model, args.model_path)
        model.to(self.device)
        model.eval()
        self.model = model

        ## 정규화 여부 확인
        self.norm_df = load_norm(args.model_path)

        self.loss_fn = nn.MSELoss(reduction='none')



    ## 결과를 출력
    def get_output(self, input: list) -> tuple:
        """
        :param input: 변수 list가 들어와야 합니다. model의 변수 갯수와 일치해야 합니다.
        :return:
        """

        ## 변수갯수가 맞는지 Validation Check
        assert self.model.input_dim == len(input), '모델의 변수 갯수[{}]와 input의 길이[{}]가 일치해야 합니다.'.format(self.model.input_dim, len(input))

        ## numpy 로 전환
        input = np.array(input)

        ## 정규화
        if self.norm_df is not None:
            temp_means = self.norm_df['mean'].values[:len(input)]
            temp_stds = self.norm_df['std'].values[:len(input)]
            input = (input-temp_means)/temp_stds

        ## torch로 전환
        input = torch.tensor(input, dtype=torch.float).to(self.device)
        input = input.view(1, -1)

        with torch.no_grad():
            predict_values = self.model(input)

        recons = predict_values[0].cpu()
        inputs = predict_values[1].cpu()

        ## 변수별 confidence score
        confidence = self.loss_fn(recons, inputs)

        ## 모델 confidence
        confidence_score = torch.sum(confidence, dim=1)

        ## numpy 형태로 변환
        confidence = confidence.cpu().numpy().squeeze().tolist()
        confidence_score = confidence_score.cpu().numpy().squeeze().item()

        return confidence_score, confidence

    ## 설정파일에 반드시 포함해야 할 옵션들이 있는지 확인하는 과정
    def validation_check(self, args) -> bool:
        check_list = ['model_path', 'no_cuda']
        for check_item in check_list:
            assert hasattr(args, check_item), "반드시 {} 옵션을 포함하고 있어야 합니다.".format(check_item)

        assert os.path.isdir(args.model_path), '모델 관련 파일이 없습니다. 다시 확인하세요 [{}]'.format(args.model_path)
        return True


CHECKPOINT_NAME = 'pytorch_model.bin'
CONFIG_NAME = "training_args.bin"
JSON_NAME = 'hidden.txt'
NORM_NAME = 'norm.csv'

def save_config(config, path):
    torch.save(config, os.path.join(path, CONFIG_NAME))

def load_saved_config(path):
    return torch.load(os.path.join(path, CONFIG_NAME))

def load_hidden(path):
    json_path = os.path.join(path, JSON_NAME)
    if not os.path.isfile(json_path):
        return None
    return load_json(json_path)

def load_norm(path):
    norm_path = os.path.join(path, NORM_NAME)
    if not os.path.isfile(norm_path):
        return None

    ## 파일 불러오기
    if '.csv' in norm_path:
        norm_df = pd.read_csv(norm_path)
    else:
        norm_df = load_pickle(norm_path)

    norm_df.columns = ['tag_name', 'mean', 'std']

    ## 파일 유효성 확인(Nan 확인)
    assert not norm_df.isnull().values.any(), 'Dataframe 내 NaN이 포함되어 있습니다.'

    return norm_df


## 모델 구성
def build_model(args, hidden_dims=None):
    model_type = args.model_type.lower()
    model = None
    if model_type == 'vae':
        from vae_model import VAutoEncoder
        model = VAutoEncoder(
                 input_dim=args.input_dim,
                 latent_dim=args.latent_dim,
                 window_size=args.window_size,
                 hidden_dims=hidden_dims if hidden_dims else None)
    if model_type == 'ae':
        from ae_model import AutoEncoder
        model = AutoEncoder(
                 input_dim=args.input_dim,
                 latent_dim=args.latent_dim,
                 window_size=args.window_size,
                 hidden_dims=hidden_dims if hidden_dims else None)

    return model

## 모델 불러오기
def load_saved_model(model, path):
    checkpoint = torch.load(os.path.join(path, CHECKPOINT_NAME))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
    return model

## Json 파일 읽기.
def load_json(path):
    with open(path) as handle:
        return json.load(handle)

## pickle 형태의 데이터를 불러오기
def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)