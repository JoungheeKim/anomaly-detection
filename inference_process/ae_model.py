import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')

## AE 모델
class AutoEncoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 window_size: int=1,
                 **kwargs) -> None:
        """
        :param input_dim: 변수 Tag 갯수
        :param latent_dim: 최종 압축할 차원 크기
        :param hidden_dims: 압축 중간 단계 차원 크기
        :param window_size: 다른 모델 및 확장을 고려한 임시변수
        :param kwargs:
        """

        super(AutoEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim*window_size

        modules = []
        if hidden_dims is None:
            hidden_dims = [16, 24]

        # Build Encoder
        temp_dim = self.input_dim
        for h_dim in reversed(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Linear(temp_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.1))
            )
            temp_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_output = nn.Linear(hidden_dims[0], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0])


        temp_dim = hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(temp_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.1))
            )
            temp_dim = h_dim

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(temp_dim, self.input_dim),
            )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x D]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = self.encoder_output(result)
        return result

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x D]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        batch_size = input.size()[0]

        input = input.reshape(batch_size, -1)
        z = self.encode(input)
        return [self.decode(z), input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]

        loss =F.mse_loss(recons, input)
        return loss