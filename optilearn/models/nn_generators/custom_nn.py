
from typing import Iterable, Optional

import torch
from torch.nn import Module

from optilearn.models.nn_generators.abstract_nn import AbstractNN
from optilearn.models.nn_generators.cnn import CNN
from optilearn.models.nn_generators.mlp import MLP
from optilearn.models.nn_generators.preference_attention import PreferenceAttention
from optilearn.models.utils.model_utils import (
    get_named_nn,
    get_model_output_shape,
)
from optilearn.models.utils.generator_utils import set_init


def key_in_dict_and_is_true(key_name: str, dictionary: dict):
    return (key_name in dictionary.keys()) and (dictionary[key_name])


class CustomNN(AbstractNN):
    """ """

    def __init__(
        self,
        dim_in: int,
        dim_p: int,
        dim_out: int,
        nn_config: dict,
        device="cpu",
    ):

        super(CustomNN, self).__init__(
            dim_in=dim_in,
            dim_out=dim_out,
            dim_p=dim_p,
            nn_config=nn_config,
        )
        self.dim_mlp_input: int = self.dim_in
        self.device = device

        self.network_list = []
        self.feature_extractor: Optional[Module] = None
        self.mlp: Module = Module()

        self._init_nn_(nn_config)

    def _init_nn_(self, nn_config):
        self._init_feature_extractor(nn_config)
        self._init_mlp(nn_config)

        self.to(self.device)

    def _init_feature_extractor(self, nn_config):
        if "feature_extractor" in nn_config.keys():
            fx_config = nn_config["feature_extractor"]

            if key_in_dict_and_is_true("pretrained_block", fx_config):
                self.feature_extractor = self._get_pretrained_layers(fx_config["pretrained_model_name"])

                dim_mlp_input = get_model_output_shape(
                    self.feature_extractor,
                    input_shape=(self.dim_in, *nn_config["input_size"]),
                )[1]

                if "fine_tune" in fx_config.keys():
                    # ToDo: Validate gradient flow

                    for parameter in self.feature_extractor.parameters():
                        parameter.requires_grad = fx_config["fine_tune"]

            elif "cnn" in fx_config:
                #logging.info("Initializing CNN with custom config.")
                self.feature_extractor = CNN(
                    dim_in=self.dim_in,
                    nn_config=fx_config["cnn"],
                    input_size=nn_config["input_size"],
                    device=self.device,
                )
                dim_mlp_input = self.feature_extractor.dim_out
                self.network_list += [self.feature_extractor]
            else:
                raise KeyError("Feature extractor expected but not defined in config.")

            self.dim_mlp_input = dim_mlp_input

        else:
            #logging.info("Skipping feature extractor")
            pass

    def _init_mlp(self, nn_config: dict):
        if "mlp" in nn_config.keys():
            mlp_layers = nn_config["mlp"]["mlp"]
        else:
            mlp_layers = []

        if key_in_dict_and_is_true("attention_head", nn_config["mlp"]):

            mlp = PreferenceAttention(
                dim_in=self.dim_mlp_input,
                dim_p=self.dim_p,
                dim_out=self.dim_out,
                nn_config=mlp_layers,
                device=self.device,
            )
            self.network_list += [mlp.mlp.model, mlp.attention.model, mlp.out.model]

        else:
            self.dim_mlp_input += self.dim_p

            #logging.info(f"Initializing MLP with {self.dim_mlp_input} inputs.")
            mlp = MLP(
                dim_in=self.dim_mlp_input,
                dim_out=self.dim_out,
                nn_config=mlp_layers,
                device=self.device,
            )
            self.network_list += [mlp.model]

        self.mlp = mlp

    def forward(self, x, pref: Optional[torch.Tensor]=None):
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
            x = torch.flatten(x, start_dim=1)

        x = self.mlp(x, pref=pref)

        return x

    def set_init(self):
        set_init(self.network_list)

    @staticmethod
    def _get_pretrained_layers(model_name):
        #logging.info("Loading pretrained block. CNN config will be ignored")

        pretrained_nn = get_named_nn(model_name)
        nn_layers = list(pretrained_nn.children())

        if isinstance(nn_layers[-1], Iterable):
            fc = nn_layers[-1].pop(-1)
        else:
            fc = nn_layers.pop(-1)
        assert isinstance(fc, torch.nn.Linear), f"<{model_name}> not yet supported"

        return pretrained_nn
