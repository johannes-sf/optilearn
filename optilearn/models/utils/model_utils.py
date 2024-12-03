import logging

import torch
import torchvision


def get_model_output_shape(model, input_shape):
    device = next(model.parameters()).device

    # In case of any batch-norm layers in the model,
    # the number of samples in a batch must be > 1
    x = torch.zeros(100, *input_shape).to(device)

    with torch.no_grad():
        x = model.forward(x)

    if not isinstance(x, torch.Tensor):
        x = x[0]

    return x.shape


def get_named_nn(model_name: str):
    try:
        logging.info(f"Attempting to load model: <{model_name}>.")
        model = torchvision.models.get_model(
            name=model_name,
            weights=torchvision.models.get_model_weights(model_name).DEFAULT,
        )
    except ValueError:
        model = torchvision.models.get_model(name=model_name)
        logging.warning(
            f"No weights available for model <{model_name}>. " f"Loading untrained model.",
        )

    logging.info(f"Loaded <{model_name}> successfully.")

    return model


def get_layer_n_features(linear_layer: torch.nn.Linear):
    assert isinstance(linear_layer, torch.nn.Linear), "Layer is not an instance of nn.Linear."
    fc_in_features = linear_layer.in_features
    fc_out_features = linear_layer.out_features

    return fc_in_features, fc_out_features
