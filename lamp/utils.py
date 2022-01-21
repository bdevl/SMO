import torch
import numpy as np
from inspect import isfunction
import itertools
import math


def conv2d_calculate_output_size(conv, input_size):

    H_in = input_size[0]
    W_in = input_size[1]

    if isinstance(conv, list):
        return conv2d_calculate_output_size(
            conv, conv2d_calculate_output_size(conv.pop(0), input_size)
        )

    assert isinstance(conv, torch.nn.Conv2d)
    assert isinstance(input_size, tuple) and len(input_size) == 2

    H_out = (
        H_in + 2 * conv.padding[0] - conv.dilation[0] * (op.kernel_size[0] - 1) - 1
    ) / conv.stride[0] + 1
    W_out = (
        W_in + 2 * conv.padding[1] - conv.dilation[1] * (op.kernel_size[1] - 1) - 1
    ) / conv.stride[1] + 1

    H_out_int = int(H_out)
    W_out_int = int(W_out)

    assert H_out_int - H_out == 0
    assert W_out_int - W_out == 0

    return (H_out_int, W_out_int)


def fetch_device(cuda_device=0) -> torch.device:

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(cuda_device))
    else:
        device = torch.device("cpu")

    return device


@torch.no_grad()
def batch_execution(
    fcall, data, batch_size=256, batch_dim=0, force_equal_chunks=False
) -> torch.Tensor:

    # breaks up large dataset into chunks of size batch_size and passes them to fcall separately
    if force_equal_chunks:
        if data.shape[batch_dim] % batch_size != 0:
            raise ValueError(
                "The batch size {} does not enable to split up {} data points into equal chunks".format(
                    batch_size, data.shape[batch_dim]
                )
            )

    return torch.cat(
        tuple([fcall(data_) for data_ in torch.split(data, batch_size, dim=batch_dim)]),
        batch_dim,
    )


class CaptureOutput(object):
    def __init__(self):

        self._layers = dict()
        self._output = dict()

    def register_layer(self, identifier, layer):

        self._layers[identifier] = layer
        self._register_hook(identifier, layer)

    def __getitem__(self, item):
        return self._output[item]

    def __call__(self):
        if len(self._layers) == 1:
            return self._output[list(self._output.keys())[0]]
        else:
            raise Exception(
                "Only works if there has exactly one output been registered"
            )

    def _register_hook(self, identifier, layer):
        def hook(model, intput, output):
            self._output[identifier] = output.detach()

        self._layer[identifier].register_forward_hook(hook)

    def __repr__(self):

        s = "Output capture has been registered for the following layers: \n"
        for keyname in self._output.keys():
            s += "Layer: {} \n".format(keyname)

        return s


class ParameterGroup(object):
    def __init__(self, *args, label=None):

        self._label = label
        self._args = args

    @property
    def label(self):
        return self._label

    @property
    def params(self):
        return itertools.chain(*self._args)

    def __call__(self):
        return self.params

    def add_parameters(self, params):

        try:
            params.__iter__
        except AttributeError:
            raise ValueError("the parameters are not iterable")

        self._args.append(params)

    def __repr__(self):
        return "Parameter Group | Label : {}".format(self.label)


def combined_parameters(*args):

    return itertools.chain(*args)


def bimv(A, x):

    assert A.dim() == x.dim() == 2, "Dimension misfit"
    return torch.matmul(A.unsqueeze(0), x.unsqueeze(2)).squeeze(2)


def coefficient_of_determination(y_pred, y, global_average=False):

    assert y_pred.shape == y.shape

    if y_pred.dim() > 2:
        batch_size = y_pred.shape[0]
        y_pred = y_pred.view(batch_size, -1)
        y = y.view(batch_size, -1)

    if global_average:
        # this is the implementation from torch.ignite
        e = torch.sum((y - y_pred) ** 2) / torch.sum((y - y.mean()) ** 2)
        return 1 - e.item()
    else:
        # component-wise mean
        assert y_pred.shape[0] > 0
        e = torch.sum((y - y_pred) ** 2, 0) / torch.sum((y - y.mean(0)) ** 2, 0)
        return (1 - e).mean().item()


def get_default_dtype():
    return torch.float32


def get_default_device():
    return torch.device("cpu")


def sparse_matrix_batched_vector_multiplication(matrix, vector_batch):

    # code from https://github.com/pytorch/pytorch/issues/14489
    batch_size = vector_batch.shape[0]
    vectors = vector_batch.transpose(0, 1).reshape(-1, batch_size)
    return matrix.mm(vectors).transpose(1, 0).reshape(batch_size, -1)


def architecture_from_linear_decay(
    dim_in, dim_out, num_hidden_layers, append_input_output_dim_to_architecture=False
):

    architecture = list(np.linspace(dim_in, dim_out, num_hidden_layers + 2).astype(int))

    if append_input_output_dim_to_architecture:
        architecture = [dim_in] + architecture + [dim_out]

    return architecture


def get_device(device):

    if isinstance(device, torch.device):
        return device

    if device.lower() == "cpu":
        return torch.device("cpu")
    elif device.lower() in ["gpu", "cuda"]:
        return torch.device("cuda:0")
    else:
        raise ValueError("device not recognized in get_device()")


def get_dtype(dtype):

    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        if dtype.lower() == "float32":
            return torch.float32
        elif dtype.lower() in ["float64", "double"]:
            return torch.float64
        else:
            raise ValueError(
                "Supplied invalid string as dtype. Must be either float32, float64 or double"
            )
    else:
        raise ValueError("Supplied invalid and unknown dtype argument.")


def get_activation_function(activ, module=True, force_string=False):

    if force_string and not isinstance(activ, str):
        raise ValueError(
            "An actual activation function instead of an ID string was passed, with force_string set to true"
        )

    if isinstance(activ, torch.nn.Module):
        return activ
    elif isfunction(activ):
        return activ
    elif isinstance(activ, str):
        if activ.lower() == "relu":
            if module:
                return torch.nn.ReLU()
            else:
                return torch.nn.functional.ReLU
        else:
            raise ValueError(
                "Supplied invalid string for activation function (does not match any known preset)"
            )
    else:
        raise ValueError(
            "Supplied invalid and unknwon argument for activation function"
        )


def DiagonalGaussianLogLikelihood(
    target, mean, logvars, target_logvars=None, reduce=torch.sum
):

    if target_logvars is None:

        sigma = logvars.mul(0.5).exp_()
        part1 = logvars
        part2 = ((target - mean) / sigma) ** 2
        log2pi = 1.8378770664093453
        L = -0.5 * (part1 + part2 + log2pi)
        if reduce is not None:
            L = reduce(L)
        return L

    else:
        raise NotImplementedError


def reparametrize(mean, logsigma):

    std = torch.exp(logsigma)
    return mean + std * torch.randn_like(std)


def relative_error(y, y_true):
    return (torch.norm(y - y_true) / torch.norm(y_true)).item()


def relative_error_batched(Y_mean, Y_true):
    return torch.mean(
        (
            torch.sqrt(torch.sum((Y_mean - Y_true) ** 2, 1))
            / torch.sqrt(torch.sum(Y_true ** 2, 1))
        )
    ).item()


def UnitGaussianKullbackLeiblerDivergence(
    mean: torch.Tensor, logvars: torch.Tensor
) -> torch.Tensor:

    KL = -0.5 * torch.sum(1 + logvars - mean.pow(2) - logvars.exp())
    return KL


def annotate_dataset_with_indices(cls):

    """
    EXTERNAL CODE: SOURCE: https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(
        cls.__name__,
        (cls,),
        {
            "__getitem__": __getitem__,
        },
    )
