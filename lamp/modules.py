import torch
from lamp.utils import get_default_device, get_default_dtype, get_dtype, get_device


class BaseModule(torch.nn.Module):
    def __init__(self):

        super().__init__()

    @property
    def num_trainable_parameters(self):
        return sum(
            (
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
        )

    @property
    def num_parameters(self):
        return sum((parameter.numel() for parameter in self.parameters()))

    def gradient_norm(self):

        return torch.mean(
            torch.cat(
                [
                    torch.norm(param.grad)
                    for param in self.parameters()
                    if param.grad is not None
                ]
            )
        )

    def gather(self, identifier, recursive=True):

        if recursive:
            modules = self.modules()
        else:
            modules = self.children()

        q = 0

        for module in modules:
            if hasattr(module, identifier):
                q += getattr(module, identifier)()

        return q

    def Eval(self):

        return EvalWrapper(self)

    def freeze(self):

        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):

        for param in self.parameters():
            param.requires_grad = True

    def load(self, path):

        self.load_state_dict(torch.load(path))
        self.train()

    def save(self, path):

        state = self.state()
        torch.save(state, path)

    def copy_values_from(self, module):

        r = self.load_state_dict(module.state_dict())

        if r.missing_keys or r.unexpected_keys:
            raise RuntimeError(
                "Copying values from one module to another did not behave as expected"
            )

    def state(self, *args, **kwargs):
        return self.state_dict(*args, **kwargs)

    def _to(self, **kwargs):

        try:
            dtype = kwargs.pop("dtype")

            if dtype is not None:
                dtype = get_dtype(dtype)

        except KeyError:
            dtype = None

        try:
            device = kwargs.pop("device")
            if device is not None:
                device = get_device(device)
        except KeyError:
            device = None

        if device is None:
            device = get_default_device()
        if dtype is None:
            dtype = get_default_dtype()

        self.to(dtype=dtype, device=device)


class EvalWrapper(object):
    def __init__(self, module):
        self._module = module
        self._train_orig_state = None

    def __enter__(self):
        self._train_orig_state = self._module.training
        self._module.eval()

    def __exit__(self, exception_type, exception_value, traceback):
        if self._train_orig_state:
            self._module.train()


class Flattening(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X.view(X.shape[0], -1)
