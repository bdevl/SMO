from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, TensorDataset
from lamp.metrics import CoefficientOfDetermination, IndividualR2
from lamp.modules import BaseModule, Flattening
from lamp.utils import DiagonalGaussianLogLikelihood
from lamp.utils import reparametrize


class DiscriminativeModel(BaseModule):
    def __init__(self):
        super().__init__()


class HomoscedasticLinear(BaseModule):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._lin = torch.nn.Linear(dim_in, dim_out)
        self._logsigma = torch.nn.Parameter(torch.ones(dim_out))
        self._dim_out = (dim_out,)

    def forward(self, x):
        batch_dim = x.shape[0]
        a = self._lin(x)
        b = self._logsigma.expand((batch_dim,) + self._dim_out)

        return a, b


class ForkedLinear(BaseModule):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._lina = torch.nn.Linear(dim_in, dim_out)
        self._linb = torch.nn.Linear(dim_in, dim_out)

    def forward(self, x):
        a = self._lina(x)
        b = self._linb(x)

        return a, b


class DiscriminativeCNN(DiscriminativeModel):
    def __init__(self, **kwargs):

        super().__init__()

        options = {
            "nx": None,
            "filters": None,
            "dropout": None,
            "dim_out": None,
            "homoscedastic": False,
            "num_additional_full_layers": 0,
            "kernel_size": 3,
            "make_deterministic": False,
            "use_phi_dim": False,
            "pooling_type": "avg",
            "dim_hidden": 30,
            "gate": False,
            "padding": 1,
            "cnn_bias": True,
            "activation": "leakyrelu",
        }

        assert set(kwargs).issubset(
            options
        ), "Provided keywarg argument does not match template options"
        options.update(kwargs)
        _valid = True
        for key, value in options.items():
            if value is None:
                warnings.warn(
                    "The required key : {} for discriminative CNN was not supplied in the **kwargs.".format(
                        kwargs
                    )
                )
                _valid = False
        assert _valid, "Constructor failed."
        assert options["nx"] in [
            32,
            64,
            128,
        ], "The spatial resolution is neither 32, 64 nor 128."
        assert options["pooling_type"] in [
            "avg",
            "max",
        ], "The pooling type must either be 'avg' or 'max' (provided: {})".format(
            options["pooling_type"]
        )
        assert options["activation"] in [
            "leakyrelu",
            "relu",
        ], "The activation function {} cannot be created".format(options["activation"])

        self._options = options
        self._cnn_dim = None
        self.L = None
        self.L2 = None

        self.construct()

    @classmethod
    def Reconstruct(cls, options: dict, state_dict: OrderedDict, *, training=None):

        assert training is not None

        discriminative = cls(options)
        discriminative.load_state_dict(state_dict)

        if training:
            discriminative.train()
        else:
            discriminative.eval()

        return discriminative

    def construct(self):

        L = torch.nn.Sequential()
        L2 = torch.nn.Sequential()
        filters = self._options["filters"]
        channels = [1] + self._options["filters"][:-1]
        kernel_size = self._options["kernel_size"]
        padding = self._options["padding"]
        cnn_bias = self._options["cnn_bias"]
        pooling_type = self._options["pooling_type"]
        dropout = self._options["dropout"]
        nx = self._options["nx"]
        use_phi_dim = self._options["use_phi_dim"]
        num_additional_full_layers = self._options["num_additional_full_layers"]
        dim_hidden = self._options["dim_hidden"]

        def activation():
            if self._options["activation"] == "leakyrelu":
                return torch.nn.LeakyReLU()
            elif self._options["activation"] == "relu":
                return torch.nn.ReLU()
            else:
                raise ValueError(
                    "The activation function {} cannot be constructed.".format(
                        self._options["activation"]
                    )
                )

        for n, filter_size in enumerate(filters):
            L.add_module(
                "cnn{}".format(n),
                torch.nn.Conv2d(
                    channels[n],
                    filter_size,
                    kernel_size=(kernel_size, kernel_size),
                    padding=padding,
                    bias=cnn_bias,
                    groups=1,
                ),
            )
            L.add_module("relu{}".format(n), activation())

            if pooling_type == "max":
                L.add_module("maxpool{}".format(n), torch.nn.MaxPool2d(2, 2))
            elif pooling_type == "avg":
                L.add_module("avgpool{}".format(n), torch.nn.AvgPool2d(2, 2))
            else:
                ValueError("Pooling type {} unknown".format(pooling_type))

        L.add_module("Flatten", Flattening())

        if dropout:
            assert isinstance(dropout, float) and 0 < dropout < 1
            L.add_module("dropout", torch.nn.Dropout(dropout))

        dim = int(filters[-1] * (nx / 2 ** len(filters)) ** 2)
        self._cnn_dim = dim

        if use_phi_dim:
            assert isinstance(use_phi_dim, int)
            dim += use_phi_dim

        for n in range(num_additional_full_layers):
            L2.add_module("flinear{}".format(n), torch.nn.Linear(dim, dim_hidden))
            dim = dim_hidden
            if self._options["gate"]:
                L2.add_module("batchnorm", torch.nn.BatchNorm1d(dim))
            L2.add_module("frelu{}".format(n), activation())

        if self._options["homoscedastic"] and not self._options["make_deterministic"]:
            L2.add_module("linear", HomoscedasticLinear(dim, self._options["dim_out"]))
        elif self._options["make_deterministic"]:
            L2.add_module(
                "linear", torch.nn.Linear(dim, self._options["dim_out"], bias=True)
            )
        else:
            L2.add_module("linear", ForkedLinear(dim, self._options["dim_out"]))

        self._L = L
        self._L2 = L2

    @property
    def options(self):
        return self._options

    @property
    def _deterministic(self):
        assert "make_deterministic" in self._options
        return self._options["make_deterministic"]

    @property
    def use_phi(self):
        return bool(self._options["use_phi_dim"])

    def reset_parameters(self):

        for L in self._L:
            try:
                L.reset_parameters()
            except:
                pass

        for L in self._L2:
            try:
                L.reset_parameters()
            except:
                pass

    @property
    def num_parameters_cnn(self):

        cnn_modules = [l[1] for l in self._L.named_children() if "cnn" in l[0]]
        num_params = list()
        for module in cnn_modules:
            num_params.append(
                sum([parameter.numel() for parameter in module.parameters()])
            )

        return sum(num_params)

    @property
    def dtype(self):
        return list(self.parameters())[0][0].dtype

    @property
    def device(self):
        return list(self.parameters())[0][0].device

    def forward(self, inp, phi=None):

        if phi is None:
            return self._L2(self._L(inp))
        else:
            return self._L2(torch.cat((self._L(inp), phi), 1))

    def propagate(self, *args, **kwargs):

        assert (
            not self._deterministic
        ), "cannot propagate samples for a deterministic discriminative model"
        mu, logsigma = self(*args, **kwargs)
        return reparametrize(mu, logsigma)


class DiscriminativeTrainer(object):
    def __init__(
        self,
        model,
        data,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-3,
        make_deterministic=False,
        shuffle=False,
    ):

        self.model = model
        self.data = data

        assert self.data["Y"].ndim > 1
        assert self.data["Y_val"].ndim > 1

        if "phi" in self.data:
            assert self.data["phi"].ndim > 1

        if model.use_phi:
            assert "phi" in self.data

        self._shuffle = shuffle
        self._dataloader = self._create_dataloader(model, batch_size)
        self._optim = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self._make_deterministic = make_deterministic

        self.monitor = dict()
        self.monitor["r2_val"] = list()
        self.monitor["ls_val"] = list()
        self.monitor["J"] = list()

    def _create_dataloader(self, model, batch_size):

        if model.use_phi:
            dataset = TensorDataset(self.data["X"], self.data["Y"], self.data["phi"])
        else:
            dataset = TensorDataset(self.data["X"], self.data["Y"])

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=self._shuffle)

        return dataloader

    @classmethod
    def FromHomogenizationData(
        cls, model, data_tr, data_val, dtype, device, dtransform=None, **kwargs
    ):

        data = dict()

        if "use_binary" in kwargs:
            raise DeprecationWarning

        assert dtransform is not None

        with torch.no_grad():
            if "X_cnn" not in data_tr:
                data["X"] = dtransform(data_tr["X_g"]).to(dtype=dtype, device=device)
            else:
                data["X"] = data_tr["X_cnn"].to(dtype=dtype, device=device)

            if "X_cnn" not in data_val:
                data["X_val"] = dtransform(data_val["X_g"]).to(
                    dtype=dtype, device=device
                )
            else:
                data["X_val"] = data_val["X_cnn"].to(dtype=dtype, device=device)

        data["Y"] = data_tr["kappa"].to(dtype=dtype, device=device)
        data["Y_val"] = data_val["kappa"].to(dtype=dtype, device=device)

        if "phi" in data_tr:
            data["phi"] = data_tr["phi"].to(dtype=dtype, device=device)
        if "phi" in data_val:
            data["phi_val"] = data_val["phi"].to(dtype=dtype, device=device)

        return cls(model, data, **kwargs)

    def assess(self, X, Y, *, AcknowledgeTransformed=False):

        if not AcknowledgeTransformed:
            raise ValueError(
                "The X inputs must already have been transformed (according to dtransform()"
            )

        if self.model.training:
            raise RuntimeError("The model is not in eval mode")

        with torch.no_grad():
            Y_mean, Y_logsigma = self.model(X)

        cod = CoefficientOfDetermination()
        with torch.no_grad():
            R2 = cod(Y_mean, Y)
            logscore = DiagonalGaussianLogLikelihood(
                target=Y, mean=Y_mean, logvars=2 * Y_logsigma, reduce=torch.mean
            ).item()

        var_global = torch.var(Y).item()
        var_explained = torch.mean(torch.exp(2 * Y_logsigma)).item()

        normalized = (
            ((Y_mean - Y) / torch.exp(Y_logsigma)).detach().cpu().numpy().flatten()
        )
        num_stddevs = torch.abs((Y_mean - Y)) / torch.exp(Y_logsigma)

        return R2, logscore, var_global, var_explained, num_stddevs, normalized

    def r2_separate(self):

        with torch.no_grad():
            if self.model.use_phi or self._make_deterministic:
                raise NotImplementedError

            Y_mean, Y_logsigma = self.model(self.data["X_val"])
            r2 = IndividualR2(Y_mean, self.data["Y_val"])

        return r2

    def _validation(self, eval_mode=True):

        if eval_mode:
            self.model.eval()
        cod = CoefficientOfDetermination()

        with torch.no_grad():

            if self._make_deterministic:
                if self.model.use_phi:
                    Y_mean = self.model(self.data["X_val"], self.data["phi_val"])
                else:
                    Y_mean = self.model(self.data["X_val"])
                r2 = cod(Y_mean, self.data["Y_val"])
                self.monitor["r2_val"].append(r2)
                ls_val = 0
            else:

                if self.model.use_phi:
                    Y_mean, Y_logsigma = self.model(
                        self.data["X_val"], self.data["phi_val"]
                    )
                else:
                    Y_mean, Y_logsigma = self.model(self.data["X_val"])

                r2 = cod(Y_mean, self.data["Y_val"])
                self.monitor["r2_val"].append(r2)

                ls_val = DiagonalGaussianLogLikelihood(
                    target=self.data["Y_val"],
                    mean=Y_mean,
                    logvars=2 * Y_logsigma,
                    reduce=torch.mean,
                ).item()
                self.monitor["ls_val"].append(ls_val)

        self.model.train()

        return r2, ls_val

    def standard_deviation_val(self):

        #
        self.model.eval()
        with torch.no_grad():
            Y_mean, Y_logsigma = self.model(self.data["X_val"])
            D = torch.abs(Y_mean - self.data["Y_val"]) / torch.exp(Y_logsigma)

        return D

    def train(self, num_epochs, verbose=True, output_interval=1):

        self.model.train()

        if self._make_deterministic:
            lossfct = torch.nn.MSELoss()

        for epoch in range(num_epochs):

            J_loc = 0
            N_loc = 0

            for data in self._dataloader:

                if self.model.use_phi:
                    X_, Y_, phi_ = data
                else:
                    X_, Y_ = data

                self._optim.zero_grad()

                if self._make_deterministic:
                    if self.model.use_phi:
                        Y_mean = self.model(X_.detach(), phi=phi_.detach())
                    else:
                        Y_mean = self.model(X_.detach())

                    J = lossfct(Y_mean, Y_)
                else:
                    if self.model.use_phi:
                        Y_mean, Y_logsigma = self.model(X_.detach(), phi=phi_.detach())
                    else:
                        Y_mean, Y_logsigma = self.model(X_.detach())

                    J = -DiagonalGaussianLogLikelihood(
                        Y_.flatten().detach(),
                        Y_mean.flatten(),
                        2 * Y_logsigma.flatten(),
                        reduce=torch.mean,
                    )

                J.backward()
                self._optim.step()

                J_loc += J.item()
                N_loc += 1

            self.monitor["J"].append(J_loc / N_loc)

            if not self._make_deterministic:
                r2_val, ls_val = self._validation()
            else:
                r2_val, ls_val = self._validation()

            if verbose and (((epoch + 1) % output_interval == 0) or epoch == 0):
                print(
                    "Epoch {} / {}   ---Avg. Loss: {:.3f} | Val. R2: {:.3f} | Val. LS: {:.3f}".format(
                        epoch + 1, num_epochs, self.monitor["J"][-1], r2_val, ls_val
                    )
                )
