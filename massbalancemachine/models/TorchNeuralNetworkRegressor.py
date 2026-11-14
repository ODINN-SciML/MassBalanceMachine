import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml
import copy

from data_processing.Dataset import Normalizer

# Hidden-layer activations of the precipitation correction networks (bias_cor and
# bias_cor_elev), chosen per network by its `activation` key in the model config
ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}


def _activation(netParams, name):
    """Activation class of the network `name` of the model config: `activation`, "relu"
    (default) or "leaky_relu". With ReLU, the units of the small last hidden layer can all
    die, which leaves the correction constant."""
    key = netParams.get("activation", "relu")
    assert (
        key in ACTIVATIONS
    ), f"{name}.activation must be one of {sorted(ACTIVATIONS)}, got {key!r}."
    return ACTIVATIONS[key]


class FourierFeatures(nn.Module):
    """Random Fourier features of the normalized inputs of a network (Tancik et al.,
    2020): x -> [x, sin(2 pi x B), cos(2 pi x B)], with B a fixed (not trained) matrix of
    n_frequencies columns drawn from N(0, sigma^2) with `seed`. sigma is in cycles per unit
    of normalized input: a feature normalized over [lower, upper] is resolved down to
    about (upper - lower) / sigma, so that a larger sigma lets the network that follows
    vary over shorter distances. B is a buffer, saved with the checkpoint."""

    def __init__(self, n_inputs, n_frequencies, sigma, seed=0, include_input=True):
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed))
        self.register_buffer(
            "B",
            float(sigma) * torch.randn(n_inputs, n_frequencies, generator=generator),
        )
        self.include_input = include_input
        self.n_out = 2 * n_frequencies + (n_inputs if include_input else 0)

    def forward(self, x):
        projection = 2 * np.pi * x @ self.B
        features = [torch.sin(projection), torch.cos(projection)]
        if self.include_input:
            features.insert(0, x)
        return torch.cat(features, dim=1)


FOURIER_FEATURES_KEYS = {"n_frequencies", "sigma", "seed", "include_input"}


def _input_encoding(netParams, n_inputs, name):
    """Layers that encode the inputs of the network `name` before its first linear
    layer, and the number of features they give. Optional `fourier_features` of the
    network config (`FourierFeatures`, keys n_frequencies and sigma, and seed (0) and
    include_input (true)); without it, none: the network reads its inputs as they are.
    """
    options = netParams.get("fourier_features")
    if options is None:
        return [], n_inputs
    unknown = set(options) - FOURIER_FEATURES_KEYS
    assert not unknown, (
        f"Unknown {name}.fourier_features options {sorted(unknown)}, expected some of "
        f"{sorted(FOURIER_FEATURES_KEYS)}."
    )
    for key in ("n_frequencies", "sigma"):
        assert key in options, f"{name}.fourier_features needs {key}."
    assert (
        int(options["n_frequencies"]) > 0 and float(options["sigma"]) > 0
    ), f"{name}.fourier_features needs n_frequencies > 0 and sigma > 0."
    encoding = FourierFeatures(
        n_inputs,
        int(options["n_frequencies"]),
        float(options["sigma"]),
        seed=options.get("seed", 0),
        include_input=bool(options.get("include_input", True)),
    )
    return [encoding], encoding.n_out


# Default ratio of the melt factor of ice to the one of snow of the surface type
# (`sfc_type`). PyGEM uses ddf_snow / ddf_ice = 0.7 (`ddfsnow_iceratio`). OGGM's
# SfcTypeTIModel starts fresh snow at half the ice melt factor and ages it towards
# ice, melt_f(t) = melt_f_ice * (1 - 0.5 * exp(-t / 1 yr)): the winter snow melted in
# summer, 3 to 9 months old, melts at 0.6 to 0.75 of ice. The model has no aging, so
# its snow is the seasonal snow at the time it melts, about 1 / 0.7 of ice for both.
DEFAULT_ICE_RATIO = 1 / 0.7

# Conversion of the downward shortwave radiation `ssrd` of ERA5-Land (J m-2 per day,
# monthly means of the daily accumulations) into the depth of ice or snow (m w.e.)
# it could melt in a month: days per month / (latent heat of fusion * density of
# water)
DAYS_PER_MONTH = 30.4
LATENT_HEAT_FUSION = 3.34e5  # J kg-1
WATER_DENSITY = 1000.0  # kg m-3
SSRD_TO_MELT = DAYS_PER_MONTH / (LATENT_HEAT_FUSION * WATER_DENSITY)

# Inputs of cor_abl that the model derives from its own inputs rather than reads from
# the data: name -> (model inputs it needs, default normalization bounds). The bounds
# can be overridden with the `normalization` block of the params.
#   wind_speed   sqrt(u10^2 + v10^2), m/s: the speed of the monthly MEAN wind vector
#                of ERA5-Land, much lower than the monthly mean speed where the
#                direction varies (0.4 m/s median, 1.5 at most over the stakes of
#                region 11), hence the bounds [0, 2]
DERIVED_COR_ABL_INPUTS = {"wind_speed": (("u10", "v10"), (0.0, 2.0))}


def _init_bounded_scalar(module, name, spec, default, default_bounds, label):
    """Register on `module` a scalar of the model that is either fixed or learned.

    `spec` is {value, trainable, min, max}. A fixed scalar is the non-persistent
    buffer `<name>_value`: not saved in the checkpoints, it is always the value of
    the params. A learned one is the parameter `<name>_raw`, the scalar being
    min + (max - min) * sigmoid(raw), started at `value`. `default` is the value
    and `default_bounds` the (min, max) when `spec` does not give them.
    """
    spec = spec or {}
    unknown = set(spec) - {"value", "trainable", "min", "max"}
    assert not unknown, f"Unknown {label} options: {sorted(unknown)}."
    value = float(spec.get("value", default))
    trainable = bool(spec.get("trainable", False))
    setattr(module, f"{name}_trainable", trainable)
    if trainable:
        lower = float(spec.get("min", default_bounds[0]))
        upper = float(spec.get("max", default_bounds[1]))
        assert (
            lower < value < upper
        ), f"{label} must satisfy min < value < max, got min={lower}, value={value}, max={upper}."
        setattr(module, f"{name}_min", lower)
        setattr(module, f"{name}_max", upper)
        frac = (value - lower) / (upper - lower)
        setattr(module, f"{name}_raw", nn.Parameter(torch.logit(torch.ones(1) * frac)))
    else:
        assert (
            "min" not in spec and "max" not in spec
        ), f"{label}.min and max only apply with trainable: true."
        module.register_buffer(f"{name}_value", torch.ones(1) * value, persistent=False)
    return value


def _bounded_scalar(module, name):
    """Value of a scalar registered by `_init_bounded_scalar`."""
    if not getattr(module, f"{name}_trainable"):
        return getattr(module, f"{name}_value")
    lower, upper = getattr(module, f"{name}_min"), getattr(module, f"{name}_max")
    return lower + (upper - lower) * torch.sigmoid(getattr(module, f"{name}_raw"))


class TILikeModel(nn.Module):
    # Whether the forward pass takes the time order of the rows, `forward(inputs,
    # sfc)`: only a model with a surface type, see `TILikeSfcModel`
    needs_time_order = False

    def __init__(self, modelParams, normalizing_bounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_labels = modelParams["inputs"]
        self.normalizing_bounds = normalizing_bounds
        self.ind_precip = self.input_labels.index("tp_sum")
        self.ind_temp = self.input_labels.index("t2m")
        self.ind_elev_diff = self.input_labels.index("ELEVATION_DIFFERENCE")
        # self.inputs = [
        #     "ELEVATION_DIFFERENCE",
        #     "aspect",
        #     "fal", # forcast albedo
        #     "slhf", # surface latent heat flux
        #     "slope",
        #     "sshf", # surface sensible heat flux
        #     "ssrd", # surface solar radiation downwards
        #     "str", # surface net thermal radiation
        #     "t2m", # 2m temperature
        #     "tp", # total precipitation
        #     "svf", # sky view factor
        # ]
        # self.inp_cor_T = ["t2m", "slhf", "sshf", "ssrd", "str"]
        self.inp_cor_T = (
            modelParams["cor_T"]["inputs"] if "cor_T" in modelParams else None
        )
        self.inp_grad_T = (
            modelParams["grad_T"]["inputs"] if "grad_T" in modelParams else None
        )
        self.inp_bias_T = (
            modelParams["bias_T"]["inputs"] if "bias_T" in modelParams else None
        )
        self.inp_dir = (
            modelParams["cor_dir"]["inputs"] if "cor_dir" in modelParams else None
        )
        self.inp_terrain = (
            modelParams["cor_terrain"]["inputs"]
            if "cor_terrain" in modelParams
            else None
        )
        self.inp_sw_contrib = (
            modelParams["sw_contrib"]["inputs"] if "sw_contrib" in modelParams else None
        )
        # self.inp_cor_fac = [
        #     "aspect",
        #     "fal",
        #     "slhf",
        #     "slope",
        #     "sshf",
        #     "ssrd",
        #     "str",
        #     "svf",
        # ]
        # self.inp_cor_fac = [
        #     "fal",
        #     "slhf",
        #     "sshf",
        #     "ssrd",
        #     "str",
        # ]
        self.inp_cor_acc = modelParams["cor_acc"]["inputs"]
        self.inp_cor_abl = modelParams["cor_abl"]["inputs"]
        self.ind_inp_cor_T = (
            sorted([self.input_labels.index(inp) for inp in self.inp_cor_T])
            if self.inp_cor_T is not None
            else None
        )
        self.ind_inp_grad_T = (
            sorted([self.input_labels.index(inp) for inp in self.inp_grad_T])
            if self.inp_grad_T is not None
            else None
        )
        self.ind_inp_bias_T = (
            sorted([self.input_labels.index(inp) for inp in self.inp_bias_T])
            if self.inp_bias_T is not None
            else None
        )
        self.ind_inp_dir = (
            sorted([self.input_labels.index(inp) for inp in self.inp_dir])
            if self.inp_dir is not None
            else None
        )
        self.ind_inp_terrain = (
            sorted([self.input_labels.index(inp) for inp in self.inp_terrain])
            if self.inp_terrain is not None
            else None
        )
        if self.inp_dir is not None:
            self.ind_svf = sorted([self.input_labels.index(inp) for inp in ["svf"]])
            self.ind_ssrd = sorted([self.input_labels.index(inp) for inp in ["ssrd"]])
        # self.ind_inp_cor_fac = sorted(
        #     [self.input_labels.index(inp) for inp in self.inp_cor_fac]
        # )
        self.ind_inp_sw_contrib = (
            sorted([self.input_labels.index(inp) for inp in self.inp_sw_contrib])
            if self.inp_sw_contrib is not None
            else None
        )
        self.ind_inp_cor_acc = sorted(
            [self.input_labels.index(inp) for inp in self.inp_cor_acc]
        )
        # The inputs of cor_abl read from the model inputs, in the order of the model
        # inputs, then those derived from them (DERIVED_COR_ABL_INPUTS), in the order
        # of the params
        self.cor_abl_derived = [
            inp for inp in self.inp_cor_abl if inp in DERIVED_COR_ABL_INPUTS
        ]
        self.ind_inp_cor_abl = sorted(
            [
                self.input_labels.index(inp)
                for inp in self.inp_cor_abl
                if inp not in DERIVED_COR_ABL_INPUTS
            ]
        )
        for inp in self.cor_abl_derived:
            needed = DERIVED_COR_ABL_INPUTS[inp][0]
            missing = [c for c in needed if c not in self.input_labels]
            assert (
                not missing
            ), f"The cor_abl input {inp} needs {missing} in the model inputs."
        # Scaling of the accumulation term
        self.beta1 = float(modelParams.get("beta1", 10.0))
        assert self.beta1 > 0, f"beta1 must be positive, got {self.beta1}."
        # Scaling of the ablation term, melt = beta2 * cor_abl * PDD in m w.e. per month.
        # With beta2 = 1, cor_abl is the degree-day factor in m w.e. K-1 month-1
        self.beta2 = float(modelParams.get("beta2", 0.0049))
        assert self.beta2 > 0, f"beta2 must be positive, got {self.beta2}."
        # Range of the slope of the snow fraction sigmoid,
        # slope = s_min + (s_max - s_min) * (tanh(tau_P_s) + 1) / 2
        snow_slope_params = modelParams.get("snow_slope", {})
        self.snow_slope_min = float(snow_slope_params.get("min", 0.0))
        self.snow_slope_max = float(snow_slope_params.get("max", 4.0))
        init_tau_P_s = float(snow_slope_params.get("init", 1.5))
        assert (
            0.0 <= self.snow_slope_min < init_tau_P_s < self.snow_slope_max
        ), f"The snow slope must satisfy 0 <= min < init < max, got min={self.snow_slope_min}, init={init_tau_P_s}, max={self.snow_slope_max}."
        init_tau_P_c = 1.0
        init_beta_pdd = 1.0
        init_lapse_rate_P_cor = 0.1
        self.tau_P_s = torch.nn.Parameter(
            torch.arctanh(
                torch.ones(1)
                * (
                    2
                    * (init_tau_P_s - self.snow_slope_min)
                    / (self.snow_slope_max - self.snow_slope_min)
                    - 1
                )
            )
        )
        self.tau_P_c = torch.nn.Parameter(torch.ones(1) * init_tau_P_c)
        self.beta_pdd = torch.nn.Parameter(
            torch.arctanh((torch.ones(1) * init_beta_pdd / 2) - 1)
        )
        # Whether each scalar is optimized, otherwise it keeps its initial value
        trainable = modelParams.get("trainable", {})
        unknown = set(trainable) - {"tau_P_c", "tau_P_s", "beta_pdd"}
        assert not unknown, f"Unknown trainable parameters: {sorted(unknown)}."
        self.tau_P_s.requires_grad = bool(trainable.get("tau_P_s", True))
        self.tau_P_c.requires_grad = bool(trainable.get("tau_P_c", True))
        self.beta_pdd.requires_grad = bool(trainable.get("beta_pdd", True))
        self.lapse_rate_P_cor = torch.nn.Parameter(
            torch.arctanh((torch.ones(1) * (init_lapse_rate_P_cor / 0.2) * 2) - 1)
        )  # 0.2*(tanh(x)+1)/2
        self.lapse_rate_P_cor.requires_grad = True

        # Range of the precipitation correction, P_cor = p_min + (p_max - p_min) * sigmoid,
        # the sigmoid being a product of two with `bias_cor_elev`
        bias_cor_bnds = modelParams.get("bias_cor", {})
        self.p_min = float(bias_cor_bnds.get("p_min", 0.0))
        self.p_max = float(bias_cor_bnds.get("p_max", 2.0))
        assert (
            0.0 <= self.p_min < self.p_max
        ), f"The precipitation correction range must satisfy 0 <= p_min < p_max, got [{self.p_min}, {self.p_max}]."
        # Range of the temperature bias, bias = b_max * tanh
        self.b_max = float(modelParams.get("bias_T", {}).get("b_max", 5.0))
        assert self.b_max > 0, f"b_max must be positive, got {self.b_max}."
        # Range of the temperature lapse rate, alpha = max - (max - min) * sigmoid, in
        # °C/km. The default range is [-15, -2]
        default_lapse_rate = (-15.0, -2.0)
        grad_T_bnds = modelParams.get("grad_T", {})
        self.lapse_rate_min = float(grad_T_bnds.get("min", default_lapse_rate[0]))
        self.lapse_rate_max = float(grad_T_bnds.get("max", default_lapse_rate[1]))
        assert (
            self.lapse_rate_min < self.lapse_rate_max
        ), f"The lapse rate range must satisfy min < max, got [{self.lapse_rate_min}, {self.lapse_rate_max}]."
        # Range of the shortwave radiation contribution, R_sw = R_sw_max * sigmoid,
        # see `_shortwave`
        sw_params = modelParams.get("sw_contrib", {})
        self.R_sw_max = float(sw_params.get("max", 1.0))
        assert (
            self.R_sw_max > 0
        ), f"The maximum of R_sw must be positive, got {self.R_sw_max}."
        # With scale_by_ssrd, the network gives the fraction of the downward
        # shortwave radiation that melts snow, R_sw_max * sigmoid, and R_sw is that
        # fraction of the melt `ssrd` could produce
        self.sw_scale_by_ssrd = bool(sw_params.get("scale_by_ssrd", False))
        if self.sw_scale_by_ssrd:
            assert (
                "ssrd" in self.input_labels
            ), "sw_contrib.scale_by_ssrd needs ssrd in the model inputs."
            self.ind_ssrd_sw = self.input_labels.index("ssrd")
        # Optional temperature gate of R_sw: sigmoid((T - threshold) / width), T the
        # downscaled monthly temperature (°C), with its own threshold, distinct from
        # the 0 °C of the PDD
        gate = sw_params.get("gate")
        self.sw_gate = gate is not None
        if self.sw_gate:
            unknown = set(gate) - {"threshold", "width"}
            assert not unknown, f"Unknown sw_contrib.gate options: {sorted(unknown)}."
            _init_bounded_scalar(
                self,
                "sw_gate_threshold",
                gate.get("threshold"),
                default=0.0,
                default_bounds=(-5.0, 1.0),
                label="sw_contrib.gate.threshold",
            )
            self.sw_gate_width = float(gate.get("width", 1.0))
            assert (
                self.sw_gate_width > 0
            ), f"sw_contrib.gate.width must be positive, got {self.sw_gate_width}."

        if "bias_cor" in modelParams:
            self.inp_bias_cor = modelParams["bias_cor"]["inputs"]
            self.ind_inp_bias_cor = sorted(
                [self.input_labels.index(inp) for inp in self.inp_bias_cor]
            )

            bias_cor_params = modelParams["bias_cor"]
            act = _activation(bias_cor_params, "bias_cor")
            encoding, n_features = _input_encoding(
                bias_cor_params, len(self.inp_bias_cor), "bias_cor"
            )
            bias_cor = encoding + [nn.Linear(n_features, bias_cor_params["layers"][0])]
            for i in range(len(bias_cor_params["layers"]) - 1):
                bias_cor.append(act())
                bias_cor.append(
                    nn.Linear(
                        bias_cor_params["layers"][i], bias_cor_params["layers"][i + 1]
                    )
                )
            bias_cor.append(act())
            bias_cor.append(nn.Linear(bias_cor_params["layers"][-1], 1))
            self.bias_cor = nn.Sequential(*bias_cor)
        else:
            self.inp_bias_cor = None
            self.bias_cor = None

        # Optional elevation factor of the precipitation correction, which then reads
        # sigmoid(bias_cor(...)) * sigmoid(bias_cor_elev(...)): with bias_cor on the
        # location only, the correction is separable in space and elevation
        if "bias_cor_elev" in modelParams:
            assert (
                self.bias_cor is not None
            ), "bias_cor_elev is a factor of bias_cor, which the model does not define."
            self.inp_bias_cor_elev = modelParams["bias_cor_elev"]["inputs"]
            self.ind_inp_bias_cor_elev = sorted(
                [self.input_labels.index(inp) for inp in self.inp_bias_cor_elev]
            )

            bias_cor_elev_params = modelParams["bias_cor_elev"]
            act = _activation(bias_cor_elev_params, "bias_cor_elev")
            bias_cor_elev = [
                nn.Linear(
                    len(self.inp_bias_cor_elev), bias_cor_elev_params["layers"][0]
                )
            ]
            for i in range(len(bias_cor_elev_params["layers"]) - 1):
                bias_cor_elev.append(act())
                bias_cor_elev.append(
                    nn.Linear(
                        bias_cor_elev_params["layers"][i],
                        bias_cor_elev_params["layers"][i + 1],
                    )
                )
            bias_cor_elev.append(act())
            bias_cor_elev.append(nn.Linear(bias_cor_elev_params["layers"][-1], 1))
            self.bias_cor_elev = nn.Sequential(*bias_cor_elev)
        else:
            self.inp_bias_cor_elev = None
            self.bias_cor_elev = None

        if "cor_T" in modelParams:
            cor_T_params = modelParams["cor_T"]
            cor_T = [nn.Linear(len(self.inp_cor_T), cor_T_params["layers"][0])]
            for i in range(len(cor_T_params["layers"]) - 1):
                cor_T.append(nn.ReLU())
                cor_T.append(
                    nn.Linear(cor_T_params["layers"][i], cor_T_params["layers"][i + 1])
                )
            cor_T.append(nn.ReLU())
            cor_T.append(nn.Linear(cor_T_params["layers"][-1], 2))
            self.cor_T = nn.Sequential(*cor_T)
        else:
            self.cor_T = None
        if "grad_T" in modelParams:
            grad_T_params = modelParams["grad_T"]
            grad_T = [nn.Linear(len(self.inp_grad_T), grad_T_params["layers"][0])]
            for i in range(len(grad_T_params["layers"]) - 1):
                grad_T.append(nn.ReLU())
                grad_T.append(
                    nn.Linear(
                        grad_T_params["layers"][i], grad_T_params["layers"][i + 1]
                    )
                )
            grad_T.append(nn.ReLU())
            grad_T.append(nn.Linear(grad_T_params["layers"][-1], 1))
            self.grad_T = nn.Sequential(*grad_T)
        else:
            self.grad_T = None
        if "bias_T" in modelParams:
            bias_T_params = modelParams["bias_T"]
            encoding, n_features = _input_encoding(
                bias_T_params, len(self.inp_bias_T), "bias_T"
            )
            bias_T = encoding + [nn.Linear(n_features, bias_T_params["layers"][0])]
            for i in range(len(bias_T_params["layers"]) - 1):
                bias_T.append(nn.ReLU())
                bias_T.append(
                    nn.Linear(
                        bias_T_params["layers"][i], bias_T_params["layers"][i + 1]
                    )
                )
            bias_T.append(nn.ReLU())
            bias_T.append(nn.Linear(bias_T_params["layers"][-1], 1))
            self.bias_T = nn.Sequential(*bias_T)
        else:
            self.bias_T = None
        if "cor_dir" in modelParams:
            cor_dir_params = modelParams["cor_dir"]
            cor_dir = [nn.Linear(len(self.inp_dir), cor_dir_params["layers"][0])]
            for i in range(len(cor_dir_params["layers"]) - 1):
                cor_dir.append(nn.ReLU())
                cor_dir.append(
                    nn.Linear(
                        cor_dir_params["layers"][i], cor_dir_params["layers"][i + 1]
                    )
                )
            cor_dir.append(nn.ReLU())
            cor_dir.append(nn.Linear(cor_dir_params["layers"][-1], 1, bias=False))
            self.cor_dir = nn.Sequential(*cor_dir)
        else:
            self.cor_dir = None
        if "cor_terrain" in modelParams:
            cor_terrain_params = modelParams["cor_terrain"]
            cor_terrain = [
                nn.Linear(len(self.inp_terrain), cor_terrain_params["layers"][0])
            ]
            for i in range(len(cor_terrain_params["layers"]) - 1):
                cor_terrain.append(nn.ReLU())
                cor_terrain.append(
                    nn.Linear(
                        cor_terrain_params["layers"][i],
                        cor_terrain_params["layers"][i + 1],
                    )
                )
            cor_terrain.append(nn.ReLU())
            cor_terrain.append(
                nn.Linear(cor_terrain_params["layers"][-1], 1, bias=False)
            )
            self.cor_terrain = nn.Sequential(*cor_terrain)
        else:
            self.cor_terrain = None
        if "sw_contrib" in modelParams:
            sw_contrib_params = modelParams["sw_contrib"]
            R_sw_contrib = [
                nn.Linear(len(self.inp_sw_contrib), sw_contrib_params["layers"][0])
            ]
            for i in range(len(sw_contrib_params["layers"]) - 1):
                R_sw_contrib.append(nn.ReLU())
                R_sw_contrib.append(
                    nn.Linear(
                        sw_contrib_params["layers"][i],
                        sw_contrib_params["layers"][i + 1],
                    )
                )
            R_sw_contrib.append(nn.ReLU())
            R_sw_contrib.append(nn.Linear(sw_contrib_params["layers"][-1], 1))
            self.R_sw_contrib = nn.Sequential(*R_sw_contrib)
        else:
            self.R_sw_contrib = None

        cor_acc_params = modelParams["cor_acc"]
        cor_acc = [nn.Linear(len(self.inp_cor_acc), cor_acc_params["layers"][0])]
        for i in range(len(cor_acc_params["layers"]) - 1):
            cor_acc.append(nn.ReLU())
            cor_acc.append(
                nn.Linear(cor_acc_params["layers"][i], cor_acc_params["layers"][i + 1])
            )
        cor_acc.append(nn.ReLU())
        cor_acc.append(nn.Linear(cor_acc_params["layers"][-1], 1))
        cor_acc.append(nn.Softplus())
        self.cor_acc = nn.Sequential(*cor_acc)

        cor_abl_params = modelParams["cor_abl"]
        cor_abl = [nn.Linear(len(self.inp_cor_abl), cor_abl_params["layers"][0])]
        for i in range(len(cor_abl_params["layers"]) - 1):
            cor_abl.append(nn.ReLU())
            cor_abl.append(
                nn.Linear(cor_abl_params["layers"][i], cor_abl_params["layers"][i + 1])
            )
        cor_abl.append(nn.ReLU())
        cor_abl.append(nn.Linear(cor_abl_params["layers"][-1], 1))
        self.cor_abl = nn.Sequential(*cor_abl)
        # Output of the ablation factor: "softplus" (unbounded, the default) or
        # "sigmoid", cor_abl = min + (max - min) * sigmoid, within prescribed bounds
        self.cor_abl_output = cor_abl_params.get("output", "softplus")
        assert self.cor_abl_output in (
            "softplus",
            "sigmoid",
        ), f"cor_abl.output must be 'softplus' or 'sigmoid', got {self.cor_abl_output}."
        if self.cor_abl_output == "sigmoid":
            assert (
                "min" in cor_abl_params and "max" in cor_abl_params
            ), "cor_abl.output 'sigmoid' needs the bounds cor_abl.min and cor_abl.max."
            self.cor_abl_min = float(cor_abl_params["min"])
            self.cor_abl_max = float(cor_abl_params["max"])
            assert (
                0.0 <= self.cor_abl_min < self.cor_abl_max
            ), f"The ablation factor range must satisfy 0 <= min < max, got [{self.cor_abl_min}, {self.cor_abl_max}]."
        else:
            assert (
                "min" not in cor_abl_params and "max" not in cor_abl_params
            ), "cor_abl.min and cor_abl.max only apply with cor_abl.output 'sigmoid'."

        assert self.needs_time_order or "sfc_type" not in modelParams, (
            "A model with a surface type (sfc_type) is a TILikeSfcModel, see "
            "`createModel`."
        )

    def _mass_balance(self, P_solid, PDD, cor_acc, cor_abl, R_sw):
        """Monthly mass balance (m w.e.) from the terms of the forward pass."""
        return self.beta1 * cor_acc * P_solid - self.beta2 * cor_abl * PDD - R_sw

    def _shortwave(self, inputs, T_downscaled):
        """Shortwave radiation contribution R_sw (m w.e. per month) of the normalized
        inputs, `T_downscaled` the downscaled monthly temperature (°C).

        By default R_sw = R_sw_max * sigmoid(sw_contrib(inputs)). With
        `scale_by_ssrd`, that is the fraction of the melt the downward shortwave
        radiation could produce, ssrd * SSRD_TO_MELT. With a `gate`, R_sw is
        multiplied by sigmoid((T_downscaled - threshold) / width). For a model with a
        surface type, it is the contribution on snow, see `TILikeSfcModel`.
        """
        R_sw = self.R_sw_max * F.sigmoid(
            self.R_sw_contrib(inputs[:, self.ind_inp_sw_contrib]).view(-1)
        )
        if self.sw_scale_by_ssrd:
            ssrd = Normalizer._unorm(
                inputs[:, self.ind_ssrd_sw],
                self.normalizing_bounds["ssrd"][0],
                self.normalizing_bounds["ssrd"][1],
            )
            R_sw = R_sw * ssrd.clamp(min=0.0) * SSRD_TO_MELT
        if self.sw_gate:
            R_sw = R_sw * torch.sigmoid(
                (T_downscaled - self.sw_gate_threshold()) / self.sw_gate_width
            )
        return R_sw

    def sw_gate_threshold(self):
        """Threshold (°C) of the temperature gate of R_sw."""
        return _bounded_scalar(self, "sw_gate_threshold")

    def snow_slope(self):
        """Slope (1/°C) of the sigmoid splitting precipitation into snow, from the
        learnable tau_P_s, within [snow_slope_min, snow_slope_max]."""
        return (
            self.snow_slope_min
            + (self.snow_slope_max - self.snow_slope_min)
            * (torch.tanh(self.tau_P_s) + 1)
            / 2
        )

    def pdd_curvature(self):
        """Curvature (1/°C) of the softplus giving the positive degree days, from
        the learnable beta_pdd. The larger, the closer to max(T, 0)."""
        return (torch.tanh(self.beta_pdd) + 1) * 2

    def scalar_parameters(self):
        """Values of the learnable scalars as the model uses them.

        Returns a dict with the slope of the snow fraction sigmoid (1/°C), its
        threshold temperature tau_P_c (°C), the curvature of the PDD softplus
        (1/°C), together with the raw parameters they are computed from.
        """
        with torch.no_grad():
            return {
                "snow_slope": self.snow_slope().item(),
                "snow_threshold": self.tau_P_c.item(),
                "pdd_curvature": self.pdd_curvature().item(),
                "tau_P_s": self.tau_P_s.item(),
                "tau_P_c": self.tau_P_c.item(),
                "beta_pdd": self.beta_pdd.item(),
                **(
                    {"sw_gate_threshold": self.sw_gate_threshold().item()}
                    if self.sw_gate
                    else {}
                ),
            }

    def _net_from_df(self, net, columns, df, name):
        """Raw output of `net` on the unnormalized `columns` of df."""
        if net is None:
            raise RuntimeError(f"This model does not define a {name} network.")
        missing_columns = [column for column in columns if column not in df]
        if missing_columns:
            raise ValueError(f"Missing {name} columns: {missing_columns}")

        # In the order of the model inputs, as the forward pass takes them
        columns = sorted(columns, key=self.input_labels.index)
        parameter = next(net.parameters())
        inputs = torch.as_tensor(
            df[columns].to_numpy(dtype=np.float32),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        for column_index, column in enumerate(columns):
            lower_bound, upper_bound = self.normalizing_bounds[column]
            inputs[:, column_index] = Normalizer._norm(
                inputs[:, column_index], lower_bound, upper_bound
            )
        return net(inputs)[:, 0]

    def _lapse_rate(self, inp_grad_T):
        """Temperature lapse rate, within [lapse_rate_min, lapse_rate_max]."""
        return self.lapse_rate_max - (
            self.lapse_rate_max - self.lapse_rate_min
        ) * F.sigmoid(self.grad_T(inp_grad_T)[:, 0])

    def _cor_abl_inputs(self, inputs):
        """Normalized inputs of cor_abl from the normalized model inputs: the columns
        of the model inputs, then the derived inputs (DERIVED_COR_ABL_INPUTS)."""
        x = inputs[:, self.ind_inp_cor_abl]
        if not self.cor_abl_derived:
            return x
        derived = []
        for inp in self.cor_abl_derived:
            if inp == "wind_speed":
                u, v = (
                    Normalizer._unorm(
                        inputs[:, self.input_labels.index(c)],
                        self.normalizing_bounds[c][0],
                        self.normalizing_bounds[c][1],
                    )
                    for c in ("u10", "v10")
                )
                value = torch.sqrt(u**2 + v**2)
            else:
                raise NotImplementedError(f"Derived cor_abl input {inp}.")
            lower, upper = self.normalizing_bounds.get(
                inp, DERIVED_COR_ABL_INPUTS[inp][1]
            )
            derived.append((value - lower) / (upper - lower))
        return torch.cat([x, torch.stack(derived, dim=1)], dim=1)

    def _ablation_factor(self, inp_cor_abl):
        """Ablation factor of the normalized inputs of cor_abl, unbounded (softplus) or
        within [cor_abl_min, cor_abl_max] (sigmoid)."""
        raw = self.cor_abl(inp_cor_abl).view(-1)
        if self.cor_abl_output == "sigmoid":
            return self.cor_abl_min + (self.cor_abl_max - self.cor_abl_min) * F.sigmoid(
                raw
            )
        return F.softplus(raw)

    def _P_cor(self, inputs):
        """Precipitation correction of the normalized model inputs."""
        factor = F.sigmoid(self.bias_cor(inputs[:, self.ind_inp_bias_cor])[:, 0])
        if self.bias_cor_elev is not None:
            factor = factor * F.sigmoid(
                self.bias_cor_elev(inputs[:, self.ind_inp_bias_cor_elev])[:, 0]
            )
        return self.p_min + (self.p_max - self.p_min) * factor

    @property
    def inp_P_cor(self):
        """Inputs the precipitation correction depends on."""
        if self.bias_cor is None:
            return None
        return list(dict.fromkeys(self.inp_bias_cor + (self.inp_bias_cor_elev or [])))

    def predict_bias_cor_from_df(self, df):
        """Predict the precipitation correction for a set of points in a DataFrame.

        The input columns are expected to contain unnormalized values. The
        returned tensor has one value per row of df, the precipitation correction
        ``P_cor`` of the forward pass, within [p_min, p_max].
        """
        factor = F.sigmoid(
            self._net_from_df(
                self.bias_cor, self.inp_bias_cor, df, "precipitation correction"
            )
        )
        if self.bias_cor_elev is not None:
            factor = factor * F.sigmoid(
                self._net_from_df(
                    self.bias_cor_elev,
                    self.inp_bias_cor_elev,
                    df,
                    "precipitation correction elevation factor",
                )
            )
        return self.p_min + (self.p_max - self.p_min) * factor

    def predict_temp_bias_from_df(self, df):
        """Predict the temperature bias for a set of points in a DataFrame.

        The input columns are expected to contain unnormalized values. The
        returned tensor has one value per row of df, the temperature bias of the
        forward pass in °C, within [-b_max, b_max].
        """
        raw = self._net_from_df(self.bias_T, self.inp_bias_T, df, "temperature bias")
        return self.b_max * F.tanh(raw)

    def get_cor_T(self, inputs):
        P = Normalizer._unorm(
            inputs[:, self.ind_precip],
            self.normalizing_bounds["tp_sum"][0],
            self.normalizing_bounds["tp_sum"][1],
        )
        T = Normalizer._unorm(
            inputs[:, self.ind_temp],
            self.normalizing_bounds["t2m"][0],
            self.normalizing_bounds["t2m"][1],
        )  # in Celsius degrees
        elev_diff_unorm = Normalizer._unorm(
            inputs[:, self.ind_elev_diff],
            self.normalizing_bounds["ELEVATION_DIFFERENCE"][0],
            self.normalizing_bounds["ELEVATION_DIFFERENCE"][1],
        )
        if self.cor_T is not None:
            inp_cor_T = inputs[:, self.ind_inp_cor_T]
        else:
            inp_grad_T = inputs[:, self.ind_inp_grad_T]
            inp_bias_T = inputs[:, self.ind_inp_bias_T]
        inp_cor_acc = inputs[:, self.ind_inp_cor_acc]
        inp_cor_abl = self._cor_abl_inputs(inputs)

        if self.cor_T is not None:
            # Temperature correction for precipitation and PDD
            # Acts as a shift of the temperature
            cor_T_val = self.cor_T(inp_cor_T)
            elev_diff = inputs[:, self.ind_elev_diff]
            cor_T = elev_diff * cor_T_val[:, 0] + cor_T_val[:, 1]
        elif self.cor_dir is not None and self.cor_terrain is not None:
            # grad_T_val = self.grad_T(inp_grad_T)
            # bias_T_val = self.bias_T(inp_bias_T)
            # cor_T_val = torch.concatenate([grad_T_val, bias_T_val], dim=1)
            # In km, so that the lapse rate is in °C/km whatever the normalization
            elev_diff = elev_diff_unorm / 1000
            # cor_T = elev_diff * cor_T_val[:, 0] + cor_T_val[:, 1]

            inp_dir = inputs[:, self.ind_inp_dir]
            inp_terrain = inputs[:, self.ind_inp_terrain]
            alpha = self._lapse_rate(inp_grad_T)
            # alpha = self.grad_T(inp_grad_T)[:, 0]
            bias = self.b_max * F.tanh(self.bias_T(inp_bias_T)[:, 0])
            svf_unorm = Normalizer._unorm(
                inputs[:, self.ind_svf],
                self.normalizing_bounds["svf"][0],
                self.normalizing_bounds["svf"][1],
            )[:, 0]
            ssrd = inputs[:, self.ind_ssrd][:, 0]
            R_dir = self.cor_dir(inp_dir)[:, 0]
            R_terrain = self.cor_terrain(inp_terrain)[:, 0]
            cor_T_val = torch.concatenate(
                [
                    alpha[:, None],
                    bias[:, None],
                    ssrd[:, None],
                    svf_unorm[:, None],
                    R_dir[:, None],
                    R_terrain[:, None],
                ],
                dim=1,
            )
            cor_T = (
                # elev_diff * alpha + ssrd * (1 - svf_unorm) * R_dir + R_terrain + bias
                elev_diff * alpha
                + R_dir
                + R_terrain
                + bias
            )

            R_sw = 0.0
        else:
            # In km, so that the lapse rate is in °C/km whatever the normalization
            elev_diff = elev_diff_unorm / 1000
            alpha = self._lapse_rate(inp_grad_T)
            bias = self.b_max * F.tanh(self.bias_T(inp_bias_T)[:, 0])
            cor_T_val = torch.stack([alpha, bias], dim=1)
            cor_T = elev_diff * alpha + bias

            # Short wave radiation contribution
            R_sw = self._shortwave(inputs, T + cor_T)

        if self.bias_cor is not None:
            # P_cor = (
            #     1
            #     + (elev_diff_unorm / 100)
            #     * 0.2
            #     * (torch.tanh(self.lapse_rate_P_cor) + 1)
            #     / 2
            # ) * (F.sigmoid(self.bias_cor(inp_bias_cor)[:, 0]) * 2)
            P_cor = self._P_cor(inputs)
        else:
            P_cor = 1.0
        snow_slope, curv_pdd = self.snow_slope(), self.pdd_curvature()
        P_solid = P * P_cor * F.sigmoid(snow_slope * (self.tau_P_c - cor_T - T))
        PDD = F.softplus((T + cor_T) * curv_pdd) / curv_pdd

        cor_acc = self.cor_acc(inp_cor_acc).view(-1)
        cor_abl = self._ablation_factor(inp_cor_abl)

        return (
            cor_T,
            cor_T_val,
            P,
            T,
            P_solid,
            P_cor,
            PDD,
            elev_diff_unorm,
            cor_acc,
            cor_abl,
            R_sw,
        )

    def forward(self, inputs):
        """Monthly mass balance (m w.e.) of the normalized inputs, one row per point
        and month, shape (n, 1)."""
        return self._mass_balance(*self._terms(inputs)).view(-1, 1)

    def _terms(self, inputs):
        """Terms of the mass balance of the normalized inputs: P_solid, PDD,
        cor_acc, cor_abl and R_sw, one value per row."""
        P = Normalizer._unorm(
            inputs[:, self.ind_precip],
            self.normalizing_bounds["tp_sum"][0],
            self.normalizing_bounds["tp_sum"][1],
        )
        T = Normalizer._unorm(
            inputs[:, self.ind_temp],
            self.normalizing_bounds["t2m"][0],
            self.normalizing_bounds["t2m"][1],
        )  # in Celsius degrees
        elev_diff_unorm = Normalizer._unorm(
            inputs[:, self.ind_elev_diff],
            self.normalizing_bounds["ELEVATION_DIFFERENCE"][0],
            self.normalizing_bounds["ELEVATION_DIFFERENCE"][1],
        )
        if self.cor_T is not None:
            inp_cor_T = inputs[:, self.ind_inp_cor_T]
        else:
            inp_grad_T = inputs[:, self.ind_inp_grad_T]
            inp_bias_T = inputs[:, self.ind_inp_bias_T]
        inp_cor_acc = inputs[:, self.ind_inp_cor_acc]
        inp_cor_abl = self._cor_abl_inputs(inputs)

        if self.cor_T is not None:
            # Temperature correction for precipitation and PDD
            # Acts as a shift of the temperature
            cor_T_val = self.cor_T(inp_cor_T)
            elev_diff = inputs[:, self.ind_elev_diff]
            cor_T = elev_diff * cor_T_val[:, 0] + cor_T_val[:, 1]
        elif self.cor_dir is not None and self.cor_terrain is not None:
            # grad_T_val = self.grad_T(inp_grad_T)
            # bias_T_val = self.bias_T(inp_bias_T)
            # cor_T_val = torch.concatenate([grad_T_val, bias_T_val], dim=1)
            # In km, so that the lapse rate is in °C/km whatever the normalization
            elev_diff = elev_diff_unorm / 1000
            # cor_T = elev_diff * cor_T_val[:, 0] + cor_T_val[:, 1]

            inp_dir = inputs[:, self.ind_inp_dir]
            inp_terrain = inputs[:, self.ind_inp_terrain]
            alpha = self._lapse_rate(inp_grad_T)
            # alpha = self.grad_T(inp_grad_T)[:, 0]
            bias = self.b_max * F.tanh(self.bias_T(inp_bias_T)[:, 0])
            # svf_unorm = Normalizer._unorm(
            #     inputs[:, self.ind_svf],
            #     self.normalizing_bounds["svf"][0],
            #     self.normalizing_bounds["svf"][1],
            # )[:, 0]
            # ssrd = inputs[:, self.ind_ssrd][:, 0]
            R_dir = self.cor_dir(inp_dir)[:, 0]
            R_terrain = self.cor_terrain(inp_terrain)[:, 0]
            cor_T = (
                # elev_diff * alpha + ssrd * (1 - svf_unorm) * R_dir + R_terrain + bias
                elev_diff * alpha
                + R_dir
                + R_terrain
                + bias
            )

            R_sw = 0.0
        else:
            # In km, so that the lapse rate is in °C/km whatever the normalization
            elev_diff = elev_diff_unorm / 1000
            alpha = self._lapse_rate(inp_grad_T)
            bias = self.b_max * F.tanh(self.bias_T(inp_bias_T)[:, 0])
            cor_T = elev_diff * alpha + bias

            # Short wave radiation contribution
            R_sw = self._shortwave(inputs, T + cor_T)

        if self.bias_cor is not None:
            # P_cor = (
            #     1
            #     + (elev_diff_unorm / 100)
            #     * 0.2
            #     * (torch.tanh(self.lapse_rate_P_cor) + 1)
            #     / 2
            # ) * (F.sigmoid(self.bias_cor(inp_bias_cor)[:, 0]) * 2)
            P_cor = self._P_cor(inputs)
        else:
            P_cor = 1.0
        snow_slope, curv_pdd = self.snow_slope(), self.pdd_curvature()
        P_solid = P * P_cor * F.sigmoid(snow_slope * (self.tau_P_c - cor_T - T))
        PDD = F.softplus((T + cor_T) * curv_pdd) / curv_pdd

        # Accumulation factor correction
        cor_acc = self.cor_acc(inp_cor_acc).view(-1)

        # Ablation factor correction
        cor_abl = self._ablation_factor(inp_cor_abl)

        # TODO: changer scaling de beta2 pour correspondre aux valeurs typiques calées avec ODINN
        return P_solid, PDD, cor_acc, cor_abl, R_sw


class TILikeSfcModel(TILikeModel):
    """TILikeModel with a surface type (snow or ice), the `sfc_type` block of the
    parameters. cor_abl and R_sw are those of snow; on ice they are multiplied by
    ice_ratio and sw_ratio, mixed by the snow cover fraction of every row (see
    `_snow_cover`). Its forward pass takes the time order of the rows, `sfc`.

    ice_ratio (temperature melt) is 1 / 0.7 by default and sw_ratio (absorbed
    shortwave radiation, (1 - albedo of ice) / (1 - albedo of snow)) is 1: with
    `ice_ratio: {value: 1}` and a sw_ratio, the surface type acts on R_sw only."""

    needs_time_order = True

    def __init__(self, modelParams, normalizing_bounds, *args, **kwargs):
        super().__init__(modelParams, normalizing_bounds, *args, **kwargs)
        sfc_params = modelParams["sfc_type"] or {}
        unknown = set(sfc_params) - {
            "snow_scale",
            "snow_midpoint",
            "ice_ratio",
            "sw_ratio",
        }
        assert not unknown, f"Unknown sfc_type options: {sorted(unknown)}."
        # Width (m w.e.) of the transition from ice to snow: the snow cover fraction
        # goes from 0.27 to 0.73 between snow_midpoint -/+ snow_scale
        self.snow_scale = float(sfc_params.get("snow_scale", 0.1))
        assert (
            self.snow_scale > 0
        ), f"sfc_type.snow_scale must be positive, got {self.snow_scale}."
        # Snow water equivalent (m w.e.) at which half of the surface is snow. Every
        # point starts its hydrological year at swe = 0, so with a positive midpoint
        # every point, in the ablation and the accumulation areas, starts close to
        # bare ice (snow fraction sigmoid(-snow_midpoint / snow_scale)) until the
        # snow of the autumn builds up
        self.snow_midpoint = float(sfc_params.get("snow_midpoint", 0.0))
        assert (
            self.snow_midpoint >= 0
        ), f"sfc_type.snow_midpoint must be non-negative, got {self.snow_midpoint}."
        # Ratio of the melt factor of ice to the one of snow. Fixed by default, at
        # DEFAULT_ICE_RATIO; with `trainable`, learned within [min, max] (default
        # [1, 3]) from `value`
        ice_ratio = _init_bounded_scalar(
            self,
            "ice_ratio",
            sfc_params.get("ice_ratio"),
            default=DEFAULT_ICE_RATIO,
            default_bounds=(1.0, 3.0),
            label="sfc_type.ice_ratio",
        )
        assert ice_ratio > 0, f"The ice ratio must be positive, got {ice_ratio}."
        # Ratio of the shortwave radiation contribution on ice to the one on snow,
        # (1 - albedo of ice) / (1 - albedo of snow), about 2-3 for an ice albedo of
        # 0.3-0.4 and a snow albedo of 0.7-0.85. 1 by default (no effect); learned
        # within [min, max] (default [1, 4]) with `trainable`
        sw_ratio = _init_bounded_scalar(
            self,
            "sw_ratio",
            sfc_params.get("sw_ratio"),
            default=1.0,
            default_bounds=(1.0, 4.0),
            label="sfc_type.sw_ratio",
        )
        assert sw_ratio > 0, f"The shortwave ratio must be positive, got {sw_ratio}."

    def forward(self, inputs, sfc):
        """Monthly mass balance (m w.e.) of the normalized inputs, one row per point
        and month, shape (n, 1). `sfc` is the time order of the rows
        (`precomputed_meta["sfc"]` of the dataloader), see `_snow_cover`."""
        return self._mass_balance_sfc(*self._terms(inputs), sfc).view(-1, 1)

    def ice_ratio(self):
        """Ratio of the melt factor of ice to the one of snow: the fixed
        ice_ratio_value, or, when it is learned, from ice_ratio_raw within
        [ice_ratio_min, ice_ratio_max]."""
        return _bounded_scalar(self, "ice_ratio")

    def sw_ratio(self):
        """Ratio of the shortwave radiation contribution on ice to the one on snow,
        fixed (sw_ratio_value) or learned (sw_ratio_raw)."""
        return _bounded_scalar(self, "sw_ratio")

    def _ice_mix(self, f_snow, ratio):
        """Factor of a snow term on a surface with the snow fraction f_snow, `ratio`
        being the ratio of ice to snow."""
        return f_snow + (1 - f_snow) * ratio

    def _snow_cover(self, acc, melt_snow, R_sw, sfc):
        """Snow water equivalent and snow fraction of every row.

        The snow water equivalent (m w.e.) is the cumulative mass balance with the
        melt factor and the shortwave contribution of snow, `acc - melt_snow -
        R_sw`, of the previous months of the
        same point since the start of the hydrological year, plus half of the
        current month (its middle). It is not bounded below: once the snow is gone,
        it keeps decreasing and summer snowfalls do not bring the snow back. The snow cover
        fraction is sigmoid((swe - snow_midpoint) / snow_scale).

        `sfc` (dict) orders the rows in time, see `dataloader.surface_type`.

        Returns two torch.Tensor with one value per row: swe and the snow cover
        fraction f_snow.
        """
        if sfc is None:
            raise ValueError(
                "This model has a surface type (sfc_type) and needs the time order of "
                "its inputs, `precomputed_meta['sfc']` built by the dataloader when "
                "`cfg.surface_type` is set (see `cli.common.setSurfaceType`)."
            )
        perm = sfc["perm"].to(acc.device)
        start = sfc["start"].to(acc.device)
        assert perm.shape[0] == acc.shape[0], (
            f"The time order covers {perm.shape[0]} rows but the inputs have "
            f"{acc.shape[0]}."
        )
        # In double precision: the cumulative sum runs over all the sequences
        delta = (acc - melt_snow - R_sw)[perm].double()
        before = torch.cumsum(delta, 0) - delta
        swe_sorted = before - before[start] + 0.5 * delta
        swe = torch.empty_like(acc)
        swe[perm] = swe_sorted.to(acc.dtype)
        return swe, torch.sigmoid((swe - self.snow_midpoint) / self.snow_scale)

    def _mass_balance_sfc(self, P_solid, PDD, cor_acc, cor_abl, R_sw, sfc):
        """Monthly mass balance (m w.e.) from the terms of the forward pass, the
        melt factor and the shortwave contribution mixing snow and ice."""
        acc = self.beta1 * cor_acc * P_solid
        melt = self.beta2 * cor_abl * PDD
        _, f_snow = self._snow_cover(acc, melt, R_sw, sfc)
        MB = (
            acc
            - melt * self._ice_mix(f_snow, self.ice_ratio())
            - R_sw * self._ice_mix(f_snow, self.sw_ratio())
        )
        if sfc.get("spinup") is not None:
            # The spin-up rows only build the snow cover, see
            # `data_processing.transform_to_monthly.transform_to_monthly`
            MB = MB.masked_fill(sfc["spinup"].to(MB.device), 0.0)
        return MB

    def snow_cover(self, inputs, sfc):
        """Snow water equivalent (m w.e.), snow cover fraction, effective ablation
        factor and effective shortwave contribution (cor_abl and R_sw times the
        snow/ice mix, the latter in m w.e. per month) of the normalized inputs, see
        `_snow_cover`."""
        P_solid, PDD, cor_acc, cor_abl, R_sw = self._terms(inputs)
        acc = self.beta1 * cor_acc * P_solid
        melt = self.beta2 * cor_abl * PDD
        swe, f_snow = self._snow_cover(acc, melt, R_sw, sfc)
        cor_abl_eff = cor_abl * self._ice_mix(f_snow, self.ice_ratio())
        R_sw_eff = torch.as_tensor(R_sw, device=f_snow.device) * self._ice_mix(
            f_snow, self.sw_ratio()
        )
        return swe, f_snow, cor_abl_eff, R_sw_eff

    def scalar_parameters(self):
        """The scalars of `TILikeModel.scalar_parameters` and the ratios of ice to
        snow of the melt factor and of the shortwave contribution (with their raw
        parameters when they are learned)."""
        params = super().scalar_parameters()
        with torch.no_grad():
            for name in ["ice_ratio", "sw_ratio"]:
                params[name] = _bounded_scalar(self, name).item()
                if getattr(self, f"{name}_trainable"):
                    params[f"{name}_raw"] = getattr(self, f"{name}_raw").item()
        return params


def needs_time_order(model) -> bool:
    """Whether the forward pass of `model` (a module or its
    CustomTorchNeuralNetRegressor) takes the time order of the rows, see
    `TILikeSfcModel`."""
    return getattr(getattr(model, "module", model), "needs_time_order", False)


def forward_rows(model, inputs, sfc=None):
    """Forward pass of `model` on the rows `inputs`, given the time order `sfc` of
    the rows (`precomputed_meta.get("sfc")`) when the model needs it."""
    if needs_time_order(model):
        return model.forward(inputs, sfc)
    return model.forward(inputs)


def createModel(cfg, modelParams, nInp=None):
    nInp = nInp or len(cfg.featureColumns)
    dropout = modelParams.get("dropout", 0.0)
    if modelParams["type"] == "sequential":
        assert len(modelParams["layers"]) > 0
        l = [nn.Linear(nInp, modelParams["layers"][0])]
        for i in range(len(modelParams["layers"]) - 1):
            l.append(nn.ReLU())
            if dropout > 0:
                l.append(nn.Dropout(dropout))
            l.append(nn.Linear(modelParams["layers"][i], modelParams["layers"][i + 1]))
        l.append(nn.ReLU())
        if dropout > 0:
            l.append(nn.Dropout(dropout))
        l.append(nn.Linear(modelParams["layers"][-1], 1))
        network = nn.Sequential(*l)
        return network
    if modelParams["type"] == "sequential_downscaled":
        assert len(modelParams["layers"]) > 0
        return SequentialDownscaledModel(modelParams, cfg.bnds)
    elif modelParams["type"] == "TIlike":
        if "sfc_type" in modelParams:
            return TILikeSfcModel(modelParams, cfg.bnds)
        return TILikeModel(modelParams, cfg.bnds)
    else:
        raise ValueError(f"Model {modelParams['type']} is not supported.")


def selectModel(cfg, version):
    if version == "minimalistic":
        paramsFile = "minimalistic.yml"
    with open("scripts/netcfg/" + paramsFile) as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return createModel(cfg, params["model"])


def buildModel(cfg, version=None, params=None):
    assert (version is None) ^ (
        params is None
    ), "Either version or params must be provided."
    if version is not None:
        model = selectModel(cfg, version)
    else:
        if "model" in params:
            params = params["model"]
        model = createModel(cfg, params)
    return model


def aggrMetadata(metadata, groupByCol):
    """
    Aggregates metadata by taking the first value encountered in each
    aggregated group. These values are supposed to be unique per group.

    Args:
        metadata (pd.DataFrame): Input metadata to aggregate.
        groupByCol (str): The column to use for aggregation.

    Returns an aggregated pd.DataFrame.
    """
    metadataKeys = metadata.keys()
    aggMap = {"YEAR": "first", "ID": "first", "RGIId": "first"}
    if "GLWD_ID" in metadataKeys:
        aggMap["GLWD_ID"] = "first"
    if "GLWD_M_ID" in metadataKeys:
        aggMap["GLWD_M_ID"] = "first"
    if "POINT_LAT" in metadataKeys:
        aggMap["POINT_LAT"] = "first"
    if "POINT_LON" in metadataKeys:
        aggMap["POINT_LON"] = "first"
    if "PERIOD" in metadataKeys:
        aggMap["PERIOD"] = "first"
    if "POINT_ELEVATION" in metadataKeys:
        aggMap["POINT_ELEVATION"] = "first"
    if "ELEVATION_DIFFERENCE" in metadataKeys:
        aggMap["ELEVATION_DIFFERENCE"] = "first"
    metadataAggr = metadata.groupby(groupByCol).agg(aggMap)
    return metadataAggr


def aggrPredict(pred, idAggr, reduce="sum", out=None):
    """
    Performs temporal aggregation of the data.

    Args:
        pred (torch.Tensor): Predicted values
        idAggr (np.ndarray): Integer ID of the data used to aggregate them.
        reduce ('sum' or 'mean'): Reduction mode, default is 'sum'.

    Returns a torch.Tensor whose size is the number of unique IDs in idAggr.
    """
    assert isinstance(
        idAggr, (np.ndarray, torch.Tensor)
    ), "Argument idAggr must be either a numpy.ndarray or a torch.Tensor."
    assert isinstance(pred, torch.Tensor), "Argument predAggr must be a torch.Tensor."
    idAggrTorch = (
        torch.tensor(idAggr).to(pred.device)
        if isinstance(idAggr, np.ndarray)
        else idAggr
    )
    if out is None:
        out = torch.zeros(
            (len(torch.unique(idAggrTorch)),), device=pred.device, dtype=pred.dtype
        )
    # predSumAnnual = out.scatter_reduce(0, idAggrTorch, pred, reduce=reduce)
    # include_self=False: the initial value of `out` must not count as a sample,
    # otherwise a mean over N values is computed as sum / (N + 1)
    predSumAnnual = out.scatter_reduce_(
        0, idAggrTorch, pred, reduce=reduce, include_self=False
    )
    return predSumAnnual  # This shares memory with out


def aggrPredictGlwd(pred, idAggr, out=None):
    """
    Performs spatial aggregation of the data glacier wide.

    Args:
        pred (torch.Tensor): Predicted values
        idAggr (np.ndarray): Integer ID of the data used to aggregate them.

    Returns a torch.Tensor whose size is the number of unique IDs in idAggr.
    """
    assert isinstance(
        idAggr, (np.ndarray, torch.Tensor)
    ), "Argument idAggr must be either a numpy.ndarray or a torch.Tensor."
    assert isinstance(pred, torch.Tensor), "Argument pred must be a torch.Tensor."
    idAggrTorch = (
        torch.tensor(idAggr).to(pred.device)
        if isinstance(idAggr, np.ndarray)
        else idAggr
    )
    if out is None:
        out = torch.zeros(
            (len(np.unique(idAggr)),), device=pred.device, dtype=pred.dtype
        )
    # Aggregations of glacier wide values are always averaged. include_self=False: the
    # initial value of `out` must not count as a sample, otherwise the mean over N
    # values is computed as sum / (N + 1)
    predSumAnnualGlwd = out.scatter_reduce_(
        0, idAggrTorch, pred, reduce="mean", include_self=False
    )
    return predSumAnnualGlwd  # This shares memory with out


class CustomTorchNeuralNetRegressor(nn.Module):
    """
    Custom Torch neural network regressor that supports geodetic data aggregation.

    Args:
        module (torch.Module): Neural nework architecture with its associated weights.
    """

    def __init__(self, module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = module

    @property
    def needs_time_order(self):
        """Whether the forward pass takes the time order of the rows, see
        `TILikeSfcModel`."""
        return needs_time_order(self.module)

    def forward(self, x, *args):
        """
        Forward evaluation of the model, with the time order of the rows for a
        model that needs it (`forward_rows`).
        """
        return self.module(x, *args)

    def predict(self):
        # TODO: implement
        pass

    def evalMetrics(self):
        # TODO: implement
        pass

    def cumulative_pred(self):
        # TODO: implement
        pass

    def evaluate_group_pred(self, geodataloader, val=False):
        grouped_ids = pd.DataFrame()
        with torch.no_grad():
            iterator = geodataloader.glaciersVal if val else geodataloader.glaciers
            stakeMethod = geodataloader.stakesVal if val else geodataloader.stakes
            for g in iterator():
                # Get input features, metadata and ground truth
                stakes, metadata, point_balance, precomputed_meta = stakeMethod(g)

                # Make prediction
                stakesTorch = torch.tensor(stakes.astype(np.float32)).to(
                    geodataloader.device
                )
                pred = forward_rows(self, stakesTorch, precomputed_meta.get("sfc"))[
                    :, 0
                ]

                # Aggregate per stake and periods
                groundTruthTorch = torch.tensor(point_balance.astype(np.float32)).to(
                    geodataloader.device
                )
                int_id = torch.tensor(precomputed_meta["int_id"].astype(np.int64)).to(
                    geodataloader.device
                )
                nunique = precomputed_meta["nunique_ids"]
                trueMean = torch.zeros((nunique,), device=pred.device, dtype=pred.dtype)
                predSum = torch.zeros((nunique,), device=pred.device, dtype=pred.dtype)
                aggrPredict(groundTruthTorch, int_id, reduce="mean", out=trueMean)
                aggrPredict(pred, int_id, out=predSum)
                metadata = metadata.assign(ID_int=precomputed_meta["int_id"])
                grouped_ids_glacier = aggrMetadata(metadata, "ID_int")

                # Create grouped prediction DataFrame
                assert grouped_ids_glacier.index.name == "ID_int"
                d = {
                    "target": trueMean.cpu(),
                    "ID_int": grouped_ids_glacier.index,
                    "ID": grouped_ids_glacier.ID,
                    "pred": predSum.cpu(),
                    "PERIOD": grouped_ids_glacier.PERIOD,
                    "YEAR": grouped_ids_glacier.YEAR,
                    "RGIId": grouped_ids_glacier.RGIId,
                }
                if "POINT_ELEVATION" in grouped_ids_glacier.columns:
                    d["POINT_ELEVATION"] = grouped_ids_glacier.POINT_ELEVATION
                if "PERIOD" in grouped_ids_glacier.columns:
                    d["PERIOD"] = grouped_ids_glacier.PERIOD
                grouped_ids_glacier = pd.DataFrame(d)

                grouped_ids = pd.concat(
                    [grouped_ids, grouped_ids_glacier], ignore_index=True
                )

                if geodataloader.allStakesPerIter:
                    break

        if grouped_ids.shape[0] > 0:
            # ID_int does not make sense since it is used only to perform the aggregation with PyTorch, the variable to use is ID instead
            grouped_ids.drop(columns=["ID_int"], inplace=True)

        return grouped_ids
