import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict, defaultdict
import warnings

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch import nn

from .composition import (
    generate_features,
    _element_composition,
)

from sklearn.preprocessing import StandardScaler, Normalizer

import json

plt.rcParams.update({"font.size": 16})

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

# %%
fig_dir = r"figures/Classics/"
data_type_torch = torch.float32
data_type_np = np.float32


# %%
class CONSTANTS:
    def __init__(self):
        self.crab_red = "#f2636e"
        self.dense_blue = "#2c2cd5"
        self.colors = list(sns.color_palette("Set1", n_colors=7, desat=0.5))

        self.markers = [
            "o",
            "x",
            "s",
            "^",
            "D",
            "P",
            "1",
            "2",
            "3",
            "4",
            "p",
            "*",
            "h",
            "H",
            "+",
            "d",
            "|",
            "_",
        ]

        self.eps = ["oliynyk", "jarvis", "mat2vec", "onehot", "magpie", "random_200"]

        self.benchmark_props = [
            "aflow__ael_bulk_modulus_vrh",
            "aflow__ael_debye_temperature",
            "aflow__ael_shear_modulus_vrh",
            "aflow__agl_thermal_conductivity_300K",
            "aflow__agl_thermal_expansion_300K",
            "aflow__Egap",
            "aflow__energy_atom",
            "CritExam__Ed",
            "CritExam__Ef",
            "mp_bulk_modulus",
            "mp_elastic_anisotropy",
            "mp_e_hull",
            "mp_mu_b",
            "mp_shear_modulus",
            "OQMD_Bandgap",
            "OQMD_Energy_per_atom",
            "OQMD_Formation_Enthalpy",
            "OQMD_Volume_per_atom",
        ]

        self.benchmark_names = [
            "AFLOW Bulk modulus",
            "AFLOW Debye temperature",
            "AFLOW Shear modulus",
            "AFLOW Thermal conductivity",
            "AFLOW Thermal expansion",
            "AFLOW Band gap",
            "AFLOW Energy per atom",
            "Bartel Decomposition (Ed)",
            "Bartel Formation (Ef)",
            "MP Bulk modulus",
            "MP Elastic anisotropy",
            "MP Energy above convex hull",
            "MP Magnetic moment",
            "MP Shear modulus",
            "OQMD Band gap",
            "OQMD Energy per atom",
            "OQMD Formation enthalpy",
            "OQMD Volume per atom",
        ]

        self.matbench_props = [
            "castelli",
            "dielectric",
            "elasticity_log10(G_VRH)",
            "elasticity_log10(K_VRH)",
            "expt_gap",
            "expt_is_metal",
            "glass",
            "jdft2d",
            "mp_e_form",
            "mp_gap",
            "mp_is_metal",
            "phonons",
            "steels_yield",
        ]

        self.matbench_names = [
            "Castelli perovskites",
            "Refractive index",
            "Shear modulus (log10)",
            "Bulk modulus (log10)",
            "Experimental band gap",
            "Experimental metallicity",
            "Experimental glass formation",
            "DFT Exfoliation energy",
            "MP Formation energy",
            "MP Band gap",
            "MP Metallicity",
            "Phonon peak",
            "Steels yield",
        ]

        self.benchmark_names_dict = dict(
            zip(self.benchmark_props, self.benchmark_names)
        )
        self.matbench_names_dict = dict(zip(self.matbench_props, self.matbench_names))

        self.mb_units_dict = {
            "castelli": "eV/unit cell",
            "dielectric": "unitless",
            "elasticity_log10(G_VRH)": "log(GPa)",
            "elasticity_log10(K_VRH)": "log(GPa)",
            "expt_gap": "eV",
            "expt_is_metal": "binary",
            "glass": "binary",
            "jdft2d": "meV/atom",
            "mp_e_form": "eV/atom",
            "mp_gap": "eV",
            "mp_is_metal": "binary",
            "phonons": "$cm^{−1}$",
            "steels_yield": "MPa",
        }

        self.bm_units_dict = {
            "aflow__ael_bulk_modulus_vrh": None,
            "aflow__ael_debye_temperature": None,
            "aflow__ael_shear_modulus_vrh": None,
            "aflow__agl_thermal_conductivity_300K": None,
            "aflow__agl_thermal_expansion_300K": None,
            "aflow__Egap": None,
            "aflow__energy_atom": None,
            "CritExam__Ed": None,
            "CritExam__Ef": None,
            "mp_bulk_modulus": None,
            "mp_elastic_anisotropy": None,
            "mp_e_hull": None,
            "mp_mu_b": None,
            "mp_shear_modulus": None,
            "OQMD_Bandgap": None,
            "OQMD_Energy_per_atom": None,
            "OQMD_Formation_Enthalpy": None,
            "OQMD_Volume_per_atom": None,
        }

        self.mp_units_dict = {
            "energy_atom": "eV/atom",
            "ael_shear_modulus_vrh": "GPa",
            "ael_bulk_modulus_vrh": "GPa",
            "ael_debye_temperature": "K",
            "Egap": "eV",
            "agl_thermal_conductivity_300K": "W/m*K",
            "agl_log10_thermal_expansion_300K": "1/K",
        }

        self.mp_sym_dict = {
            "energy_atom": "$E_{atom}$",
            "ael_shear_modulus_vrh": "$G$",
            "ael_bulk_modulus_vrh": "$B$",
            "ael_debye_temperature": "$\\theta_D$",
            "Egap": "$E_g$",
            "agl_thermal_conductivity_300K": "$\\kappa$",
            "agl_log10_thermal_expansion_300K": "$\\alpha$",
        }

        self.classification_list = ["mp_is_metal", "expt_is_metal", "glass"]

        self.classic_models_dict = {
            "Ridge": "Ridge",
            "SGDRegressor": "SGD",
            "ExtraTreesRegressor": "ExtraTrees",
            "RandomForestRegressor": "RF",
            "AdaBoostRegressor": "AdaBoost",
            "GradientBoostingRegressor": "GradBoost",
            "KNeighborsRegressor": "kNN",
            "SVR": "SVR",
            "lSVR": "lSVR",
        }

        # fmt: off
        self.atomic_symbols = [ "None", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                               "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc",
                               "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga" "Ge",
                               "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc",
                               "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
                               "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
                               "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
                               "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
                               "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
                               "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
                               "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
        # fmt: on

        self.idx_symbol_dict = {(i): sym for i, sym in enumerate(self.atomic_symbols)}


# %%
def get_cbfv(data, elem_prop="oliynyk", scale=False, extend_features=False, formula_col="formula"):
    """
    Loads the compound csv file and featurizes it, then scales the features
    using StandardScaler.

    Parameters
    ----------
    path : str
        DESCRIPTION.
    elem_prop : str, optional
        DESCRIPTION. The default is 'oliynyk'.

    Returns
    -------
    X_scaled : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    formula : TYPE
        DESCRIPTION.

    """
    if type(data) is str:
        df = pd.read_csv(data, keep_default_na=False, na_values=[""])
    else:
        df = data
    if formula_col not in df.columns.values.tolist():
        df[formula_col] = df["cif_id"].str.split("_ICSD").str[0]
    # elem_prop = 'mat2vec'
    # elem_prop = 'oliynyk'
    mini = False
    # mini = True
    X, y, formula, skipped = generate_features(
        df, elem_prop, mini=mini, extend_features=extend_features
    )
    if scale:
        # scale each column of data to have a mean of 0 and a variance of 1
        scaler = StandardScaler()
        # normalize each row in the data
        normalizer = Normalizer()

        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(
            normalizer.fit_transform(X_scaled),
            columns=X.columns.values,
            index=X.index.values,
        )

        return X_scaled, y, formula, skipped
    else:
        return X, y, formula, skipped


# %%
def BCEWithLogitsLoss(output, log_std, target):
    loss = nn.functional.binary_cross_entropy_with_logits(output, target)
    return loss


def RobustL1(output, log_std, target):
    """
    Robust L1 loss using a lorentzian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    absolute = torch.abs(output - target)
    loss = np.sqrt(2.0) * absolute * torch.exp(-log_std) + log_std
    return torch.mean(loss)


def RobustL2(output, log_std, target):
    """
    Robust L2 loss using a gaussian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    squared = torch.pow(output - target, 2.0)
    loss = 0.5 * squared * torch.exp(-2.0 * log_std) + log_std
    return torch.mean(loss)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %%
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.float):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def count_gs_param_combinations(d):
    cnt_dict = OrderedDict({})
    # array = []
    if isinstance(d, (list)) and not isinstance(d, (bool)):
        return len(d), cnt_dict
    elif isinstance(d, (int, float, complex)) and not isinstance(d, (bool)):
        return 1, cnt_dict
    elif isinstance(d, (bool)) or isinstance(d, (str)):
        return 1, cnt_dict
    elif d is None:
        return 1, cnt_dict
    elif isinstance(d, (dict, OrderedDict)):
        keys = d.keys()
        for k in keys:
            array = []
            subd = d[k]
            array.append(count_gs_param_combinations(subd)[0])
            cnt = np.prod(array)
            cnt_dict[k] = cnt
        return np.prod(list(cnt_dict.values())), cnt_dict
    return cnt, cnt_dict


# %%
class Scaler:
    def __init__(self, data):
        self.data = torch.as_tensor(data)
        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)

    def scale(self, data):
        data = torch.as_tensor(data)
        data_scaled = (data - self.mean) / self.std
        return data_scaled

    def unscale(self, data_scaled):
        data_scaled = torch.as_tensor(data_scaled)
        data = data_scaled * self.std + self.mean
        return data

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


class DummyScaler:
    def __init__(self, data):
        self.data = torch.as_tensor(data)
        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)

    def scale(self, data):
        return torch.as_tensor(data)

    def unscale(self, data_scaled):
        return torch.as_tensor(data_scaled)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]

class PowerTorchTransformer:
    def __init__(self, data, lambda_, method='yeo-johnson', epsilon=1e-8):
        """
        Initialize the PowerTransformer with a provided lambda value.
        
        Args:
            data (numpy.ndarray): 1-dimensional data to be transformed.
            lambda_ (float): Optimal lambda parameter found by Scikit-Learn.
            method (str): Transformation method ('yeo-johnson' or 'box-cox').
        """
        self.lambda_ = lambda_
        self.method = method
        self.epsilon = epsilon
        self.data = torch.as_tensor(data)
        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)

    def scale(self, data):
        data = torch.as_tensor(data, dtype=torch.float32)
        data = torch.clamp(data, min=self.epsilon)
        if self.method == 'box-cox':
            data_transformed = (data ** self.lambda_ - 1) / self.lambda_ if self.lambda_ != 0 else torch.log(data)
        elif self.method == 'yeo-johnson':
            data_transformed = torch.where(
                data >= 0,
                ((data + 1) ** self.lambda_ - 1) / self.lambda_ if self.lambda_ != 0 else torch.log1p(data),
                -(((-data + 1) ** (2 - self.lambda_) - 1) / (2 - self.lambda_)) if self.lambda_ != 2 else -torch.log1p(-data)
            )
        if torch.isnan(data_transformed).any():
            raise ValueError("Error: NaN values detected in transformed data.")
        return data_transformed

    def unscale(self, data_transformed):
        data_transformed = torch.as_tensor(data_transformed, dtype=torch.float32)

        # Clamp data_transformed to avoid extreme values before inversion
        data_transformed = torch.clamp(data_transformed, min=-1e8, max=1e8)
        if torch.isnan(data_transformed).any():
            raise ValueError("Error: NaN values detected in scaled data.")

        if self.method == 'box-cox':
            data = (data_transformed * self.lambda_ + 1) ** (1 / self.lambda_) if self.lambda_ != 0 else torch.exp(data_transformed)
        elif self.method == 'yeo-johnson':
            data = torch.where(
                data_transformed >= 0,
                (data_transformed * self.lambda_ + 1) ** (1 / self.lambda_) - 1 if self.lambda_ != 0 else torch.expm1(data_transformed),
                1 - ((-data_transformed * (2 - self.lambda_) + 1) ** (1 / (2 - self.lambda_))) if self.lambda_ != 2 else -torch.expm1(-data_transformed)
            )
        
        # Check for NaN values
        if torch.isnan(data).any():
            raise ValueError("Error: NaN values detected in unscaled data. This suggests that the transformation is not invertible.\n See https://stats.stackexchange.com/questions/541748/simple-problem-with-box-cox-transformation-in-a-time-series-model")

        # Set NaN values to 0
        data[torch.isnan(data)] = 0
        

        return data

    def state_dict(self):
        return {"lambda": self.lambda_, "method": self.method, "mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
        self.lambda_ = state_dict["lambda"]
        self.method = state_dict["method"]

class LogScaler:
    def __init__(self, data, base=10):
        """
        Initialize the LogScaler with a specified logarithmic base.
        
        Args:
            base (float): The base of the logarithm. Default is 10 (log10).
        """
        self.data = torch.as_tensor(data)
        self.base = base
        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)
        if self.base <= 1:
            raise ValueError("Base of logarithm must be greater than 1.")

    def scale(self, data):
        """
        Apply the logarithmic transformation to the data.

        Args:
            data (torch.Tensor or list or numpy.ndarray): Positive data to transform.

        Returns:
            torch.Tensor: Log-scaled data.
        """
        data = torch.as_tensor(data, dtype=torch.float32)
        if (data <= 0).any():
            raise ValueError("Data contains non-positive values, which are not allowed for logarithmic scaling.")
        
        # Compute log with the specified base
        if self.base == 10:
            return torch.log10(data)
        elif self.base == torch.exp(torch.tensor(1.0)):  # Natural log (base e)
            return torch.log(data)
        else:
            return torch.log(data) / torch.log(torch.tensor(self.base, dtype=torch.float32))

    def unscale(self, data_scaled):
        """
        Reverse the logarithmic transformation to return to the original scale.

        Args:
            data_scaled (torch.Tensor or list or numpy.ndarray): Log-scaled data.

        Returns:
            torch.Tensor: Original data before scaling.
        """
        data_scaled = torch.as_tensor(data_scaled, dtype=torch.float32)
        
        # Compute inverse log with the specified base
        return torch.pow(torch.tensor(self.base, dtype=torch.float32), data_scaled)

    def state_dict(self):
        """
        Save the state of the scaler (log base).

        Returns:
            dict: A dictionary containing the log base.
        """
        return {"base": self.base, "mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        """
        Load the scaler's state (log base).

        Args:
            state_dict (dict): A dictionary containing the log base.
        """
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
        self.base = state_dict["base"]
        if self.base <= 1:
            raise ValueError("Base of logarithm must be greater than 1.")

# %%
class EDMDataset(Dataset):
    """
    Get X and y from EDM dataset.
    """

    def __init__(self, dataset, n_comp, extra_features=None, one_hot_names=None, one_hot_tensors=None):
        self.data = dataset
        self.n_comp = n_comp

        self.X = np.array(self.data[0])
        self.y = np.array(self.data[1])
        self.formula = np.array(self.data[2])
        if extra_features is None:
            self.extra_features = np.zeros((self.X.shape[0], 0))
        else:
            self.extra_features = extra_features.values

        if (one_hot_names is None) != (one_hot_tensors is None):
            raise ValueError("one_hot_names and one_hot_tensors must both be None or both not be None")
        
        if one_hot_names is None:
            self.one_hot_names = np.zeros((self.X.shape[0], 0))
        else:
            self.one_hot_names = one_hot_names

        if one_hot_tensors is  None:
            self.one_hot_tensors = np.zeros((self.X.shape[0], 0))
        else:
            self.one_hot_tensors = one_hot_tensors


        self.shape = [
            (self.X.shape),
            (self.y.shape),
            (self.formula.shape),
            (self.extra_features.shape),
            (self.one_hot_tensors.shape),
            len(self.one_hot_names)
        ]

    def __str__(self):
        string = f"EDMDataset with X.shape {self.X.shape}"
        return string

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx, :, :]
        y = self.y[idx]
        formula = self.formula[idx]
        extra_features = self.extra_features[idx]

        X = torch.as_tensor(X, dtype=data_type_torch)
        y = torch.as_tensor(y, dtype=data_type_torch)
        extra_features = torch.as_tensor(extra_features, dtype=data_type_torch)
        one_hot_tensors = torch.as_tensor(self.one_hot_tensors[idx], dtype=data_type_torch)

        return (X, y, formula, extra_features, one_hot_tensors, self.one_hot_names[idx])


def get_edm(data, n_elements="infer", inference=False, verbose=True, groupby=False, formula_col="formula"):
    """
    Build a element descriptor matrix.

    Parameters
    ----------
    data: str or DataFrame
        Filepath to data or DataFrame.

    Returns
    -------
    X_scaled : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    formula : TYPE
        DESCRIPTION.

    """
    # fmt: off
    all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                   'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                   'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                   'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                   'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                   'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                   'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                   'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                   'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                   'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                   'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    # fmt: on

    # mat_prop = 'phonons'
    # i = 0
    # path = rf'data\matbench_cv\{mat_prop}\test{i}.csv'
    if type(data) is str:
        df = pd.read_csv(data, keep_default_na=False, na_values=[""])
    else:
        df = data
    if formula_col not in df.columns.values.tolist():
        df[formula_col] = df["cif_id"].str.split("_ICSD").str[0]

    df = df.copy()
    df.loc[:, "count"] = [
        len(_element_composition(form)) for form in df.formula.values.tolist()
    ]
    # df = df[df["count"] != 1]  # drop pure elements
    if not inference and groupby:
        df = df.groupby(by=formula_col).mean().reset_index()  # mean of duplicates

    list_ohm = [OrderedDict(_element_composition(form)) for form in df[formula_col]]
    list_ohm = [
        OrderedDict(sorted(mat.items(), key=lambda x: -x[1])) for mat in list_ohm
    ]

    y = df["target"].values.astype(data_type_np)
    formula = df[formula_col].values
    if n_elements == "infer":
        n_elements = 16

    edm_array = np.zeros(
        shape=(len(list_ohm), n_elements, len(all_symbols) + 1), dtype=data_type_np
    )
    elem_num = np.zeros(shape=(len(list_ohm), n_elements), dtype=data_type_np)
    elem_frac = np.zeros(shape=(len(list_ohm), n_elements), dtype=data_type_np)
    for i, comp in enumerate(
        tqdm(list_ohm, desc="Generating EDM", unit="formulae", disable=not verbose)
    ):
        for j, (elem, count) in enumerate(list_ohm[i].items()):
            if j == n_elements:
                # Truncate EDM representation to n_elements
                break
            try:
                edm_array[i, j, all_symbols.index(elem) + 1] = count
                elem_num[i, j] = all_symbols.index(elem) + 1
            except ValueError:
                print(f"skipping composition {comp}")

    # Scale features
    for i in range(edm_array.shape[0]):
        frac = edm_array[i, :, :].sum(axis=-1) / (edm_array[i, :, :].sum(axis=-1)).sum()
        elem_frac[i, :] = frac

    if n_elements == 16:
        n_elements = np.max(np.sum(elem_frac > 0, axis=1, keepdims=True))
        elem_num = elem_num[:, :n_elements]
        elem_frac = elem_frac[:, :n_elements]

    elem_num = elem_num.reshape(elem_num.shape[0], elem_num.shape[1], 1)
    elem_frac = elem_frac.reshape(elem_frac.shape[0], elem_frac.shape[1], 1)
    out = np.concatenate((elem_num, elem_frac), axis=1)

    return out, y, formula


# %%
class EDM_CsvLoader:
    def __init__(
        self,
        data,
        extra_features=None,
        one_hot_tensors=None,
        one_hot_names=None,
        batch_size=64,
        groupby=False,
        random_state=0,
        shuffle=True,
        pin_memory=True,
        n_elements=6,
        inference=False,
        verbose=True,
    ):
        """
        Parameters
        ----------
        data: str or DataFrame
            name of csv file containing cif and properties or DataFrame
        extra_features: str or None
            names of extended features
        one_hot_tensors: Tensor or None
            one hot tensors
        csv_val: str
            name of csv file containing cif and properties
        val_frac: float, optional (default=0.75)
            train/val ratio if val_file not given
        batch_size: float, optional (default=64)
            Step size for the Gaussian filter
        groupby: bool, optional
            Whether to reduce repeat formulas to a unique set, by default False.
        random_state: int, optional (default=123)
            Random seed for sampling the dataset. Only used if validation data is
            not given.
        shuffle: bool (default=True)
            Whether to shuffle the datasets or not
        """
        self.data = data
        self.main_data = list(
            get_edm(
                self.data,
                n_elements=n_elements,
                inference=inference,
                verbose=verbose,
                groupby=groupby,
            )
        )

        self.extra_features = extra_features
        self.one_hot_tensors = one_hot_tensors
        self.one_hot_names = one_hot_names
        self.n_train = len(self.main_data[0])
        self.n_elements = self.main_data[0].shape[1] // 2

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.random_state = random_state

    def get_data_loaders(self, inference=False):
        """
        Input the dataset, get train test split
        """
        shuffle = not inference  # don't shuffle data when inferencing
        pred_dataset = EDMDataset(self.main_data, self.n_elements, self.extra_features, self.one_hot_names, self.one_hot_tensors)
        pred_loader = DataLoader(
            pred_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
        )
        return pred_loader


# %%
class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning:
        Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this
            into Adam. Useful for comparison purposes.
        _Large Batch Optimization for Deep Learning: Training BERT in 76
            minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0,
        adam=False,
        min_trust=None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if min_trust and not 0.0 <= min_trust < 1.0:
            raise ValueError(f"Minimum trust range from 0 to 1: {min_trust}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.adam = adam
        self.min_trust = min_trust
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    err_msg = (
                        "Lamb does not support sparse gradients, "
                        + "consider SparseAdam instad."
                    )
                    raise RuntimeError(err_msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_((1 - beta1) * grad)
                # v_t
                # exp_avg_sq.mul_(beta2).addcmul_((1 - beta2) * grad *
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group[
                    "lr"
                ]  # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"] != 0:
                    adam_step.add_(group["weight_decay"], p.data)

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                if self.min_trust:
                    trust_ratio = max(trust_ratio, self.min_trust)
                state["weight_norm"] = weight_norm
                state["adam_norm"] = adam_norm
                state["trust_ratio"] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(-step_size * trust_ratio * adam_step)

        return loss


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if "slow_buffer" not in param_state:
                param_state["slow_buffer"] = torch.empty_like(fast_p.data)
                param_state["slow_buffer"].copy_(fast_p.data)
            slow = param_state["slow_buffer"]
            slow.add_(group["lookahead_alpha"] * (fast_p.data - slow))
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group["lookahead_step"] += 1
            if group["lookahead_step"] % group["lookahead_k"] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            "state": state_dict["state"],
            "param_groups": state_dict["param_groups"],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if "slow_state" not in state_dict:
            print("Loading state_dict from optimizer without Lookahead applied.")
            state_dict["slow_state"] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict[
                "param_groups"
            ],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = (
            self.base_optimizer.param_groups
        )  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)

# %%
if __name__ == "__main__":
    os.makedirs(fig_dir, exist_ok=True)
