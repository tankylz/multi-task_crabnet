import numpy
import pandas
import json
import ast
from tqdm import tqdm
from sklearn import preprocessing
# from mendeleev import get_table # this is refractored in the new version of mendeleev (Mar 14 2021 v0.7.0)
from mendeleev.fetch import fetch_table # replaced with this
from mendeleev import element


atom_nums = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O':8, 'F': 9, 'Ne': 10,
             'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
             'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
             'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
             'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
             'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
             'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
             'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
             'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
             'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100}
atom_syms = {v: k for k, v in atom_nums.items()}
# elem_feat_names = ['atomic_number', 'period', 'en_pauling', 'covalent_radius_bragg',
#                    'electron_affinity', 'atomic_volume', 'atomic_weight', 'fusion_heat'] # in the example code
elem_feat_names = [
    "atomic_weight",
    "atomic_weight_uncertainty",
    "atomic_radius",
    "atomic_radius_rahm",
    "covalent_radius_cordero",
    "covalent_radius_pyykko",
    "vdw_radius",
    "vdw_radius_uff",
    "vdw_radius_mm3",
    "vdw_radius_alvarez",
    "density",
    "lattice_constant",
    "specific_heat_capacity",
    "fusion_heat",
    "evaporation_heat",
    "heat_of_formation",
    "thermal_conductivity",
    "atomic_number",
    "electron_affinity",
    "period",
    "en_ghosh",
    "en_pauling",
    "en_allen",
    "dipole_polarizability",
    "c6_gb",
    "abundance_crust",
    "abundance_sea"] # edited based off supplementary information of the paper
n_atom_feats = len(elem_feat_names) + 1


class MatComp:
    def __init__(self, comp, mat_feats, conds, target, idx):
        self.comp = comp
        self.mat_feats = mat_feats
        self.conds = conds
        self.target = target
        self.idx = idx


def load_elem_feats():
    tb_atom_feats = fetch_table('elements')
    atom_feats = numpy.nan_to_num(numpy.array(tb_atom_feats[elem_feat_names]))[:100, :]
    extra_props = numpy.zeros((atom_feats.shape[0], 4))

    for i in range(0, extra_props.shape[0]):
        ion_eng = element(i + 1).ionenergies
        atom_vol = element(i + 1).atomic_volume

        # accounting for allotropes
        pt =  element(i + 1).phase_transitions
        melt_pts = [p.melting_point for p in pt if p.melting_point is not None]
        boil_pts = [p.boiling_point for p in pt if p.boiling_point is not None]
        melt_pt = numpy.mean(melt_pts) if melt_pts else None
        boil_pt = numpy.mean(boil_pts) if boil_pts else None

        if 1 in ion_eng: # take first ionization energy
            extra_props[i, 0] = ion_eng[1]
        else:
            extra_props[i, 0] = 0

        if atom_vol is not None:
            extra_props[i, 1] = atom_vol
        else:
            extra_props[i, 1] = 0

        if boil_pt is not None:
            extra_props[i, 2] = boil_pt
        else:
            extra_props[i, 2] = 0

        if melt_pt is not None:
            extra_props[i, 3] = melt_pt
        else:
            extra_props[i, 3] = 0

    return preprocessing.scale(numpy.hstack([atom_feats, extra_props]))


def load_mats_from_comps(dataset_file_name, comp_idx, target_idx, cond_idx=None, norm_target=False):
    data = numpy.array(pandas.read_excel(dataset_file_name))
    comps = data[:, comp_idx]
    targets = data[:, target_idx].reshape(-1, 1)
    mats = list()
    elem_feats = load_elem_feats()

    if norm_target:
        target_mean = numpy.mean(targets)
        target_std = numpy.std(targets)
        targets = preprocessing.scale(targets)

    for i in tqdm(range(0, comps.shape[0])):
        conds = None

        if cond_idx is None:
            mat_feats = calc_mat_feat(elem_feats, comps[i])
        else:
            mat_feats = numpy.hstack([calc_mat_feat(elem_feats, comps[i]), data[i, cond_idx]])
            conds = data[i, cond_idx]

        mats.append(MatComp(comps[i], mat_feats, conds, targets[i], idx=i))

    if norm_target:
        return mats, target_mean, target_std
    else:
        return mats


def get_mat_feats(dataset_file_name, comp_idx, target_idx, cond_idx=None):
    data = numpy.array(pandas.read_excel(dataset_file_name))
    comps = data[:, comp_idx]
    targets = data[:, target_idx].reshape(-1, 1)
    mat_feats = list()
    elem_feats = load_elem_feats()

    for i in tqdm(range(0, comps.shape[0])):
        if cond_idx is None:
            mat_feats.append(calc_mat_feat(elem_feats, comps[i]))
        else:
            mat_feats.append(numpy.hstack([calc_mat_feat(elem_feats, comps[i]), data[i, cond_idx].astype(numpy.float)]))

    if cond_idx is None:
        return numpy.hstack([numpy.vstack(mat_feats), targets]), comps
    else:
        return numpy.hstack([numpy.vstack(mat_feats), targets]), comps, data[:, cond_idx]


def calc_mat_feat(elem_feats, comp):
    elems = ast.literal_eval(str(parse_formula(comp)))
    e_sum = numpy.sum([float(elems[key]) for key in elems])
    w_sum_vec = numpy.zeros(elem_feats.shape[1])
    atom_feats = list()

    for e in elems:
        atom_vec = elem_feats[atom_nums[e] - 1, :]
        atom_feats.append(atom_vec)
        w_sum_vec += (float(elems[e]) / e_sum) * atom_vec

    return numpy.hstack([w_sum_vec, numpy.std(atom_feats, axis=0), numpy.min(atom_feats, axis=0), numpy.max(atom_feats, axis=0)])

def parse_formula(comp):
    elem_dict = dict()
    elem = comp[0]
    num = ''

    for i in range(1, len(comp)):
        if comp[i].islower() and num == '':
            elem += comp[i]
        elif comp[i].isupper():
            if num == '':
                elem_dict[elem] = 1.0
            else:
                elem_dict[elem] = float(num)

            elem = comp[i]
            num = ''
        elif comp[i].isnumeric() or comp[i] == '.' or comp[i] == '-' or (comp[i] == 'e' and num != ''):
            num += comp[i]

        if i == len(comp) - 1:
            if num == '':
                elem_dict[elem] = 1.0
            else:
                elem_dict[elem] = float(num)

    return elem_dict


def dict_to_comp(comp_dict):
    comp = ''

    for e in comp_dict.keys():
        comp += e + str(comp_dict[e])

    return comp


def load_emb_elem_feats(file_elem_embs):
    with open(file_elem_embs) as f:
        data = json.load(f)

    elem_feats = list()

    for elem in data.keys():
        elem_feats.append(data[elem])

    return numpy.vstack(elem_feats)
