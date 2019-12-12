import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from sklearn.metrics import confusion_matrix

# Labels for species that were put in the wrong image folders in the dataset.
invalid_labels = [
    122, # pinus virginia
    134, # prunus virginina
    150, # quercus_muehlenbergii
    130, # prunus_sargentii
]

# Mapping from labels to species for the LeafSnap image embeddings
labels_to_species = {
    0: 'abies_concolor',
    1: 'abies_nordmanniana',
    2: 'acer_campestre',
    3: 'acer_ginnala',
    4: 'acer_griseum',
    5: 'acer_negundo',
    6: 'acer_palmatum',
    7: 'acer_pensylvanicum',
    8: 'acer_platanoides',
    9: 'acer_pseudoplatanus',
    10: 'acer_rubrum',
    11: 'acer_saccharinum',
    12: 'acer_saccharum',
    13: 'aesculus_flava',
    14: 'aesculus_glabra',
    15: 'aesculus_hippocastamon',
    16: 'aesculus_pavi',
    17: 'ailanthus_altissima',
    18: 'albizia_julibrissin',
    19: 'amelanchier_arborea',
    20: 'amelanchier_canadensis',
    21: 'amelanchier_laevis',
    22: 'asimina_triloba',
    23: 'betula_alleghaniensis',
    24: 'betula_jacqemontii',
    25: 'betula_lenta',
    26: 'betula_nigra',
    27: 'betula_populifolia',
    28: 'broussonettia_papyrifera',
    29: 'carpinus_betulus',
    30: 'carpinus_caroliniana',
    31: 'carya_cordiformis',
    32: 'carya_glabra',
    33: 'carya_ovata',
    34: 'carya_tomentosa',
    35: 'castanea_dentata',
    36: 'catalpa_bignonioides',
    37: 'catalpa_speciosa',
    38: 'cedrus_atlantica',
    39: 'cedrus_deodara',
    40: 'cedrus_libani',
    41: 'celtis_occidentalis',
    42: 'celtis_tenuifolia',
    43: 'cercidiphyllum_japonicum',
    44: 'cercis_canadensis',
    45: 'chamaecyparis_pisifera',
    46: 'chamaecyparis_thyoides',
    47: 'chionanthus_retusus',
    48: 'chionanthus_virginicus',
    49: 'cladrastis_lutea',
    50: 'cornus_florida',
    51: 'cornus_kousa',
    52: 'cornus_mas',
    53: 'corylus_colurna',
    54: 'crataegus_crus-galli',
    55: 'crataegus_laevigata',
    56: 'crataegus_phaenopyrum',
    57: 'crataegus_pruinosa',
    58: 'crataegus_viridis',
    59: 'cryptomeria_japonica',
    60: 'diospyros_virginiana',
    61: 'eucommia_ulmoides',
    62: 'evodia_daniellii',
    63: 'fagus_grandifolia',
    64: 'ficus_carica',
    65: 'fraxinus_americana',
    66: 'fraxinus_nigra',
    67: 'fraxinus_pennsylvanica',
    68: 'ginkgo_biloba',
    69: 'gleditsia_triacanthos',
    70: 'gymnocladus_dioicus',
    71: 'halesia_tetraptera',
    72: 'ilex_opaca',
    73: 'juglans_cinerea',
    74: 'juglans_nigra',
    75: 'juniperus_virginiana',
    76: 'koelreuteria_paniculata',
    77: 'larix_decidua',
    78: 'liquidambar_styraciflua',
    79: 'liriodendron_tulipifera',
    80: 'maclura_pomifera',
    81: 'magnolia_acuminata',
    82: 'magnolia_denudata',
    83: 'magnolia_grandiflora',
    84: 'magnolia_macrophylla',
    85: 'magnolia_soulangiana',
    86: 'magnolia_stellata',
    87: 'magnolia_tripetala',
    88: 'magnolia_virginiana',
    89: 'malus_angustifolia',
    90: 'malus_baccata',
    91: 'malus_coronaria',
    92: 'malus_floribunda',
    93: 'malus_hupehensis',
    94: 'malus_pumila',
    95: 'metasequoia_glyptostroboides',
    96: 'morus_alba',
    97: 'morus_rubra',
    98: 'nyssa_sylvatica',
    99: 'ostrya_virginiana',
    100: 'oxydendrum_arboreum',
    101: 'paulownia_tomentosa',
    102: 'phellodendron_amurense',
    103: 'picea_abies',
    104: 'picea_orientalis',
    105: 'picea_pungens',
    106: 'pinus_bungeana',
    107: 'pinus_cembra',
    108: 'pinus_densiflora',
    109: 'pinus_echinata',
    110: 'pinus_flexilis',
    111: 'pinus_koraiensis',
    112: 'pinus_nigra',
    113: 'pinus_parviflora',
    114: 'pinus_peucea',
    115: 'pinus_pungens',
    116: 'pinus_resinosa',
    117: 'pinus_rigida',
    118: 'pinus_strobus',
    119: 'pinus_sylvestris',
    120: 'pinus_taeda',
    121: 'pinus_thunbergii',
    122: 'pinus_virginiana',
    123: 'pinus_wallichiana',
    124: 'platanus_acerifolia',
    125: 'platanus_occidentalis',
    126: 'populus_deltoides',
    127: 'populus_grandidentata',
    128: 'populus_tremuloides',
    129: 'prunus_pensylvanica',
    130: 'prunus_sargentii',
    131: 'prunus_serotina',
    132: 'prunus_serrulata',
    133: 'prunus_subhirtella',
    134: 'prunus_virginiana',
    135: 'prunus_yedoensis',
    136: 'pseudolarix_amabilis',
    137: 'ptelea_trifoliata',
    138: 'pyrus_calleryana',
    139: 'quercus_acutissima',
    140: 'quercus_alba',
    141: 'quercus_bicolor',
    142: 'quercus_cerris',
    143: 'quercus_coccinea',
    144: 'quercus_falcata',
    145: 'quercus_imbricaria',
    146: 'quercus_macrocarpa',
    147: 'quercus_marilandica',
    148: 'quercus_michauxii',
    149: 'quercus_montana',
    150: 'quercus_muehlenbergii',
    151: 'quercus_nigra',
    152: 'quercus_palustris',
    153: 'quercus_phellos',
    154: 'quercus_robur',
    155: 'quercus_rubra',
    156: 'quercus_shumardii',
    157: 'quercus_stellata',
    158: 'quercus_velutina',
    159: 'quercus_virginiana',
    160: 'robinia_pseudo-acacia',
    161: 'salix_babylonica',
    162: 'salix_caroliniana',
    163: 'salix_matsudana',
    164: 'salix_nigra',
    165: 'sassafras_albidum',
    166: 'staphylea_trifolia',
    167: 'stewartia_pseudocamellia',
    168: 'styrax_japonica',
    169: 'styrax_obassia',
    170: 'syringa_reticulata',
    171: 'taxodium_distichum',
    172: 'tilia_americana',
    173: 'tilia_cordata',
    174: 'tilia_europaea',
    175: 'tilia_tomentosa',
    176: 'toona_sinensis',
    177: 'tsuga_canadensis',
    178: 'ulmus_americana',
    179: 'ulmus_glabra',
    180: 'ulmus_parvifolia',
    181: 'ulmus_procera',
    182: 'ulmus_pumila',
    183: 'ulmus_rubra',
    184: 'zelkova_serrata',
}

images_dir = 'images/'


def get_valid_indices(labels):
    """
    Determine the indices for labels that are not corrupted in the dataset.

    :param - labels: ground-truth labels
    return: indices from labels excluding the incorrectly labeled species
    """
    idxs = (labels != invalid_labels[0])
    for l in invalid_labels[1:]:
        idxs = idxs & (labels != l)
    return idxs


def plot_cm(predictions, labels, path):
    """
    Plot confusion matrix.

    :param - predictions: list for which index i holds the prediction for embedding i
    :param - labels: list for which index i holds ground truth label for embedding i
    :param - path: path to which to save the plotted confusion matrix
    """
    cm = calculate_cm(predictions, labels)
    cm = normalize_cm(cm)
    plot_heatmap(cm, path, 'Predicted Label', 'True Label')


def calculate_cm(predictions, labels):
    """
    Calculate confusion matrix.

    :param - predictions: list for which index i holds the prediction for embedding i
    :param - labels: list for which index i holds ground truth label for embedding i
    return: 2D numpy array representing the confusion matrix
    """
    return confusion_matrix(labels, predictions)


def normalize_cm(confusion_matrix):
    """
    Row-normalize a confusion matrix such that each row sums to 1.

    :param - confusion_matrix: 2D numpy array representing a confusion matrix.
    return: 2D numpy array representing row-normalized confusion matrix
    """
    return confusion_matrix / np.sum(confusion_matrix, axis=1)[:, None]


def plot_heatmap(matrix, path, xlabel=None, ylabel=None):
    """
    Plots the provided matrix as a heatmap.

    :param - matrix: 2D numpy array representing values to plot in heatmap
    :param - title: title for the plot
    :param - path: save path for plot
    :param - xlabel: label for x-axis
    :param - ylabel: label for y-axis
    """
    plt.close('all')
    df_cm = pd.DataFrame(matrix)
    _ = plt.figure(figsize=(10, 7))
    heatmap = sns.heatmap(df_cm)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.tight_layout()
    make_dir(images_dir)
    plt.savefig(path)


def make_dir(dir_name):
    '''
    Creates a directory if it doesn't yet exist.
    
    :param - dir_name: Name of directory to be created.
    '''
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, dir_name + '/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
