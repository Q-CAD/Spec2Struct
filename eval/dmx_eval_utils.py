"""Forward-model-free structure/composition validity helpers.

Ported from DiffCSP's evaluation utilities, with the hydra/torch dependencies
stripped out:

  * chemical_symbols -> ase.data.chemical_symbols (identical Z-indexed list)
  * smact_validity / structure_validity keep DiffCSP's definitions verbatim,
    with the API updates noted inline for smact >= 4.0

Used by dmx2_roundtrip.py (validity columns), dmx2_template_gen.py and
generate_for_eval.py (lattice parameter conversion).
"""
import itertools

import numpy as np
import smact
from smact.screening import pauling_test

from ase.data import chemical_symbols  # Z-indexed; identical to DiffCSP's list


def lattices_to_params(lattice_matrix):
    """3x3 lattice matrix (rows = lattice vectors) -> (lengths[3], angles[3] deg)."""
    m = np.asarray(lattice_matrix, dtype=float)
    lengths = np.sqrt((m ** 2).sum(axis=1))
    angles = np.zeros(3)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        cos = np.dot(m[j], m[k]) / (lengths[j] * lengths[k])
        angles[i] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return lengths, angles


def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    """Charge-neutrality + electronegativity validity. `comp` = tuple of Z ints."""
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    # smact>=4.0 returns None (not []) for elements lacking oxidation-state data;
    # past the single-element/alloy shortcuts such a composition cannot be
    # charge-balanced -> compositionally invalid. (Older smact returned [].)
    if any(oc is None or len(oc) == 0 for oc in ox_combos):
        return False

    threshold = np.max(count)
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # smact>=4.0: neutral_ratios returns just the list of valid ratios
        # (older API returned a (bool, ratios) tuple). Charge balance exists iff
        # the returned list is non-empty.
        cn_r = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
        cn_e = len(cn_r) > 0
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def structure_validity(crystal, cutoff=0.5):
    """All pairwise distances > cutoff (A) and non-degenerate volume."""
    dist_mat = crystal.distance_matrix
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    return True
