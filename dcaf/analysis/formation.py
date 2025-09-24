import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from amuse.units import units, constants
from amuse.datamodel import Particles
import os, re
from amuse.io import read_set_from_file
from amuse.units import units
from natsort import natsort

def local_kinematics( new_stars, stars, k=6):
    #TODO this will show outliers if the 6th neighbour is part of a binary
    """
    For each new star, using its k nearest neighbours (default 6) among `stars`,
    compute and return:
      - corr: cosine similarity between v_new and local COM velocity of neighbours
      - v_esc: local escape speed from enclosed neighbour mass at r_k
      - v_rel: |v_new - v_com| (relative speed w.r.t. neighbour COM flow)
      - r_k  : distance to the k-th nearest neighbour (adaptive local scale)
      - M_encl: enclosed mass of the k neighbours
      - d_closest: distance to the closest neighbour (1st NN)

    """
    n_new = len(new_stars)
    n_old = len(stars)
    if n_new == 0:
        return (np.array([]), 
                np.array([]) | units.kms,
                np.array([]) | units.kms,
                np.array([]) | units.pc,
                np.array([]) | units.MSun,
                np.array([]) | units.pc)
    if n_old == 0:
        nan = np.full(n_new, np.nan)
        return (nan,
                nan | units.kms,
                nan | units.kms,
                nan | units.pc,
                nan | units.MSun,
                nan | units.pc)

    k_eff = min(k, n_old)

    # KD-tree over existing stars
    X_old = np.column_stack([
        stars.x.value_in(units.pc),
        stars.y.value_in(units.pc),
        stars.z.value_in(units.pc),
    ])
    tree = cKDTree(X_old)

    X_new = np.column_stack([
        new_stars.x.value_in(units.pc),
        new_stars.y.value_in(units.pc),
        new_stars.z.value_in(units.pc),
    ])

    d_pc, idx = tree.query(X_new, k=k_eff)
    if d_pc.ndim == 1:  # handle k_eff == 1
        d_pc = d_pc[:, None]
        idx = idx[:, None]

    # Neighbour properties (numeric arrays)
    m = stars.mass.value_in(units.MSun)[idx]            # (N_new, k_eff)
    vx = stars.vx.value_in(units.kms)[idx]
    vy = stars.vy.value_in(units.kms)[idx]
    vz = stars.vz.value_in(units.kms)[idx]
    Vnei = np.stack([vx, vy, vz], axis=2)               # (N_new, k_eff, 3)

    # Enclosed mass M_encl (sum of the k_eff neighbour masses)
    M_encl_num = m.sum(axis=1)                          # Msun

    # r_k is the distance to the k_eff-th neighbour per star
    # Sort distances per row to get the k-th order statistic
    order = np.argsort(d_pc, axis=1)
    d_sorted = np.take_along_axis(d_pc, order, axis=1)  # (N_new, k_eff)
    r_k_num = d_sorted[:, -1]                           # pc
    d_closest_num = d_sorted[:, 0]                      # pc

    #get neighbour density for storing it
    d_self, _ = tree.query(X_old, k=k+1)   # include self at index 0
    r6_all = d_self[:, k]                  # 6th neighbour distance for each star
    n_all = k / ((4.0/3.0) * np.pi * r6_all**3)   # 1/pc^3 (numeric)
    idx_closest = idx[:, 0]   # (N_new,)
    n_closest = n_all[idx_closest] | (1/units.pc**3)

    # Neighbour COM (center-of-mass) velocity (mass-weighted)
    Wsum = m.sum(axis=1, keepdims=True) + 1e-30
    Vcom = (m[:, :, None] * Vnei).sum(axis=1) / Wsum    # (N_new, 3) km/s

    # New-star velocities
    Vnew = np.column_stack([
        new_stars.vx.value_in(units.kms),
        new_stars.vy.value_in(units.kms),
        new_stars.vz.value_in(units.kms),
    ])                                                 # (N_new, 3)

    # Correlation (cosine) between v_new and Vcom
    vnew_norm = np.linalg.norm(Vnew, axis=1) + 1e-30
    vcom_norm = np.linalg.norm(Vcom, axis=1) + 1e-30
    corr = (Vnew * Vcom).sum(axis=1) / (vnew_norm * vcom_norm)

    # Relative speed |v_new - Vcom|
    v_rel_num = np.linalg.norm(Vnew - Vcom, axis=1)     # km/s
    # --- NEW: Speed ratio (magnitudes) ---
    speed_ratio = vnew_norm / vcom_norm

    # Local escape speed at r_k from enclosed mass 
    # v_esc^2 = 2 G M_encl / r_k
    M_encl = M_encl_num | units.MSun
    r_k = r_k_num | units.pc
    v_esc = (2 * constants.G * M_encl / r_k).sqrt().as_quantity_in(units.kms)

    # Attach units to remaining outputs
    v_rel = v_rel_num | units.kms
    d_closest = d_closest_num | units.pc

    return corr, v_esc, v_rel, r_k, M_encl, d_closest, n_closest, speed_ratio


def compute_formation_kinematics(files, output_csv, file_format="amuse"):
    """
      • NEW stars = IDs in current snapshot but NOT in previous.
      • Neighbour/kinematics set = stars in CURRENT snapshot that ALSO existed before (current ∩ previous).
      • Calls: corr, v_esc, v_rel, r_k, M_encl, d_closest = local_kinematics(new_stars, neighbours)
      • Writes one row per NEW star with units: pc, km/s, Msun; plus snapshot_id, time_Myr, star_id, k_eff (if your function returns it, ignore here).

    NOTE: This function does NOT implement kinematics; it just orchestrates I/O and set logic.
    """

    rows = []
    prev_ids = np.array([], dtype=np.int64)
    files = natsort.natsorted(files)

    prev_stars = read_set_from_file(files[0])
    for idx, fpath in enumerate(files[1:]):
        current_stars = read_set_from_file(fpath)

        cur_ids = set(current_stars.key)
        # clean disapearing particles
        if hasattr(prev_stars,'id'):
            mask = np.isin(prev_stars.key, list(cur_ids))
            prev_stars = prev_stars[mask]
        else:
            prev_stars = current_stars
            continue


        new_stars = (current_stars - prev_stars).copy()
        neigh_stars = (current_stars - new_stars).copy()

        if len(new_stars) > 0 and len(neigh_stars) > 0:
            corr, v_esc, v_rel, r_k, M_encl, d_closest,n_closest,speed_ratio = local_kinematics(new_stars, neigh_stars)

            # Time and snapshot_id
            t_attr = getattr(current_stars.collection_attributes, "model_time", None)
            time_Myr = t_attr.as_quantity_in(units.Myr).value_in(units.Myr) if t_attr is not None else np.nan
            m = re.search(r"(\d+)", os.path.basename(fpath))
            snapshot_id = int(m.group(1)) if m else idx

            # IDs of new stars for traceability
            if hasattr(new_stars, "id"):
                sid = np.asarray(new_stars.id)
            elif hasattr(new_stars, "key"):
                sid = np.asarray(new_stars.key)
            else:
                sid = np.arange(len(new_stars), dtype=int)

            # Collect one row per NEW star (numeric values with unit-suffixed columns)
            rows.append(pd.DataFrame({
                "snapshot_id":   np.full(len(new_stars), snapshot_id, dtype=int),
                "time":      np.full(len(new_stars), time_Myr, dtype=float),
                "star_id":       sid,
                "corr":          np.asarray(corr, dtype=float),
                "v_esc":     v_esc.value_in(units.kms),
                "v_rel":     v_rel.value_in(units.kms),
                "r6":         r_k.value_in(units.pc),
                "M_encl":   M_encl.value_in(units.MSun),
                "d_closest":  d_closest.value_in(units.pc),
                "n_closest":  n_closest.value_in(units.pc**-3),
                "speed_ratio":  np.asarray(speed_ratio),
            }))

        # advance window
        #prev_ids = cur_ids
        prev_stars = current_stars

    out = (pd.concat(rows, ignore_index=True) if rows else
           pd.DataFrame(columns=["snapshot_id","time","star_id","corr",
                                 "v_esc","v_rel","r6","M_encl","d_closest",
                                 "n_closest","speed_ratio"]))
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out.to_csv(output_csv, index=False)
