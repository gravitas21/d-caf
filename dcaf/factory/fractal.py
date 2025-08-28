import numpy as np
from amuse.units import units
from amuse.datamodel import Particles

"""
Fractal dimension tools for dynamic formation of stars.

This module implements a box counting fractal dimension estimator and two 
fractal dimension generators with different methods.


1) Single-level, alpha-weighted method (fixed grid):
   - Choose a grid with B boxes per axis over current bounds.
   - Force currently occupied boxes to be valid.
   - Use D_target to determine how many *additional* boxes should be made valid.
   - Assign new stars to valid boxes using a parameter-free uniform choice over
     valid boxes, or an occupancy bias exponent alpha (0..2) if desired.
   - Sample positions uniformly within chosen boxes.
   Pro: simple and fast
   Con: requires an arbitrary alpha parameter to bias towards already occupied
   cells

2) Hierarchical, global cascade:
   - Build levels with subdivision factor m (default 2) up to L_max.
   - At each level ell, the target number of occupied cells is T_ell = ceil(m^{ell*D}).
   - Starting from the root and proceeding level-by-level, select exactly the
     number of *new* empty cells needed to meet T_ell within the active region,
     always keeping already-occupied cells valid and discarding irrelevant empty
     branches. Place one star in each newly activated cell (ensuring they become
     occupied), then continue to finer levels. Any remaining stars go into the
     active leaves at L_max.
    Pro: do not require additional parameters
    Con: more complex and potentially slower.

Both methods assign velocities by sampling k-neighbour local COM velocity and
axis-wise 1D dispersions (Gaussian), with optional mass weighting and a small
floor to avoid zero-variance.
"""

import numpy as np

def boxcount_dimension_3d(points,
                          n_scales=8,
                          min_boxes_per_axis=4,
                          return_fit=False,
                          robust_fraction=None):
    """
    Estimate the fractal (box-counting) dimension of a 3D point cloud,
    with optional robust cube selection and diagnostics.

    Parameters
    ----------
    points : (N,3) array-like
        Star positions (any units). NaNs are ignored.
    n_scales : int
        Number of box sizes to test (powers of 2).
    min_boxes_per_axis : int
        Smallest number of boxes per axis at the coarsest scale.
    return_fit : bool
        If True, also return a dict with diagnostics (see below).
    robust_fraction : float or None
        If provided (e.g., 0.90), first select a *cube* centered at the
        coordinate-wise median that contains this fraction of points, measured
        by Chebyshev/max-norm radius. Only points inside this cube are used for
        the measurement. The cube edges are then used to normalize to [0,1]^3.
        If None, use full min/max rectangular bounds.

    Returns
    -------
    Dhat : float
        Estimated fractal (box-counting) dimension.
    info : dict (optional if return_fit=True)
        Keys for downstream plotting/animation:
          - 'mask': (N,) bool, which input points were used
          - 'center': (3,), the median center (robust) or mid of full bounds
          - 'halfwidth': float, cube half-width
          - 'bounds_mins': (3,), lower bounds used for normalization
          - 'bounds_maxs': (3,), upper bounds used for normalization
          - 'used_fraction': float in (0,1], fraction of points used
          - 'bs': (n_scales,), boxes-per-axis at each scale
          - 'eps': (n_scales,), box sizes (1/bs)
          - 'N_boxes': (n_scales,), occupied-box counts
          - 'x': (n_scales,), log(bs) used in the linear fit
          - 'y': (n_scales,), log(N_boxes) used in the linear fit
          - 'coeffs': (2,), slope/intercept from np.polyfit
    """
    P = np.asarray(points, float)
    P = P[np.all(np.isfinite(P), axis=1)]
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] == 0:
        raise ValueError("points must be (N,3) and non-empty")

    N = P.shape[0]

    # --- Select measurement region and mask ---
    if robust_fraction is not None:
        if not (0.0 < robust_fraction <= 1.0):
            raise ValueError("robust_fraction must be in (0, 1].")
        center = np.median(P, axis=0)
        r_inf = np.max(np.abs(P - center), axis=1)  # Chebyshev radius
        h = float(np.quantile(r_inf, robust_fraction))
        if not np.isfinite(h) or h <= 0.0:
            h = 1.0
        mask = r_inf <= h
        used_fraction = float(np.mean(mask))
        bounds_mins = center - h
        bounds_maxs = center + h
        halfwidth = h
    else:
        mask = np.ones(N, dtype=bool)
        used_fraction = 1.0
        bounds_mins = P.min(axis=0)
        bounds_maxs = P.max(axis=0)
        center = 0.5 * (bounds_mins + bounds_maxs)
        halfwidth = 0.5 * float(np.max(bounds_maxs - bounds_mins))

    span = bounds_maxs - bounds_mins
    span[span == 0] = 1.0  # avoid division by zero

    # Normalize used points to [0,1]^3
    Q = (P[mask] - bounds_mins) / span

    # --- Multi-scale box counts ---
    b0 = max(min_boxes_per_axis, 2)
    bs = np.array([b0 * (2 ** k) for k in range(n_scales)], dtype=int)
    eps = 1.0 / bs.astype(float)

    N_boxes = []
    for b in bs:
        edges = [np.linspace(0.0, 1.0, b + 1)] * 3
        H, _ = np.histogramdd(Q, bins=edges)
        N_boxes.append(np.count_nonzero(H))
    N_boxes = np.array(N_boxes, float)

    # Fit log N vs log b
    x = np.log(bs.astype(float))
    y = np.log(N_boxes + 1e-12)
    coeffs = np.polyfit(x, y, 1)
    Dhat = float(coeffs[0])

    if return_fit:
        info = {
            'mask': mask,
            'center': center,
            'halfwidth': halfwidth,
            'bounds_mins': bounds_mins,
            'bounds_maxs': bounds_maxs,
            'used_fraction': used_fraction,
            'bs': bs,
            'eps': eps,
            'N_boxes': N_boxes,
            'x': x,
            'y': y,
            'coeffs': coeffs,
        }
        return Dhat, info
    return Dhat

def _bounds_with_padding(stars, pad_fraction=0.02, min_box=None):
    ux, uy, uz = stars.x.unit, stars.y.unit, stars.z.unit
    x = stars.x.value_in(ux)
    y = stars.y.value_in(uy)
    z = stars.z.value_in(uz)

    mins = np.array([x.min(), y.min(), z.min()], float)
    maxs = np.array([x.max(), y.max(), z.max()], float)

    span = maxs - mins

    # Apply minimum span if requested
    if min_box is not None:
        for i in range(3):
            if span[i] < min_box:
                mid = 0.5 * (mins[i] + maxs[i])
                mins[i] = mid - 0.5 * min_box
                maxs[i] = mid + 0.5 * min_box
                span[i] = min_box

    # Add padding
    pad = pad_fraction * span
    # If span is zero (after enforcing min_box), add 1.0 of unit as fallback
    pad[span == 0.0] = 1.0

    mins -= pad
    maxs += pad

    return (mins[0], maxs[0], ux), (mins[1], maxs[1], uy), (mins[2], maxs[2], uz)

def _grid_edges(bounds_x, bounds_y, bounds_z, B):
    xmin, xmax, ux = bounds_x; ymin, ymax, uy = bounds_y; zmin, zmax, uz = bounds_z
    ex = np.linspace(xmin, xmax, B + 1)
    ey = np.linspace(ymin, ymax, B + 1)
    ez = np.linspace(zmin, zmax, B + 1)
    return (ex, ux), (ey, uy), (ez, uz)


def _hist3_counts(stars, edges):
    (ex, ux), (ey, uy), (ez, uz) = edges
    x = stars.x.value_in(ux); y = stars.y.value_in(uy); z = stars.z.value_in(uz)
    H, _ = np.histogramdd((x, y, z), bins=(ex, ey, ez))
    return H.astype(int)


def _flat_index_to_ijk(idx, B):
    k = idx % B
    j = (idx // B) % B
    i = idx // (B * B)
    return i, j, k


def _sample_points_in_cells(flat_indices, edges, rng):
    (ex, ux), (ey, uy), (ez, uz) = edges
    B = len(ex) - 1
    n = len(flat_indices)
    xs = np.empty(n); ys = np.empty(n); zs = np.empty(n)
    for t, idx in enumerate(flat_indices):
        i, j, k = _flat_index_to_ijk(int(idx), B)
        xs[t] = rng.uniform(ex[i], ex[i+1])
        ys[t] = rng.uniform(ey[j], ey[j+1])
        zs[t] = rng.uniform(ez[k], ez[k+1])
    return xs, ys, zs, ux, uy, uz


def _nearest_velocity_stats(xyz_exist, vxyz_exist, query_xyz, k, use_mass_weight=False, masses=None, sigma_floor_fraction=0.05):
    N = xyz_exist.shape[0]
    k = int(min(max(1, k), N))
    vcoms = np.empty((len(query_xyz), 3)); sigmas = np.empty((len(query_xyz), 3))

    if use_mass_weight and masses is not None:
        m = masses
        msum = m.sum() if m.size else 1.0
        vglob = (vxyz_exist * m[:, None]).sum(axis=0) / max(msum, 1e-30)
        varglob = (m[:, None] * (vxyz_exist - vglob) ** 2).sum(axis=0) / max(msum, 1e-30)
    else:
        vglob = vxyz_exist.mean(axis=0)
        varglob = vxyz_exist.var(axis=0)
    sig_glob = np.sqrt(np.maximum(varglob, 0.0))

    for t, q in enumerate(query_xyz):
        d2 = np.sum((xyz_exist - q) ** 2, axis=1)
        idx = np.argpartition(d2, k - 1)[:k]
        V = vxyz_exist[idx]
        if use_mass_weight and masses is not None:
            w = masses[idx]
            wsum = w.sum()
            vcom = (V * w[:, None]).sum(axis=0) / max(wsum, 1e-30)
            var = (w[:, None] * (V - vcom) ** 2).sum(axis=0) / max(wsum, 1e-30)
        else:
            vcom = V.mean(axis=0)
            var = V.var(axis=0)
        sig = np.sqrt(np.maximum(var, (sigma_floor_fraction * sig_glob) ** 2))
        vcoms[t] = vcom; sigmas[t] = sig
    return vcoms, sigmas


def generate_fractal_single_level(
    stars,
    D_target,
    n_new,
    boxes_per_axis=16,
    pad_fraction=0.02,
    alpha=None,              # None → uniform over valid; float → occupancy exponent
    neighbor_k=32,
    use_mass_weight=False,
    sigma_floor_fraction=0.05,
    seed=None,
    return_as_particles=True,
):
    """
    Single-level generator using a fixed grid and an optional occupancy-bias exponent.

    Steps:
      1) Build BxBxB grid over padded bounds.
      2) Mark as valid: all currently occupied boxes + enough empty boxes so that
         #valid ≈ B^D (clamped to total boxes). Empty boxes are chosen uniformly.
      3) Assign boxes to new stars:
         - If alpha is None: uniform over valid boxes;
         - Else: weights ∝ (count_in_box + eps)^alpha over *valid* boxes.
         Ensure that each newly chosen empty valid box receives at least one star
         before distributing remaining stars (so it truly becomes occupied).
      4) Sample positions uniformly inside the chosen boxes.
      5) Assign velocities from k-neighbour COM and 1D dispersions.

    Returns: AMUSE Particles (default) or raw arrays.
    """
    if not (0.0 <= D_target <= 3.0):
        raise ValueError("D_target must be in [0, 3]")
    if n_new <= 0:
        raise ValueError("n_new must be > 0")

    rng = np.random.default_rng(seed)

    # Grid and counts
    B = int(max(2, boxes_per_axis))
    bx, by, bz = _bounds_with_padding(stars, pad_fraction)
    edges = _grid_edges(bx, by, bz, B)
    H = _hist3_counts(stars, edges)
    counts = H.ravel()
    total_boxes = counts.size

    # Occupied set O and target valid size
    occupied_mask = counts > 0
    K_occ = int(occupied_mask.sum())
    N_target = int(np.clip(round(B ** D_target), 1, total_boxes))

    valid = occupied_mask.copy()
    empty_indices = np.flatnonzero(~occupied_mask)
    need = max(0, N_target - K_occ)
    if need > 0 and empty_indices.size > 0:
        choose = rng.choice(empty_indices, size=min(need, empty_indices.size), replace=False)
        valid[choose] = True

    # Build the assignment list of box indices for n_new stars
    chosen_boxes = []

    # First, ensure each newly-selected empty valid box gets 1 star
    new_valid_empty = np.flatnonzero(valid & (~occupied_mask))
    n_seed = min(len(new_valid_empty), n_new)
    if n_seed > 0:
        chosen_boxes.extend(list(new_valid_empty[:n_seed]))

    remaining = n_new - len(chosen_boxes)
    if remaining > 0:
        valid_indices = np.flatnonzero(valid)
        if alpha is None:
            w = np.ones_like(valid_indices, float)
        else:
            eps = 1e-6
            w = (counts[valid_indices].astype(float) + eps) ** float(alpha)
        w_sum = w.sum()
        if w_sum <= 0:
            w = np.ones_like(w, float)
            w_sum = w.sum()
        w = w / w_sum
        extra = rng.choice(valid_indices, size=remaining, replace=True, p=w)
        chosen_boxes.extend(list(extra))

    chosen_boxes = np.asarray(chosen_boxes, int)

    # Sample positions inside chosen boxes
    xs, ys, zs, ux, uy, uz = _sample_points_in_cells(chosen_boxes, edges, rng)

    # Neighbour velocities
    x_exist = stars.x.value_in(ux); y_exist = stars.y.value_in(uy); z_exist = stars.z.value_in(uz)
    xyz_exist = np.column_stack([x_exist, y_exist, z_exist])
    uvx, uvy, uvz = stars.vx.unit, stars.vy.unit, stars.vz.unit
    vx_exist = stars.vx.value_in(uvx); vy_exist = stars.vy.value_in(uvy); vz_exist = stars.vz.value_in(uvz)
    vxyz_exist = np.column_stack([vx_exist, vy_exist, vz_exist])
    masses = None
    if use_mass_weight and hasattr(stars, 'mass'):
        masses = np.asarray(stars.mass.value_in(stars.mass.unit))

    vcoms, sigmas = _nearest_velocity_stats(
        xyz_exist, vxyz_exist, np.column_stack([xs, ys, zs]), neighbor_k,
        use_mass_weight=use_mass_weight, masses=masses, sigma_floor_fraction=sigma_floor_fraction,
    )

    vx_new = vcoms[:, 0] + rng.normal(0.0, 1.0, size=len(xs)) * sigmas[:, 0]
    vy_new = vcoms[:, 1] + rng.normal(0.0, 1.0, size=len(xs)) * sigmas[:, 1]
    vz_new = vcoms[:, 2] + rng.normal(0.0, 1.0, size=len(xs)) * sigmas[:, 2]

    if return_as_particles:
        new = Particles(len(xs))
        new.x = xs | ux; new.y = ys | uy; new.z = zs | uz
        new.vx = vx_new | uvx; new.vy = vy_new | uvy; new.vz = vz_new | uvz
        return new
    else:
        return xs, ys, zs, vx_new, vy_new, vz_new, ux, uvx


def generate_fractal_cascade(
    stars,
    D_target,
    n_new,
    m=2,
    L_max=None,
    pad_fraction=0.02,
    neighbor_k=32, # for velocity coherence
    use_mass_weight=False,
    sigma_floor_fraction=0.05,
    seed=None,
    return_as_particles=True,
    min_box = None
):
    if not (0.0 <= D_target <= 3.0):
        raise ValueError("D_target must be in [0, 3]")
    if n_new <= 0:
        raise ValueError("n_new must be > 0")

    rng = np.random.default_rng(seed)
    bx, by, bz = _bounds_with_padding(stars, pad_fraction,min_box=min_box)

    if L_max is None:
        N_final = max(1, int(len(stars) + n_new))
        L_guess = int(np.floor(np.log(max(10 * N_final, 8)) / np.log(max(m ** 3, 8))))
        L_max = int(np.clip(L_guess, 3, 10))

    edges_levels = []
    H_levels = []
    for ell in range(L_max + 1):
        B = m ** ell
        edges = _grid_edges(bx, by, bz, B)
        edges_levels.append(edges)
        H_levels.append(_hist3_counts(stars, edges))

    def ijk_from_flat(idx, B):
        k = idx % B
        j = (idx // B) % B
        i = idx // (B * B)
        return int(i), int(j), int(k)

    def flat_from_ijk(i, j, k, B):
        return int(i) * B * B + int(j) * B + int(k)

    active_masks = {0: np.array([True], dtype=bool)}
    S_levels = {ell: np.array([], dtype=int) for ell in range(L_max + 1)}

    stars_remaining = int(n_new)

    for ell in range(0, L_max + 1):
        H = H_levels[ell]
        B = H.shape[0]
        total_cells = H.size

        if ell == 0:
            A = active_masks[0]
        else:
            A_prev = active_masks.get(ell - 1, None)
            if A_prev is None or not np.any(A_prev):
                A = np.zeros(total_cells, dtype=bool)
            else:
                H_prev = H_levels[ell - 1].ravel()
                occupied_prev = H_prev > 0
                kept_parents = A_prev & (occupied_prev | np.isin(np.arange(H_prev.size), S_levels[ell - 1]))
                A = np.zeros(total_cells, dtype=bool)
                B_prev = H_levels[ell - 1].shape[0]
                for p in np.flatnonzero(kept_parents):
                    ip, jp, kp = ijk_from_flat(int(p), B_prev)
                    for di in range(m):
                        for dj in range(m):
                            for dk in range(m):
                                ic = ip * m + di
                                jc = jp * m + dj
                                kc = kp * m + dk
                                A[flat_from_ijk(ic, jc, kc, B)] = True
        active_masks[ell] = A
        if not np.any(A):
            continue

        H_flat = H.ravel()
        occupied = (H_flat > 0) & A
        empty = (~(H_flat > 0)) & A
        K_occ = int(occupied.sum())

        T_ell = int(np.clip(np.ceil(m ** (ell * D_target)), 1, int(A.sum())))
        need = max(0, T_ell - K_occ)

        if need > 0 and stars_remaining > 0 and np.any(empty):
            take = min(need, stars_remaining, int(empty.sum()))
            empties = np.flatnonzero(empty)
            choose = rng.choice(empties, size=take, replace=False)
            S_levels[ell] = choose.astype(int)
            stars_remaining -= take
        if stars_remaining <= 0:
            break

    placement_cells = [(ell, int(idx)) for ell in range(L_max + 1) for idx in S_levels[ell]]

    if stars_remaining > 0:
        A_L = active_masks.get(L_max, np.zeros(H_levels[L_max].size, dtype=bool))
        H_L_flat = H_levels[L_max].ravel()
        occupied_L = (H_L_flat > 0) & A_L
        candidates = np.flatnonzero(occupied_L)
        if candidates.size == 0:
            candidates = np.flatnonzero(A_L)
        extra = list(rng.choice(candidates, size=stars_remaining, replace=True))
        placement_cells.extend([(L_max, int(i)) for i in extra])

    xs_all, ys_all, zs_all = [], [], []
    ux = uy = uz = None
    for (ell, idx) in placement_cells:
        edges = edges_levels[ell]
        xs, ys, zs, ux, uy, uz = _sample_points_in_cells([idx], edges, rng)
        xs_all.append(xs[0]); ys_all.append(ys[0]); zs_all.append(zs[0])

    xs = np.asarray(xs_all); ys = np.asarray(ys_all); zs = np.asarray(zs_all)

    x_exist = stars.x.value_in(ux); y_exist = stars.y.value_in(uy); z_exist = stars.z.value_in(uz)
    xyz_exist = np.column_stack([x_exist, y_exist, z_exist])
    uvx, uvy, uvz = stars.vx.unit, stars.vy.unit, stars.vz.unit
    vx_exist = stars.vx.value_in(uvx); vy_exist = stars.vy.value_in(uvy); vz_exist = stars.vz.value_in(uvz)
    vxyz_exist = np.column_stack([vx_exist, vy_exist, vz_exist])
    masses = None
    if use_mass_weight and hasattr(stars, 'mass'):
        masses = np.asarray(stars.mass.value_in(stars.mass.unit))

    vcoms, sigmas = _nearest_velocity_stats(
        xyz_exist, vxyz_exist, np.column_stack([xs, ys, zs]), neighbor_k,
        use_mass_weight=use_mass_weight, masses=masses, sigma_floor_fraction=sigma_floor_fraction,
    )

    rngn = np.random.default_rng(seed + 1 if seed is not None else None)
    vx_new = vcoms[:, 0] + rngn.normal(0.0, 1.0, size=len(xs)) * sigmas[:, 0]
    vy_new = vcoms[:, 1] + rngn.normal(0.0, 1.0, size=len(xs)) * sigmas[:, 1]
    vz_new = vcoms[:, 2] + rngn.normal(0.0, 1.0, size=len(xs)) * sigmas[:, 2]

    if return_as_particles:
        new = Particles(len(xs))
        new.x = xs | ux; new.y = ys | uy; new.z = zs | uz
        new.vx = vx_new | uvx; new.vy = vy_new | uvy; new.vz = vz_new | uvz
        return new
    else:
        return xs, ys, zs, vx_new, vy_new, vz_new, ux, uvx

if __name__ == '__main__':
    pass

