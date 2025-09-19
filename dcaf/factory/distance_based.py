#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D-CAF factory: minimal distance-based star generator (with fast CDF sampler)

Keep it **clear and tiny**:
- Default distance PDF: lognormal (mu=-2.15, sigma=0.9 in ln r)
  for ultra-fast, reusable sampling.
- Uniform random reference star (no density weighting)
- Place the new star at that distance in a random 3D direction (no box logic)
- Set velocity from nearest neighbors: per-axis MAD dispersion, median removed
- Enforce a **minimum separation** using a KD-tree (no close binaries)

Inputs: existing AMUSE Particles, number of new stars
Output: new AMUSE Particles (positions & velocities set; masses untouched)

Assumes AMUSE and SciPy are installed.
"""

import numpy as np
from scipy.spatial import cKDTree
from amuse.datamodel import Particles
from amuse.units import units
from dcaf.utilities.sampler import PDFSampler, lognormal_pdf
from dcaf.utilities.helpers import ( robust_stats, sample_sphere_surface,
                                    weights_by_density)

def generate_stars(existing,
                   n_new,
                   box_size = None,
                   pdf_func=lognormal_pdf,
                   pdf_unit = units.parsec,
                   min_separation=0.01 | units.parsec,
                   max_separation = 100 |units.parsec,
                   neighbor_k_vel=20,
                   beta = 1, #for density weights.
                   framework = None,
                   seed=42):
    """
    Generate stars based on existing set sampling distances to old stars from a
    pdf_func.

    pdf_func means to represent the star formation simulation we want to mimic.
    It could also draw from a tabulated pdf (TODO)

    Method: 
    1) Convert AMUSE Particles to plain NumPy arrays (positions/velocities)
    2) Build a PDFSampler once over [r_min, r_max] for the requested PDF
    3) Keep an evolving catalogue (cat_pos, cat_vel) and KD-tree for min-sep
    4) For each new star: sample distance r, pick a uniform reference, place in
       a random direction, enforce min-sep, 
    5) If no framework was given, set velocity from velocity dispersion of
    neighbors. If framework is given, will call framework.method,
    get_velocity_dispersion_at_point 

    6) Return the new AMUSE Particles
    """
    if len(existing) == 0:
        raise ValueError("existing Particles is empty")

    if n_new == 0 :
        return Particles()

    # construct arrays without units for now
    if pdf_unit is None:
        L = existing.x.unit
    else:
        L = pdf_unit
    S = existing.vx.unit
    #Original positions
    pos0 = np.column_stack([
        existing.x.value_in(L),
        existing.y.value_in(L),
        existing.z.value_in(L),
    ])
    vel0 = np.column_stack([
        existing.vx.value_in(S),
        existing.vy.value_in(S),
        existing.vz.value_in(S),
    ])
    # obtain PDF domain
    r_min = float(min_separation.value_in(L))

    if box_size is not None:
        r_max = np.sqrt(3)*box_size.value_in(L)
    else:
        r_max = float(max_separation.value_in(L))

    if not np.isfinite(r_max) or r_max <= r_min:
        raise ValueError('generate stars: rmin (%s) and rmax (%s) are not valid'%(r_min,r_max) )

    if pdf_func is None:
        pdf_func = lognormal_pdf

    sampler = PDFSampler(pdf_func, (r_min, r_max) , nsample=4096)

    # Keep existing stars updated
    gen = np.random.default_rng(seed)
    newP, newV = [], []
    cat_pos = pos0.copy()
    cat_vel = vel0.copy()
    tree = cKDTree(cat_pos)
    weights = weights_by_density(cat_pos,tree=tree,beta = beta)

    # Adding the stars
    for _ in range(n_new):
        placed = False
        while not placed :
            rtarget = float(sampler.sample(1, rng=gen)[0])
            #is the following line needed? just in case
            rtarget = max(rtarget, r_min)
            # remake the pool for each star. We allow new stars formed around the
            # same reference star.
            ref_pool = list(range(len(pos0)))
            while len(ref_pool) > 0 : 
                # --- choose reference particle among ORIGINAL stars ---
                ### weight by density modulated by beta
                p = weights[ref_pool]
                p = p / p.sum()
                ref = gen.choice(ref_pool,p=p )
                center = pos0[ref]

                # Find nearest neighbours on the surface of radius rtarget around the
                # reference star
                K = 256  
                C = sample_sphere_surface(center=center, r=rtarget, n=K, rng=gen) 
                dmin, _ = tree.query(C, k=1)

                # choose a rtarget within tolerance, no need is the same
                # reference star
                # defined by the minimum allowed separation
                valid = (dmin > r_min) & ( np.abs(dmin - rtarget) <= 0.5*r_min )
                if np.any(valid):

                    # pick one valid candidate at random
                    pick = gen.choice(np.flatnonzero(valid))
                    cand = C[pick]

                    # velocity from neighbors 
                    kk  = min(max(2, neighbor_k_vel), len(cat_pos))
                    nbr = tree.query(cand[None, :], k=kk)[1]
                    nbr = np.atleast_1d(nbr).reshape(-1)
                    v_med, v_sig = robust_stats(cat_vel[nbr])

                    if framework:
                        x,y,z = cand | L
                        v_sig = framework.get_velocity_dispersion_at_point(x,y,z)
                        v_sig = v_sig.value_in(S)

                    ## keep the total velocity consistent with v_sig and moving
                    # along its neighbours
                    kk_eff = max(kk,10) # avoid just copying the neighbour
                                        #velocity at small kk 
                    resid = (1.0 - 1.0/float(kk_eff)) ** 0.5
                    eps = gen.normal(0.0, 1.0, 3)
                    v =  v_med + resid * (eps * v_sig)

                    #v = gen.normal(0.0, 1.0, 3) * v_sig + v_med

                    # accept and update catalogue + tree
                    newP.append(cand); newV.append(v)
                    cat_pos = np.vstack([cat_pos, cand[None, :]])
                    cat_vel = np.vstack([cat_vel, v[None, :]])
                    tree = cKDTree(cat_pos)
                    placed = True
                    break
                else:
                    ref_pool.remove(ref)
                    continue

    new_parts = Particles(len(newP))
    if len(newP) > 0:
        newP = np.asarray(newP, float); newV = np.asarray(newV, float)
        new_parts.x = newP[:, 0] | L; new_parts.y = newP[:, 1] | L; new_parts.z = newP[:, 2] | L
        new_parts.vx = newV[:, 0] | S; new_parts.vy = newV[:, 1] | S; new_parts.vz = newV[:, 2] | S
    return new_parts

if __name__ == "__main__":
    pass

