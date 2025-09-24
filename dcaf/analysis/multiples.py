import os
import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree
from amuse.units import units, constants
from amuse.datamodel import Particles
from collections import Counter
from amuse.io import read_set_from_file


class SystemTree:
    def __init__(self):
        self.parents = None
        self.level   = None
        self.a       = []
        self.e       = []
        self.Ebind   = []
        self.P       = []
        self.q       = []
        self.ain_over_aout = None      # shape (N, 2), float
        self.ain_child_idx = None      # shape (N, 2), int (which child the ratio came from)
        self.__root_ids = None

    @property
    def root_ids(self):
        if self.__root_ids is None:
            self.__root_ids = self.get_root_ids()
        return self.__root_ids

    def make_leaves(self, n):
        self.parents = np.full((n, 2), -1, dtype=int)
        self.level   = np.zeros(n, dtype=int)
        self.a       = ([0 ] * n ) | units.au
        self.e       = np.array([0.0] * n)
        self.Ebind   = ([0 ] * n) | (units.MSun * (units.kms**2))
        self.P       = ([0] * n ) | units.Myr
        self.q       = np.array([0.0] * n)
        self.ain_over_aout = np.full((n, 2), np.nan, dtype=float)
        self.ain_child_idx = np.full((n, 2), -1, dtype=int)

    def append_binary(self, c1, c2, semi, e, Ebind, period,q):
        self.parents = np.vstack([self.parents, np.array([[c1, c2]], dtype=int)])
        self.level   = np.append(self.level, max(self.level[c1], self.level[c2]) + 1)
        self.a.extend(semi.as_vector_with_length( 1 ));
        self.e = np.concatenate( [self.e,[e]] )
        self.Ebind.extend(Ebind.as_vector_with_length(1))
        self.P.extend(period.as_vector_with_length(1))
        self.q = np.concatenate( [self.q,[q]] )
        self.ain_over_aout = np.vstack([self.ain_over_aout, np.array([[np.nan, np.nan]], dtype=float)])
        self.ain_child_idx = np.vstack([self.ain_child_idx, np.array([[-1, -1]], dtype=int)])

        new_id = len(self.level) - 1  # parent node id
        a_out = semi  # AMUSE quantity (units.au)

        # For each child that is itself a binary (level >= 1), store a_in/a_out
        fillcol = 0
        for child in (c1, c2):
            if self.level[child] >= 1:
                a_in = self.a[child]  # AMUSE quantity
                # guard against zero/NaN
                if a_out.value_in(units.au) > 0 and np.isfinite(a_in.value_in(units.au)):
                    self.ain_over_aout[new_id, fillcol] = (a_in / a_out)
                    self.ain_child_idx[new_id, fillcol] = child
                    fillcol += 1
                    if fillcol == 2:
                        break

        return new_id

    def resolve(self, node_id):
        stack = [node_id]; leaves = []
        while stack:
            nid = stack.pop()
            c1, c2 = self.parents[nid]
            if c1 == -1:
                leaves.append(nid)
            else:
                stack.extend([c1, c2])
        return leaves

    def resolve_set(self, node_ids):
        out = set()
        for nid in node_ids:
            out.update(self.resolve(nid))
        return sorted(out)

    def get_root_ids(self):
        """Return node IDs that are never listed as children (i.e. roots)."""
        n = self.parents.shape[0]
        children = set(self.parents[self.parents[:,0] != -1].ravel())
        children.discard(-1)
        return [i for i in range(n) if i not in children]

    def multiplicity_counter(self,max_n=3):
        """
        Split each root system into largest disjoint components of size ≤ max_n
        (using existing nodes), then count by size.
        Returns [N_max_n, ..., N_2, N_1].
        """
        root_ids = self.root_ids
        # leaf counts per node (memoized DFS)
        n_nodes = self.parents.shape[0]
        leaf_cnt = np.full(n_nodes, -1, dtype=int)
        def leaves(i):
            if leaf_cnt[i] >= 0: return leaf_cnt[i]
            c1, c2 = self.parents[i]
            if c1 == -1:
                leaf_cnt[i] = 1
            else:
                leaf_cnt[i] = leaves(c1) + leaves(c2)
            return leaf_cnt[i]
        for i in range(n_nodes):
            if leaf_cnt[i] < 0: leaves(i)

        # pack subself into components ≤ max_n
        def pack(i, out_sizes):
            c1, c2 = self.parents[i]
            if c1 == -1:
                out_sizes.append(1); return
            if leaf_cnt[i] <= max_n:
                out_sizes.append(leaf_cnt[i]); return
            pack(c1, out_sizes); pack(c2, out_sizes)

        sizes = []
        for rid in root_ids:
            pack(rid, sizes)

        c = Counter(sizes)
        return [c.get(k, 0) for k in range(max_n, 0, -1)]


def get_orbital_parameters(r_rel, v_rel, m1, m2):
    G = constants.G

    mu = m1 + m2
    r  = r_rel.lengths()
    v2 = v_rel.lengths()**2

    # specific orbital energy
    E_spec = 0.5 * v2 - G * mu / r

    # specific angular momentum
    h = r_rel.cross(v_rel).lengths()

    # semi-major axis
    a = - G * mu / (2.0 * E_spec)

    # eccentricity
    e_sq = 1.0 - (h**2) / (a * G * mu)
    e_sq = np.clip(e_sq, 0.0, 1e6)
    e = np.sqrt(e_sq)

    # binding energy 
    red_mass = (m1 * m2) / mu
    Ebind = E_spec * red_mass

    # period
    P = 2.0 * np.pi * ((a**3) / (G * mu))**0.5

    m1_num = m1.value_in(units.MSun)
    m2_num = m2.value_in(units.MSun)
    denom  = np.maximum(m1_num, m2_num)
    # avoid /0 if any mass is zero
    q = np.where(denom > 0, np.minimum(m1_num, m2_num) / denom, np.nan)

    return a, e, Ebind, P, q


def _merge_pairs_build_next_particles(live, pairs_ij):
    """
    Build next Particles:
      - merge each (i,j) into one CoM particle 
    """
    N = len(live)
    taken = np.zeros(N, dtype=bool)
    taken[pairs_ij.ravel()] = True

    # singles in one go
    singles_mask = ~taken
    singles_idx = np.where(singles_mask)[0]
    next_parts = live[singles_mask].copy()

    # vectorized CoMs for all pairs
    if len(pairs_ij) > 0:
        i = pairs_ij[:, 0]
        j = pairs_ij[:, 1]

        m1 = live[i].mass
        m2 = live[j].mass
        mt = m1 + m2

        r1 = live[i].position    # (M,3)
        r2 = live[j].position
        v1 = live[i].velocity    # (M,3)
        v2 = live[j].velocity

        rc = (m1[:, None] * r1 + m2[:, None] * r2) / mt[:, None]   # (M,3)
        vc = (m1[:, None] * v1 + m2[:, None] * v2) / mt[:, None]   # (M,3)

        com = Particles(len(pairs_ij))
        com.mass = mt
        com.position = rc
        com.velocity = vc

        next_parts.add_particles(com)

    return next_parts, singles_idx

def find_hierarchical_binaries(particles, k=20, a_min_ratio = 2):
    """
    Iterative rounds:
      - KD-tree on positions (pc)
      - vectorized binding over neighbour pairs (using AMUSE quantities)
      - greedy disjoint matching (desc Ebind)
      - merge all accepted pairs (2→1)

    Returns: live_particles, system_tree, live_to_global (live idx → global node id)
    """
    N0 = len(particles)
    tree = SystemTree()
    tree.make_leaves(N0)
    live = particles.copy()
    live_to_global = np.arange(N0, dtype=int)

    while len(live) > 1:
        N = len(live)
        if N <= 1:
            break

        # KD-tree (needs floats)
        pos_pc  = live.position.value_in(units.pc)     # (N,3) floats
        vel_kms = live.velocity.value_in(units.kms)    # (N,3) floats
        m_msun  = live.mass.value_in(units.MSun)       # (N,)  floats

        kd = cKDTree(pos_pc)
        _, neighs = kd.query(pos_pc, k=min(k+1, N))    # (N, k+1) (includes self)

        # candidate pairs with i<j
        I = np.repeat(np.arange(N), neighs.shape[1]-1)
        J = neighs[:, 1:].ravel()
        mask = I < J
        I = I[mask]; J = J[mask]
        if I.size == 0:
            break

        # rebuild units
        r_rel = (pos_pc[I] - pos_pc[J]) | units.pc          # (M,3)
        v_rel = (vel_kms[I] - vel_kms[J]) | units.kms       # (M,3)
        m1    = (m_msun[I]) | units.MSun                    # (M,)
        m2    = (m_msun[J]) | units.MSun                    # (M,)

        # vectorized orbital params
        a_q, e_v, Eb_q, P_q, q_q = get_orbital_parameters(r_rel, v_rel, m1, m2)

        # keep bound
        bound = (Eb_q < 0 | (units.MSun * (units.kms**2)))
        if not np.any(bound):
            break

        I = I[bound]; J = J[bound]
        a_q = a_q[bound]; e_v = e_v[bound]; Eb_q = Eb_q[bound]; P_q = P_q[bound]
        q_q = q_q[bound]

        # Lets filter cases where ain_over_aout are too small, i.e. likely not
        # stable.
        # Map live indices -> global node ids
        gi = live_to_global[I]
        gj = live_to_global[J]

        # Is each child already a binary (or deeper)?
        is_bin_i = (tree.level[gi] >= 1)
        is_bin_j = (tree.level[gj] >= 1)

        # Outer semi-major axis (parent candidate)
        a_out = a_q.value_in(units.au)  # (M,)

        # Default: leaves auto-pass (no inner orbit to compare to)
        ok_i = np.ones(I.size, dtype=bool)
        ok_j = np.ones(I.size, dtype=bool)

        # For children that ARE binaries, enforce a_out / a_in >= a_min_ratio
        if np.any(is_bin_i):
            a_in_i = np.array([tree.a[g].value_in(units.au) for g in gi[is_bin_i]], dtype=float)
            ok_i[is_bin_i] = (a_out[is_bin_i] / a_in_i) >= float(a_min_ratio)

        if np.any(is_bin_j):
            a_in_j = np.array([tree.a[g].value_in(units.au) for g in gj[is_bin_j]], dtype=float)
            ok_j[is_bin_j] = (a_out[is_bin_j] / a_in_j) >= float(a_min_ratio)

        ok = ok_i & ok_j

        # Keep only hierarchical-enough candidates
        I = I[ok]; J = J[ok]
        a_q = a_q[ok]; e_v = e_v[ok]; Eb_q = Eb_q[ok]
        P_q = P_q[ok]; q_q = q_q[ok]   # <-- fix: do NOT write "P_q = P_q[ok], q_q[ok]"
        if I.size == 0:
            break

        # greedy disjoint matching (asc Ebind)
        order = np.argsort(Eb_q.value_in(units.MSun * (units.kms**2)))#%[::-1]
        taken = np.zeros(N, dtype=bool)
        sel_pairs = []
        keep_idx = []
        for t in order:
            i, j = I[t], J[t]
            if not taken[i] and not taken[j]:
                taken[i] = taken[j] = True
                sel_pairs.append((i, j))
                keep_idx.append(t)

        if len(sel_pairs) == 0:
            break
        keep_idx = np.array(keep_idx, dtype=int)

        # register new parents in tree
        new_parent_ids = []
        for kk, (i, j) in enumerate(sel_pairs):
            gi, gj = live_to_global[i], live_to_global[j]
            pid = tree.append_binary(gi, gj, a_q[keep_idx][kk], e_v[keep_idx][kk],
                                     Eb_q[keep_idx][kk], P_q[keep_idx][kk],
                                     q_q[keep_idx][kk])
            new_parent_ids.append(pid)
        new_parent_ids = np.asarray(new_parent_ids, dtype=int)

        # build next generation of Particles (vectorized)
        next_live, singles_idx = _merge_pairs_build_next_particles(live, np.array(sel_pairs, dtype=int))

        # update mapping
        next_map = np.empty(len(next_live), dtype=int)
        next_map[:singles_idx.size] = live_to_global[singles_idx]
        next_map[singles_idx.size:] = new_parent_ids

        live = next_live
        live_to_global = next_map

    return live, tree, live_to_global

def load_system_tree(path):
    """Load a saved multiples tree (<basename>_mult.npz) into a SystemTree."""
    data = np.load(path, allow_pickle=True)
    tree = SystemTree()
    tree.parents = data["parents"].astype(int)
    tree.level   = data["level"].astype(int)
    tree.a       = data["semi"] | units.au
    tree.e       = data["e"].astype(float)
    tree.Ebind   = data["Ebind"] | units.Msun * units.kms**2
    tree.P       = data["period"] | units.yr
    tree.q       = data["q"].astype(float)
    tree.ain_over_aout = data['ain_over_aout'].astype(float)
    tree.ain_child_idx = data["ain_child_idx"].astype(int)
    return tree

class MultiplesAnalyzer:
    def __init__(self, k=20, out_dir="multiples", results_name="multiples.csv"):
        self.k = k
        self.out_dir = out_dir
        self.results_name = results_name

    def process_files(self, file_list):
        """
        For each snapshot:
          - load stars
          - find hierarchy
          - save tree to <out_dir>/<basename>_mult.npz
          - append row to <out_dir>/multiples.dat
        """
        os.makedirs(self.out_dir, exist_ok=True)
        results_path = os.path.join(self.out_dir, self.results_name)
        self._ensure_results_header(results_path)

        for sid, fn in enumerate(file_list):
            print(f'[{sid}/{len(file_list)}] Calculating hierarchy in {fn} ',
                  end ='\r'
                )
            stars = read_set_from_file(fn)  # AMUSE autodetects format
            ca = getattr(stars, "collection_attributes", None)
            mt = getattr(ca, "model_time", None) if ca is not None else None
            model_time_myr = (mt if mt is not None else (0 | units.Myr)).as_quantity_in(units.Myr).value_in(units.Myr)

            live, tree, live_to_global = find_hierarchical_binaries(stars, k=self.k)
            root_ids = np.asarray(live_to_global, dtype=int)

            base = os.path.splitext(os.path.basename(fn))[0]
            tree_path = os.path.join(self.out_dir, f"{base}_mult.npz")
            self._save_tree(tree_path, tree)

            counts = self._multiplicity_counts(tree, root_ids)
            self._append_results_row(
                results_path, sid, model_time_myr, counts
            )
        print('\n')

    def _ensure_results_header(self, path):
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("snapshot_id,model_time_Myr,singles,binaries,triples,quadruples,quintuples\n")

    def _append_results_row(self, path, sid, model_time_myr, counts):
        with open(path, "a") as f:
            f.write(
                f"{sid},{model_time_myr:.6f},"
                f"{counts['singles']},{counts['binaries']},"
                f"{counts['triples']},{counts['quadruples']},{counts['quintuples']}\n"
            )

    def _save_tree(self, path, tree):
        # store arrays in : pc, km/s, Msun, Myr
        np.savez(
            path,
            parents=tree.parents.astype(np.int64),
            level=tree.level.astype(np.int32),
            semi= tree.a.value_in(units.au),
            e = np.asarray(tree.e, dtype=float),
            Ebind = tree.Ebind.value_in( units.Msun * units.kms**2),
            period =tree.P.value_in(units.yr),
            q = np.asarray(tree.q, dtype=float),
            ain_over_aout = np.asarray(tree.ain_over_aout, dtype=float),
            ain_child_idx = np.asarray(tree.ain_child_idx, dtype=int)
        )

    def _multiplicity_counts(self, tree, root_ids):
        # system size = number of leaves under each root
        sizes = [len(tree.resolve(rid)) for rid in root_ids]
        c = Counter(sizes)
        return dict(
            singles=c.get(1, 0),
            binaries=c.get(2, 0),
            triples=c.get(3, 0),
            quadruples=c.get(4, 0),
            quintuples=c.get(5, 0),
        )
