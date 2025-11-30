"""\
This module converts SDP problem instances in SDPA-sparse format to PyG Data
objects for oracle rank schedule prediction in low-rank SDP solvers.

Example usage::
    >>> from dataset.processor import process_sdpa_to_pyg
    >>> data = process_sdpa_to_pyg("problem.dat-s", "output.pt", verbose=True)
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

NUM_GLOBAL_FEATURES = 17
NUM_NODE_FEATURES = 16
NUM_EDGE_FEATURES = 5


class SDPAParser:
    """Streaming parser for SDPA-sparse format files.

    SDPA-sparse format specifies:
        - m: number of constraints
        - nblocks: number of matrix blocks
        - block dimensions (negative = diagonal/LP block)
        - RHS vector b of length m
        - Matrix entries as (matno, block, row, col, value)

    Attributes:
        filepath: Path to the input .dat-s file.
        m: Number of constraints.
        n: Total matrix dimension (sum of block dimensions).
        nblocks: Number of SDP blocks.
        block_dims: List of block dimensions.
        block_offsets: Cumulative offsets for block indexing.
        b: Right-hand side vector.
        C: Cost matrix (sparse CSR).
        A: List of constraint matrices (sparse CSR).
    """

    def __init__(self, filepath: str) -> None:
        """Initializes parser with file path."""
        self.filepath = filepath
        self.m = 0
        self.n = 0
        self.nblocks = 0
        self.block_dims: List[int] = []
        self.block_offsets: List[int] = []
        self.b: Optional[np.ndarray] = None
        self.C: Optional[sp.csr_matrix] = None
        self.A: List[sp.csr_matrix] = []

    def parse(self, verbose: bool = False) -> None:
        """Parses SDPA file using streaming."""
        with open(self.filepath, "r") as f:
            line = self._read_noncomment_line(f)
            if line is None:
                raise ValueError(f"empty or invalid SDPA file: {self.filepath}")

            self.m = int(line.strip())
            self.nblocks = int(self._read_noncomment_line(f).strip())

            block_line = self._read_noncomment_line(f).strip()
            block_tokens = self._parse_block_dims(block_line)

            self.block_dims = [d for d in block_tokens if d > 0]
            self.nblocks = len(self.block_dims)

            self.block_offsets = [0]
            for dim in self.block_dims:
                self.block_offsets.append(self.block_offsets[-1] + dim)
            self.n = self.block_offsets[-1]

            rhs_line = self._read_noncomment_line(f).strip()
            rhs_tokens = (
                rhs_line.replace(",", " ").replace("{", " ").replace("}", " ").split()
            )
            self.b = np.array([float(x) for x in rhs_tokens], dtype=np.float64)

            if len(self.b) != self.m:
                raise ValueError(
                    f"RHS dimension mismatch: got {len(self.b)}, expected {self.m}"
                )

            C_rows, C_cols, C_data = [], [], []
            A_coo = [{"rows": [], "cols": [], "data": []} for _ in range(self.m)]

            entry_count = 0
            for line in f:
                line = line.strip()
                if (
                    not line
                    or line.startswith("*")
                    or line.startswith('"')
                    or "COMMENT" in line.upper()
                ):
                    continue

                parts = line.split()
                if len(parts) != 5:
                    continue

                try:
                    matno = int(parts[0])
                    blk = int(parts[1]) - 1
                    row = int(parts[2]) - 1
                    col = int(parts[3]) - 1
                    value = float(parts[4])
                except ValueError:
                    continue

                if blk < 0 or blk >= self.nblocks:
                    continue

                gr = self.block_offsets[blk] + row
                gc = self.block_offsets[blk] + col

                if matno == 0:
                    C_rows.append(gr)
                    C_cols.append(gc)
                    C_data.append(value)
                    if gr != gc:
                        C_rows.append(gc)
                        C_cols.append(gr)
                        C_data.append(value)
                else:
                    mat_idx = matno - 1
                    if mat_idx < self.m:
                        A_coo[mat_idx]["rows"].append(gr)
                        A_coo[mat_idx]["cols"].append(gc)
                        A_coo[mat_idx]["data"].append(value)
                        if gr != gc:
                            A_coo[mat_idx]["rows"].append(gc)
                            A_coo[mat_idx]["cols"].append(gr)
                            A_coo[mat_idx]["data"].append(value)

                entry_count += 1
                if verbose and entry_count % 500000 == 0:
                    print(f"    [parsed] {entry_count:,} entries...")

        if verbose:
            print("    converting to CSR format...")

        if C_data:
            self.C = sp.csr_matrix(
                (C_data, (C_rows, C_cols)), shape=(self.n, self.n), dtype=np.float64
            )
        else:
            self.C = sp.csr_matrix((self.n, self.n), dtype=np.float64)

        self.A = []
        for i in range(self.m):
            if A_coo[i]["data"]:
                mat = sp.csr_matrix(
                    (A_coo[i]["data"], (A_coo[i]["rows"], A_coo[i]["cols"])),
                    shape=(self.n, self.n),
                    dtype=np.float64,
                )
            else:
                mat = sp.csr_matrix((self.n, self.n), dtype=np.float64)
            self.A.append(mat)

        if verbose:
            print(f"    done. m={self.m}, n={self.n}")

    def _read_noncomment_line(self, f) -> Optional[str]:
        """Reads the next non-comment, non-empty line."""
        while True:
            line = f.readline()
            if not line:
                return None
            line = line.strip()
            if line and not line.startswith("*") and not line.startswith('"'):
                return line

    def _parse_block_dims(self, line: str) -> List[int]:
        """Parses block dimensions from various SDPA formats."""
        cleaned = line.replace("(", " ").replace(")", " ")
        cleaned = cleaned.replace(",", " ").replace("{", " ").replace("}", " ")
        tokens = cleaned.split()
        return [int(x) for x in tokens if x.lstrip("-").isdigit()]

    def get_data(
        self,
    ) -> Tuple[sp.csr_matrix, List[sp.csr_matrix], np.ndarray, int, int, List[int]]:
        """Returns parsed problem data including block structure.

        Returns:
            Tuple of (C, A, b, m, n, block_offsets) where block_offsets
            contains cumulative indices for each block.
        """
        return self.C, self.A, self.b, self.m, self.n, self.block_offsets


class FeatureExtractor:
    """Extracts features from SDP problems for rank schedule task.

    Feature categories:
        - global: size/scale, density, coupling summary, cost-alignment summary
        - node: scale/density, spectral proxies, cost alignment, distributional
        - edge: structural overlap, numerical coupling

    Attributes:
        C: Cost matrix.
        A: List of constraint matrices.
        b: RHS vector.
        m: Number of constraints.
        n: Matrix dimension.
        block_offsets: Cumulative block offsets for multi-block SDPs.
    """

    EPS = 1e-8

    def __init__(
        self,
        C: sp.csr_matrix,
        A: List[sp.csr_matrix],
        b: np.ndarray,
        m: int,
        n: int,
        block_offsets: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> None:
        """Initializes feature extractor with problem data."""
        self.C = C
        self.A = A
        self.b = b
        self.m = m
        self.n = n
        self.block_offsets = block_offsets if block_offsets else [0, n]
        self.nblocks = len(self.block_offsets) - 1
        self.verbose = verbose

        self._precompute_constraint_stats()
        self._precompute_cost_stats()

    def _precompute_constraint_stats(self) -> None:
        """Precomputes per-constraint statistics."""
        self.norms = np.zeros(self.m, dtype=np.float64)
        self.nnz_counts = np.zeros(self.m, dtype=np.int64)
        self.traces = np.zeros(self.m, dtype=np.float64)
        self.diag_norms = np.zeros(self.m, dtype=np.float64)
        self.gershgorin_bounds = np.zeros(self.m, dtype=np.float64)
        self.blocks_touched = np.zeros(self.m, dtype=np.int64)

        self.row_indices: List[np.ndarray] = []
        self.row_sizes = np.zeros(self.m, dtype=np.int64)

        block_starts = np.array(self.block_offsets[:-1])
        block_ends = np.array(self.block_offsets[1:])

        for i, A_i in enumerate(self.A):
            self.nnz_counts[i] = A_i.nnz
            if A_i.nnz > 0:

                self.norms[i] = np.sqrt(np.sum(A_i.data**2))

                diag = A_i.diagonal()
                self.traces[i] = diag.sum()
                self.diag_norms[i] = np.linalg.norm(diag)

                row_sums = np.abs(A_i).sum(axis=1).A1

                self.gershgorin_bounds[i] = row_sums.max() if len(row_sums) > 0 else 0.0

                coo = A_i.tocoo()
                unique_rows = np.unique(coo.row)
                self.row_indices.append(unique_rows)
                self.row_sizes[i] = len(unique_rows)

                if len(unique_rows) > 0 and self.nblocks > 1:
                    min_row, max_row = unique_rows.min(), unique_rows.max()

                    blocks_with_rows = np.sum(
                        (block_starts <= max_row) & (block_ends > min_row)
                    )
                    self.blocks_touched[i] = blocks_with_rows
                else:
                    self.blocks_touched[i] = 1 if len(unique_rows) > 0 else 0
            else:
                self.row_indices.append(np.array([], dtype=np.int64))

            if self.verbose and (i + 1) % 2000 == 0:
                print(f"    [precomputed stats] {i+1}/{self.m}")

        self.log_norms = np.log(1.0 + self.norms)
        self.log_nnz = np.log(1.0 + self.nnz_counts)
        self.normed_rhs = np.clip(self.b / (self.norms + self.EPS), -100.0, 100.0)

        self.mu_log_fro = self.log_norms.mean() if self.m > 0 else 0.0
        self.sigma_log_fro = (self.log_norms.std() if self.m > 0 else 0.0) + self.EPS
        self.mu_log_nnz = self.log_nnz.mean() if self.m > 0 else 0.0
        self.sigma_log_nnz = (self.log_nnz.std() if self.m > 0 else 0.0) + self.EPS
        self.mu_normed_rhs = np.abs(self.normed_rhs).mean() if self.m > 0 else 0.0
        self.sigma_normed_rhs = (
            np.abs(self.normed_rhs).std() if self.m > 0 else 0.0
        ) + self.EPS

        if self.m > 0:
            self.fro_quantiles = np.percentile(self.log_norms, [25, 50, 75])
        else:
            self.fro_quantiles = np.array([0.0, 0.0, 0.0])

        self._build_pattern_matrix()

    def _build_pattern_matrix(self) -> None:
        """Builds sparse pattern matrix for overlap computation."""
        all_rows = []
        all_cols = []

        for i in range(self.m):
            row_idx = self.row_indices[i]
            if len(row_idx) > 0:
                all_rows.extend([i] * len(row_idx))
                all_cols.extend(row_idx)

        if all_rows:
            max_col = max(all_cols) + 1
            self.pattern = sp.csr_matrix(
                (np.ones(len(all_rows), dtype=np.float32), (all_rows, all_cols)),
                shape=(self.m, max_col),
            )
        else:
            self.pattern = None

    def _precompute_cost_stats(self) -> None:
        """Precomputes cost matrix statistics."""
        self.C_frob = np.sqrt(np.sum(self.C.data**2)) if self.C.nnz > 0 else self.EPS
        self.C_nnz = self.C.nnz

        if self.C.nnz > 0:
            C_coo = self.C.tocoo()
            self.C_row_indices = np.unique(C_coo.row)
        else:
            self.C_row_indices = np.array([], dtype=np.int64)

        self.cos_with_C = np.zeros(self.m, dtype=np.float64)
        if self.C.nnz > 0:
            for i, A_i in enumerate(self.A):
                if A_i.nnz > 0:
                    inner = (A_i.multiply(self.C)).sum()
                    norm_prod = self.norms[i] * self.C_frob + self.EPS
                    self.cos_with_C[i] = inner / norm_prod

    def compute_global_features(self) -> np.ndarray:
        """Computes global (graph-level) features.

        Global features (17 total):
            size & scale (5):
                1. log_n: log(1 + n)
                2. log_m: log(1 + m)
                3. log_n_over_m: log(1 + n/m)
                4. log_C_fro: log(1 + ||C||_F)
                5. log_avg_A_fro: log(1 + avg_i ||A_i||_F)

            density (3):
                6. avg_constraint_density: avg_i nnz(A_i) / n^2
                7. var_constraint_density: var_i nnz(A_i) / n^2
                8. C_density: nnz(C) / n^2

            constraint scale distribution (3):
                9. A_fro_mean: mean of log(1 + ||A_i||_F)
                10. A_fro_std: std of log(1 + ||A_i||_F)
                11. A_fro_p50: median of log(1 + ||A_i||_F)

            cost-constraint alignment summary (4):
                12. cos_mean: mean_i cos_i
                13. cos_std: std_i cos_i
                14. cos_max: max_i cos_i
                15. cos_min: min_i cos_i

            graph summary (2, computed after edges):
                16. avg_degree: mean node degree
                17. degree_std: std of node degrees

        Returns:
            Global feature vector of shape (17,)
        """
        n_sq = float(self.n * self.n) + self.EPS

        log_n = np.log(1.0 + self.n)
        log_m = np.log(1.0 + self.m)
        log_n_over_m = np.log(1.0 + self.n / max(self.m, 1))
        log_C_fro = np.log(1.0 + self.C_frob)
        avg_A_fro = self.norms.mean() if self.m > 0 else 0.0
        log_avg_A_fro = np.log(1.0 + avg_A_fro)

        densities = self.nnz_counts / n_sq
        avg_constraint_density = densities.mean() if self.m > 0 else 0.0
        var_constraint_density = densities.var() if self.m > 0 else 0.0
        C_density = self.C_nnz / n_sq

        log_norms = np.log(1.0 + self.norms)
        A_fro_mean = log_norms.mean() if self.m > 0 else 0.0
        A_fro_std = log_norms.std() if self.m > 0 else 0.0
        A_fro_p50 = np.median(log_norms) if self.m > 0 else 0.0

        cos_mean = self.cos_with_C.mean() if self.m > 0 else 0.0
        cos_std = self.cos_with_C.std() if self.m > 0 else 0.0
        cos_max = self.cos_with_C.max() if self.m > 0 else 0.0
        cos_min = self.cos_with_C.min() if self.m > 0 else 0.0

        avg_degree = 0.0
        degree_std = 0.0

        return np.array(
            [
                log_n,
                log_m,
                log_n_over_m,
                log_C_fro,
                log_avg_A_fro,
                avg_constraint_density,
                var_constraint_density,
                C_density,
                A_fro_mean,
                A_fro_std,
                A_fro_p50,
                cos_mean,
                cos_std,
                cos_max,
                cos_min,
                avg_degree,
                degree_std,
            ],
            dtype=np.float32,
        )

    def compute_node_features(self) -> np.ndarray:
        """Computes node features for each constraint.

        Node features (16 total):
            basic scale & density (5):
                1. log_fro: log(1 + ||A_i||_F)
                2. log_nnz: log(1 + nnz(A_i))
                3. normed_trace: trace(A_i) / (||A_i||_F + eps)
                4. diag_ratio: ||diag(A_i)||_2 / (||A_i||_F + eps)
                5. normed_rhs: b_i / (||A_i||_F + eps)

            cheap spectral proxy (1):
                6. gershgorin_bound: log(1 + max_row_sum)

            cost alignment (2):
                7. cos_with_C: inner product alignment with cost
                8. sign_cos: sign(-1, 0, or 1) of alignment

            distributional position (4):
                9. zscore_log_fro: z-score of log_fro
                10. zscore_log_nnz: z-score of log_nnz
                11. zscore_normed_rhs: z-score of |normed_rhs|
                12. fro_quantile_bin: quantile bin (0-3) for log_fro

            structural / graph position (3):
                13. log_row_set_size: log(1 + |rows(A_i)|)
                14. overlap_with_C_rows: row overlap ratio with C
                15. log_degree: log(1 + deg_i) - filled after edge computation

            Block Structure (1):
                16. num_blocks_touched: log(1 +

        Returns:
            Node feature matrix of shape (m, 16)
        """
        node_feats = np.zeros((self.m, NUM_NODE_FEATURES), dtype=np.float32)

        if self.m == 0:
            return node_feats

        node_feats[:, 0] = self.log_norms
        node_feats[:, 1] = self.log_nnz
        node_feats[:, 2] = np.clip(self.traces / (self.norms + self.EPS), -100.0, 100.0)
        node_feats[:, 3] = self.diag_norms / (self.norms + self.EPS)
        node_feats[:, 4] = self.normed_rhs
        node_feats[:, 5] = np.log(1.0 + self.gershgorin_bounds)
        node_feats[:, 6] = self.cos_with_C

        sign_cos = np.zeros(self.m, dtype=np.float32)
        sign_cos[self.cos_with_C > 0.01] = 1.0
        sign_cos[self.cos_with_C < -0.01] = -1.0
        node_feats[:, 7] = sign_cos

        node_feats[:, 8] = (self.log_norms - self.mu_log_fro) / self.sigma_log_fro
        node_feats[:, 9] = (self.log_nnz - self.mu_log_nnz) / self.sigma_log_nnz

        node_feats[:, 10] = (
            np.abs(self.normed_rhs) - self.mu_normed_rhs
        ) / self.sigma_normed_rhs

        fro_quantile_bins = np.digitize(self.log_norms, self.fro_quantiles).astype(
            np.float32
        )
        node_feats[:, 11] = fro_quantile_bins / 3.0
        node_feats[:, 12] = np.log(1.0 + self.row_sizes)

        if len(self.C_row_indices) > 0:
            C_row_set = set(self.C_row_indices)
            overlap_with_C = np.zeros(self.m, dtype=np.float32)
            for i in range(self.m):
                if self.row_sizes[i] > 0:

                    row_set_i = set(self.row_indices[i])
                    overlap_with_C[i] = len(row_set_i & C_row_set) / self.row_sizes[i]
            node_feats[:, 13] = overlap_with_C

        node_feats[:, 15] = np.log(1.0 + self.blocks_touched)

        if self.verbose:
            print(f"    [node features] computed for {self.m} nodes")

        return node_feats

    def compute_edges(
        self,
        max_neighbors: int = 15,
        similarity_threshold: float = 0.05,
        sample_size: int = 150,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes edges features vecops.

        Edge features (5 total):
            structural overlap (2):
                1. jaccard_rows: Jaccard similarity of row sets
                2. overlap_min_ratio: overlap / min(|rows_i|, |rows_j|)

            numerical coupling (3):
                3. cosine_Ai_Aj: |<A_i, A_j>| / (||A_i||_F ||A_j||_F)
                4. log_min_fro: log(1 + min(||A_i||_F, ||A_j||_F))
                5. fro_diff: |log ||A_i||_F - log ||A_j||_F|

        Returns:
            Tuple of (edge_index, edge_attr) where:
                edge_index: Shape (2, num_edges), COO format indices
                edge_attr: Shape (num_edges, 5), edge features
        """
        if self.m == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros(
                (0, NUM_EDGE_FEATURES), dtype=np.float32
            )

        if self.verbose:
            print(f"    computing edges for m={self.m}...")

        if self.m >= 1000 and self.pattern is not None:
            return self._compute_edges_sparse(max_neighbors, similarity_threshold)

        edges_dict: Dict[Tuple[int, int], List[float]] = {}
        self._compute_edges_full(edges_dict, similarity_threshold)

        if not edges_dict:
            return self._knn_fallback(max_neighbors)

        edge_list = []
        edge_features = []
        for (i, j), feat in edges_dict.items():
            edge_list.extend([[i, j], [j, i]])
            edge_features.extend([feat, feat])

        return (
            np.array(edge_list, dtype=np.int64).T,
            np.array(edge_features, dtype=np.float32),
        )

    def _compute_edges_sparse(
        self,
        max_neighbors: int,
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fast edge computation using sparse pattern matrix multiplication.

        Instead of computing pairwise overlaps in Python loops, we build a
        binary pattern matrix P where P[i,k]=1 if constraint i has nonzero
        at row k. Then overlap = P @ P.T gives all pairwise overlaps in O(nnz^2/m).
        """
        if self.verbose:
            print("    computing overlap matrix via sparse multiplication...")

        overlap = self.pattern @ self.pattern.T
        overlap = overlap.tocsr()

        if self.verbose:
            print(f"    overlap matrix computed, extracting edges...")

        edge_list = []
        edge_features = []

        for i in range(self.m):
            if self.row_sizes[i] == 0:
                continue

            row_start = overlap.indptr[i]
            row_end = overlap.indptr[i + 1]
            js = overlap.indices[row_start:row_end]
            overlaps = overlap.data[row_start:row_end]

            mask = js > i
            js = js[mask]
            overlaps = overlaps[mask].astype(np.float64)

            if len(js) == 0:
                continue

            unions = self.row_sizes[i] + self.row_sizes[js] - overlaps
            jaccards = overlaps / (unions + self.EPS)

            valid = jaccards >= threshold
            js = js[valid]
            overlaps = overlaps[valid]
            jaccards = jaccards[valid]

            if len(js) == 0:
                continue

            if len(js) > max_neighbors:
                top_k = np.argpartition(-jaccards, max_neighbors)[:max_neighbors]
                js = js[top_k]
                overlaps = overlaps[top_k]
                jaccards = jaccards[top_k]

            overlap_mins = overlaps / (
                np.minimum(self.row_sizes[i], self.row_sizes[js]) + self.EPS
            )
            log_min_fros = np.minimum(self.log_norms[i], self.log_norms[js])
            fro_diffs = np.abs(self.log_norms[i] - self.log_norms[js])

            if len(js) <= 50:

                cosines = np.zeros(len(js), dtype=np.float64)
                for k, j in enumerate(js):
                    inner = np.abs((self.A[i].multiply(self.A[j])).sum())
                    cosines[k] = inner / (self.norms[i] * self.norms[j] + self.EPS)
            else:

                cosines = jaccards

            for k, j in enumerate(js):
                feat = [
                    jaccards[k],
                    overlap_mins[k],
                    cosines[k],
                    log_min_fros[k],
                    fro_diffs[k],
                ]
                edge_list.extend([[i, j], [j, i]])
                edge_features.extend([feat, feat])

            if self.verbose and (i + 1) % 5000 == 0:
                print(f"      [edge proc] {i+1}/{self.m}, edges: {len(edge_list)//2}")

        if not edge_list:
            return self._knn_fallback(max_neighbors)

        return (
            np.array(edge_list, dtype=np.int64).T,
            np.array(edge_features, dtype=np.float32),
        )

    def _compute_edge_features(self, i: int, j: int) -> Optional[List[float]]:
        """Computes edge feature vector for a pair of constraints."""
        rows_i = self.row_indices[i]
        rows_j = self.row_indices[j]

        if len(rows_i) == 0 or len(rows_j) == 0:
            return None

        set_i = set(rows_i)
        set_j = set(rows_j)
        intersection = len(set_i & set_j)

        if intersection == 0:
            return None

        union = len(set_i | set_j)
        jaccard = intersection / union
        overlap_min_ratio = intersection / (min(len(set_i), len(set_j)) + self.EPS)

        A_i, A_j = self.A[i], self.A[j]
        inner = np.abs((A_i.multiply(A_j)).sum())
        cosine_Ai_Aj = inner / (self.norms[i] * self.norms[j] + self.EPS)

        log_min_fro = min(self.log_norms[i], self.log_norms[j])
        fro_diff = np.abs(self.log_norms[i] - self.log_norms[j])

        return [jaccard, overlap_min_ratio, cosine_Ai_Aj, log_min_fro, fro_diff]

    def _compute_edges_full(
        self,
        edges_dict: Dict[Tuple[int, int], List[float]],
        threshold: float,
    ) -> None:
        """Full pairwise comparison for small problems."""
        for i in range(self.m):
            for j in range(i + 1, self.m):
                feat = self._compute_edge_features(i, j)
                if feat is not None and feat[0] >= threshold:
                    edges_dict[(i, j)] = feat

            if self.verbose and (i + 1) % 200 == 0:
                print(f"      [full edges] {i+1}/{self.m}")

    def _knn_fallback(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """K-NN fallback based on norm similarity when no overlap edges found."""
        if self.verbose:
            print("    using k-NN fallback (no overlap edges)...")

        k = min(k, self.m - 1)
        if k <= 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros(
                (0, NUM_EDGE_FEATURES), dtype=np.float32
            )

        edges_dict: Dict[Tuple[int, int], List[float]] = {}

        for i in range(self.m):
            dists = np.abs(self.log_norms - self.log_norms[i])
            dists[i] = np.inf
            nearest = np.argpartition(dists, k)[:k]

            for j in nearest:
                key = (min(i, int(j)), max(i, int(j)))
                if key not in edges_dict:
                    sim = 1.0 / (1.0 + dists[j])
                    log_min_fro = min(self.log_norms[i], self.log_norms[j])
                    fro_diff = np.abs(self.log_norms[i] - self.log_norms[j])
                    edges_dict[key] = [sim, sim, 0.0, log_min_fro, fro_diff]

        if not edges_dict:
            return np.zeros((2, 0), dtype=np.int64), np.zeros(
                (0, NUM_EDGE_FEATURES), dtype=np.float32
            )

        edge_list = []
        edge_features = []
        for (i, j), feat in edges_dict.items():
            edge_list.extend([[i, j], [j, i]])
            edge_features.extend([feat, feat])

        return (
            np.array(edge_list, dtype=np.int64).T,
            np.array(edge_features, dtype=np.float32),
        )


def process_sdpa_to_pyg(
    input_path: str,
    output_path: str,
    max_neighbors: int = 15,
    similarity_threshold: float = 0.05,
    verbose: bool = False,
) -> Data:
    """Converts SDPA file to PyG Data object.

    Args:
        input_path: Path to input .dat-s file
        output_path: Path to save output .pt file
        max_neighbors: Maximum edges per node for large problems
        similarity_threshold: Minimum Jaccard similarity for edges
        verbose: If True, print progress information

    Returns:
        PyG Data object with attributes:
            x: Node features of shape (m, 16)
            edge_index: Edge indices of shape (2, num_edges)
            edge_attr: Edge features of shape (num_edges, 5)
            global_attr: Global features of shape (17,)
            num_nodes: Number of nodes (m)
    """
    if verbose:
        print(f"[1/4] parsing {input_path}...")

    parser = SDPAParser(input_path)
    parser.parse(verbose=verbose)
    C, A, b, m, n, block_offsets = parser.get_data()

    if verbose:
        print(f"[2/4] problem size: m={m}, n={n}, nblocks={len(block_offsets)-1}")
    extractor = FeatureExtractor(
        C, A, b, m, n, block_offsets=block_offsets, verbose=verbose
    )

    if verbose:
        print("[3/4] computing features...")
    global_feats = extractor.compute_global_features()
    node_feats = extractor.compute_node_features()

    if verbose:
        print("[4/4] computing edges...")
    edge_index, edge_attr = extractor.compute_edges(
        max_neighbors=max_neighbors,
        similarity_threshold=similarity_threshold,
    )

    if edge_index.shape[1] > 0:
        degrees = np.bincount(edge_index[0], minlength=m)
        avg_degree = degrees.mean()
        degree_std = degrees.std()
        global_feats[15] = avg_degree
        global_feats[16] = degree_std

        for i in range(m):
            node_feats[i, 14] = np.log(1.0 + degrees[i])

    data = Data(
        x=torch.tensor(node_feats, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        global_attr=torch.tensor(global_feats, dtype=torch.float32),
        num_nodes=m,
    )

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torch.save(data, output_path)

    return data


def extract_features_only(
    input_path: str,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts features without saving, for visualization purposes.

    Args:
        input_path: Path to input .dat-s file
        verbose: If True, print progress information

    Returns:
        Tuple of (global_feats, node_feats, edge_feats) as numpy arrays
    """
    parser = SDPAParser(input_path)
    parser.parse(verbose=verbose)
    C, A, b, m, n, block_offsets = parser.get_data()

    extractor = FeatureExtractor(
        C, A, b, m, n, block_offsets=block_offsets, verbose=verbose
    )
    global_feats = extractor.compute_global_features()
    node_feats = extractor.compute_node_features()
    edge_index, edge_attr = extractor.compute_edges()

    if edge_index.shape[1] > 0:
        degrees = np.bincount(edge_index[0], minlength=m)
        global_feats[15] = degrees.mean()
        global_feats[16] = degrees.std()
        for i in range(m):
            node_feats[i, 14] = np.log(1.0 + degrees[i])

    return global_feats, node_feats, edge_attr


def log_graph_info(data: Data, instance_name: str) -> None:
    """Print summary statistics of the generated graph."""
    print(f"\n>>> {instance_name}")

    print("\n[global features]")
    global_names = [
        "log_n",
        "log_m",
        "log_n_over_m",
        "log_C_fro",
        "log_avg_A_fro",
        "avg_density",
        "var_density",
        "C_density",
        "A_fro_mean",
        "A_fro_std",
        "A_fro_p50",
        "cos_mean",
        "cos_std",
        "cos_max",
        "cos_min",
        "avg_degree",
        "degree_std",
    ]
    for i, name in enumerate(global_names):
        print(f"  {name:<16}: {data.global_attr[i].item():.4f}")

    print("\n[node features]")
    node_names = [
        "log_fro",
        "log_nnz",
        "normed_trace",
        "diag_ratio",
        "normed_rhs",
        "gershgorin",
        "cos_C",
        "sign_cos",
        "zscore_fro",
        "zscore_nnz",
        "zscore_rhs",
        "fro_qbin",
        "log_row_size",
        "overlap_C",
        "log_degree",
        "num_blocks",
    ]
    node_feats = data.x.numpy()
    print(f"{'feature':<14} {'min':>9} {'max':>9} {'mean':>9} {'std':>9}")
    print("-" * 54)
    for i, name in enumerate(node_names):
        col = node_feats[:, i]
        print(
            f"{name:<14} {col.min():>9.4f} {col.max():>9.4f} "
            f"{col.mean():>9.4f} {col.std():>9.4f}"
        )

    print("\n[edge features]")
    edge_names = ["jaccard", "overlap_min", "cosine", "log_min_fro", "fro_diff"]
    if data.edge_attr.shape[0] > 0:
        edge_feats = data.edge_attr.numpy()
        print(f"{'feature':<12} {'min':>9} {'max':>9} {'mean':>9}")
        print("-" * 42)
        for i, name in enumerate(edge_names):
            col = edge_feats[:, i]
            print(f"{name:<12} {col.min():>9.4f} {col.max():>9.4f} {col.mean():>9.4f}")

    print(f"\n[graph topology]")
    m = data.num_nodes
    num_edges = data.edge_index.shape[1] // 2
    avg_degree = num_edges * 2 / m if m > 0 else 0
    print(f"  nodes: {m}, edges: {num_edges}, avg_degree: {avg_degree:.2f}")


if __name__ == "__main__":
    import argparse
    import time

    arg_parser = argparse.ArgumentParser(
        description="Convert SDPA files to PyG graphs for GNN rank prediction"
    )
    arg_parser.add_argument("--input", type=str, required=True)
    arg_parser.add_argument("--output", type=str, default=None)
    arg_parser.add_argument("--max-neighbors", type=int, default=15)
    arg_parser.add_argument("--similarity-threshold", type=float, default=0.05)
    arg_parser.add_argument("-v", "--verbose", action="store_true")

    args = arg_parser.parse_args()
    if args.output is None:
        stem = Path(args.input).stem
        args.output = f"dataset/proc/{stem}.pt"

    print(f"[processing] {args.input}")
    print(f"[output] {args.output}")

    start_time = time.time()
    data = process_sdpa_to_pyg(
        args.input,
        args.output,
        max_neighbors=args.max_neighbors,
        similarity_threshold=args.similarity_threshold,
        verbose=args.verbose,
    )
    elapsed = time.time() - start_time

    log_graph_info(data, Path(args.input).stem)
    print(f"\n[done] saved to {args.output} in {elapsed:.2f}s")
