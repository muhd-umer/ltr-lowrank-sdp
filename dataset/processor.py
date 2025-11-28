"""\
This module does conversion of SDP problem instances in SDPA-sparse format to PyG Data objects for oracle rank prediction/schedule in low-rank SDP solvers.

Extracted features are made to be discriminative for predicting the optimal rank schedule for a given SDP problem instance. Such features capture various aspects of the problem structure, including:
    - Spectral features capture constraint "stiffness" (eigenvalue magnitude)
    - Structural features capture sparsity patterns and interactions
    - Scale features enable cross-problem generalization

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


class SDPAParser:
    """Streaming parser for SDPA-sparse format files.

    Parses SDP problem files using memory-efficient COO construction.
    Handles files with millions of entries through streaming I/O.

    The SDPA-sparse format specifies:
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

    Example:
        >>> parser = SDPAParser("problem.dat-s")
        >>> parser.parse(verbose=True)
        >>> C, A, b, m, n = parser.get_data()
    """

    def __init__(self, filepath: str) -> None:
        """Initializes parser with file path.

        Args:
            filepath: Path to .dat-s file.
        """
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
        """Parses SDPA file using streaming for memory efficiency.

        Args:
            verbose: If True, print progress information.

        Raises:
            ValueError: If file is empty, invalid, or has dimension mismatch.
        """
        with open(self.filepath, "r") as f:
            # skip comments and read header
            line = self._read_noncomment_line(f)
            if line is None:
                raise ValueError(f"Empty or invalid SDPA file: {self.filepath}")

            # parse header: m, nblocks, block dimensions
            self.m = int(line.strip())
            self.nblocks = int(self._read_noncomment_line(f).strip())

            block_line = self._read_noncomment_line(f).strip()
            block_tokens = self._parse_block_dims(block_line)

            # keep only sdp blocks (positive dims), skip lp blocks (negative)
            self.block_dims = [d for d in block_tokens if d > 0]
            self.nblocks = len(self.block_dims)

            # compute block offsets for global indexing
            self.block_offsets = [0]
            for dim in self.block_dims:
                self.block_offsets.append(self.block_offsets[-1] + dim)
            self.n = self.block_offsets[-1]

            # parse rhs vector
            rhs_line = self._read_noncomment_line(f).strip()
            rhs_tokens = (
                rhs_line.replace(",", " ").replace("{", " ").replace("}", " ").split()
            )
            self.b = np.array([float(x) for x in rhs_tokens], dtype=np.float64)

            if len(self.b) != self.m:
                raise ValueError(
                    f"RHS dimension mismatch: got {len(self.b)}, expected {self.m}"
                )

            # parse matrix entries using coo format for efficiency
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
                    blk = int(parts[1]) - 1  # 0-indexed
                    row = int(parts[2]) - 1
                    col = int(parts[3]) - 1
                    value = float(parts[4])
                except ValueError:
                    continue

                if blk < 0 or blk >= self.nblocks:
                    continue

                # convert to global indices
                gr = self.block_offsets[blk] + row
                gc = self.block_offsets[blk] + col

                if matno == 0:
                    # cost matrix C
                    C_rows.append(gr)
                    C_cols.append(gc)
                    C_data.append(value)
                    if gr != gc:  # symmetrize off-diagonal entries
                        C_rows.append(gc)
                        C_cols.append(gr)
                        C_data.append(value)
                else:
                    # constraint matrix A_{matno}
                    mat_idx = matno - 1
                    if mat_idx < self.m:
                        A_coo[mat_idx]["rows"].append(gr)
                        A_coo[mat_idx]["cols"].append(gc)
                        A_coo[mat_idx]["data"].append(value)
                        if gr != gc:  # symmetrize
                            A_coo[mat_idx]["rows"].append(gc)
                            A_coo[mat_idx]["cols"].append(gr)
                            A_coo[mat_idx]["data"].append(value)

                entry_count += 1
                if verbose and entry_count % 500000 == 0:
                    print(f"    [parsed] {entry_count:,} entries...")

        if verbose:
            print("    converting to CSR format...")

        # construct sparse matrices
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
        """Reads the next non-comment, non-empty line.

        Args:
            f: File handle.

        Returns:
            Next valid line or None if EOF.
        """
        while True:
            line = f.readline()
            if not line:
                return None
            line = line.strip()
            if line and not line.startswith("*") and not line.startswith('"'):
                return line

    def _parse_block_dims(self, line: str) -> List[int]:
        """Parses block dimensions from various SDPA formats.

        Args:
            line: Line containing block dimensions.

        Returns:
            List of block dimension integers.
        """
        cleaned = line.replace("(", " ").replace(")", " ")
        cleaned = cleaned.replace(",", " ").replace("{", " ").replace("}", " ")
        tokens = cleaned.split()
        return [int(x) for x in tokens if x.lstrip("-").isdigit()]

    def get_data(
        self,
    ) -> Tuple[sp.csr_matrix, List[sp.csr_matrix], np.ndarray, int, int]:
        """Returns parsed problem data.

        Returns:
            Tuple of (C, A, b, m, n) where:
                C: Cost matrix (n x n sparse CSR).
                A: List of m constraint matrices (n x n sparse CSR).
                b: RHS vector of shape (m,).
                m: Number of constraints.
                n: Matrix dimension.
        """
        return self.C, self.A, self.b, self.m, self.n


class ScalableFeatureExtractor:
    """Extracts GNN features from SDP problems for rank prediction.

    All operations are O(m * avg_nnz) to ensure scalability to large problems.
    Features are designed to be discriminative for predicting optimal rank
    schedules in low-rank SDP solvers.

    Feature Design Rationale:
        - Spectral scale (lambda_max): Primary predictor of rank requirement.
          Constraints with larger eigenvalues typically need higher rank.
        - Frobenius norm: Overall constraint magnitude/scale.
        - Trace: Sign indicates positive/negative definiteness tendency.
        - Diagonal dominance: High values suggest simpler structure.
        - Sparsity (nnz): Denser constraints often need higher rank.
        - Cost overlap: Constraints affecting objective rows are critical.

    Attributes:
        C: Cost matrix.
        A: List of constraint matrices.
        b: RHS vector.
        m: Number of constraints.
        n: Matrix dimension.
        norms: Frobenius norms of each constraint.
        nnz_counts: Nonzero counts per constraint.
        traces: Trace of each constraint matrix.
        diag_norms: L2 norm of diagonal of each constraint.
        max_abs_vals: Maximum absolute value in each constraint.
        spectral_scales: Largest eigenvalue magnitude of each constraint.
        row_sets: Set of nonzero row indices for each constraint.

    Example:
        >>> extractor = ScalableFeatureExtractor(C, A, b, m, n)
        >>> node_feats = extractor.compute_node_features()
        >>> edge_index, edge_attr = extractor.compute_edges_fast()
    """

    EPS = 1e-8

    def __init__(
        self,
        C: sp.csr_matrix,
        A: List[sp.csr_matrix],
        b: np.ndarray,
        m: int,
        n: int,
        verbose: bool = False,
    ) -> None:
        """Initializes feature extractor with problem data.

        Args:
            C: Cost matrix (n x n sparse).
            A: List of m constraint matrices.
            b: RHS vector of shape (m,).
            m: Number of constraints.
            n: Matrix dimension.
            verbose: If True, print progress information.
        """
        self.C = C
        self.A = A
        self.b = b
        self.m = m
        self.n = n
        self._precompute_constraint_stats(verbose=verbose)

    def _precompute_constraint_stats(self, verbose: bool = False) -> None:
        """Precomputes constraint statistics for efficient feature extraction.

        Computes norms, traces, spectral scales, and row index sets for all
        constraints. The spectral scale (largest eigenvalue magnitude) is
        crucial for rank prediction.

        Args:
            verbose: If True, print progress information.
        """
        self.norms = np.zeros(self.m, dtype=np.float64)
        self.nnz_counts = np.zeros(self.m, dtype=np.int64)
        self.traces = np.zeros(self.m, dtype=np.float64)
        self.diag_norms = np.zeros(self.m, dtype=np.float64)
        self.max_abs_vals = np.zeros(self.m, dtype=np.float64)
        self.spectral_scales = np.zeros(self.m, dtype=np.float64)

        self.row_sets: List[set] = []

        for i, A_i in enumerate(self.A):
            self.nnz_counts[i] = A_i.nnz
            if A_i.nnz > 0:
                # basic matrix statistics
                self.norms[i] = sp.linalg.norm(A_i, "fro")
                self.traces[i] = A_i.diagonal().sum()
                diag = A_i.diagonal()
                self.diag_norms[i] = np.linalg.norm(diag)
                self.max_abs_vals[i] = (
                    np.abs(A_i.data).max() if A_i.data.size > 0 else 0
                )

                # spectral scale; largest magnitude eigenvalue
                try:
                    vals = sp.linalg.eigsh(
                        A_i, k=1, which="LM", return_eigenvectors=False
                    )
                    self.spectral_scales[i] = np.abs(vals[0])
                except (sp.linalg.ArpackNoConvergence, ValueError, RuntimeError):
                    try:
                        vals = sp.linalg.eigsh(
                            A_i,
                            k=1,
                            which="LM",
                            return_eigenvectors=False,
                            maxiter=1000,
                            tol=1e-3,
                        )
                        self.spectral_scales[i] = np.abs(vals[0])
                    except (sp.linalg.ArpackNoConvergence, ValueError, RuntimeError):
                        self.spectral_scales[i] = sp.linalg.norm(A_i, np.inf)

                coo = A_i.tocoo()
                self.row_sets.append(set(coo.row))
            else:
                self.row_sets.append(set())

            if verbose and (i + 1) % 2000 == 0:
                print(f"    [precomputed stats] {i+1}/{self.m}")

        self.C_frob = sp.linalg.norm(self.C, "fro") if self.C.nnz > 0 else 0.0
        self.C_nnz = self.C.nnz
        if self.C.nnz > 0:
            C_coo = self.C.tocoo()
            self.C_row_set = set(C_coo.row)
        else:
            self.C_row_set = set()

    def compute_global_features(self) -> np.ndarray:
        """Computes global (graph-level) features.

        Global features provide problem-level context that helps the GNN
        generalize across different problem sizes and types.

        Features (5 total):
            1. log(n): Matrix dimension scale
            2. log(m): Constraint count scale
            3. log(1 + ||C||_F): Cost matrix magnitude
            4. log(1 + avg_nnz): Average constraint density
            5. log(1 + C_nnz): Cost matrix density

        Returns:
            Global feature vector of shape (5,).
        """
        avg_nnz = self.nnz_counts.mean() if self.m > 0 else 0

        return np.array(
            [
                np.log(max(self.n, 1)),
                np.log(max(self.m, 1)),
                np.log(1.0 + self.C_frob),
                np.log(1.0 + avg_nnz),
                np.log(1.0 + self.C_nnz),
            ],
            dtype=np.float32,
        )

    def compute_node_features(self, verbose: bool = False) -> np.ndarray:
        """Computes node features for each constraint.

        Each constraint becomes a node in the graph. Features are designed
        to capture properties relevant to rank prediction:

        Features (8 total):
            1. log(1 + ||A_i||_F): Frobenius norm (constraint scale)
            2. log(1 + nnz): Sparsity (denser often needs higher rank)
            3. trace / ||A_i||_F: Normalized trace (definiteness indicator)
            4. ||diag(A_i)|| / ||A_i||_F: Diagonal dominance
            5. b_i / ||A_i||_F: Normalized RHS value
            6. |rows(A_i) ∩ rows(C)| / |rows(A_i)|: Cost matrix overlap
            7. log(1 + max|A_i|): Maximum absolute value
            8. log(1 + |lambda_max|): Spectral scale (KEY rank predictor)

        Args:
            verbose: If True, print progress information.

        Returns:
            Node feature matrix of shape (m, 8).
        """
        node_feats = np.zeros((self.m, 8), dtype=np.float32)

        for i in range(self.m):
            norm_i = self.norms[i]

            # scale features (log-transformed)
            log_frob = np.log(1.0 + norm_i)
            log_nnz = np.log(1.0 + self.nnz_counts[i])

            # normalized features (clipped for stability)
            norm_trace = np.clip(self.traces[i] / (norm_i + self.EPS), -100.0, 100.0)
            diag_dom = self.diag_norms[i] / (norm_i + self.EPS)
            norm_rhs = np.clip(self.b[i] / (norm_i + self.EPS), -100.0, 100.0)

            # cost matrix interaction
            if self.row_sets[i] and self.C_row_set:
                overlap = len(self.row_sets[i] & self.C_row_set) / len(self.row_sets[i])
            else:
                overlap = 0.0

            # scale and spectral features
            log_max = np.log(1.0 + self.max_abs_vals[i])
            log_spectral = np.log(1.0 + self.spectral_scales[i])

            node_feats[i, :] = [
                log_frob,
                log_nnz,
                norm_trace,
                diag_dom,
                norm_rhs,
                overlap,
                log_max,
                log_spectral,
            ]

            if verbose and (i + 1) % 5000 == 0:
                print(f"    [node features] {i+1}/{self.m}")

        return node_feats

    def compute_edges_fast(
        self,
        max_neighbors: int = 10,
        overlap_threshold: float = 0.1,
        sample_size: int = 100,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes edges using structural overlap heuristic.

        Edges represent interactions between constraints. Two constraints
        are connected if their nonzero patterns overlap significantly.

        Strategy:
            - For small m (< 1000): Full pairwise overlap computation
            - For large m: Sample-based candidate selection

        The Jaccard similarity of row indices is used instead of cosine
        similarity of matrix values for efficiency. This captures structural
        overlap (which rows interact) rather than geometric coupling
        (value-weighted inner product). For sparse matrices, structural
        overlap is a valid proxy since matrices only have non-zero inner
        products if their sparsity patterns overlap.

        Edge Features (2 total):
            1. Jaccard similarity: |rows_i ∩ rows_j| / |rows_i ∪ rows_j|
            2. log(1 + min(nnz_i, nnz_j)): Scale of interaction

        Args:
            max_neighbors: Maximum edges per node (for large problems).
            overlap_threshold: Minimum Jaccard similarity to create edge.
            sample_size: Number of candidates to sample for large problems.
            verbose: If True, print progress information.

        Returns:
            Tuple of (edge_index, edge_attr) where:
                edge_index: Shape (2, num_edges), COO format indices.
                edge_attr: Shape (num_edges, 2), edge features.
        """
        if self.m == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

        if verbose:
            print(f"    Computing edges for m={self.m}...")

        edges_dict: Dict[Tuple[int, int], List[float]] = {}

        if self.m < 1000:
            self._compute_edges_full(edges_dict, overlap_threshold, verbose)
        else:
            self._compute_edges_sampled(
                edges_dict, max_neighbors, overlap_threshold, sample_size, verbose
            )

        if not edges_dict:
            return self._knn_fallback(max_neighbors, verbose)

        edge_list = []
        edge_features = []
        for (i, j), feat in edges_dict.items():
            edge_list.extend([[i, j], [j, i]])
            edge_features.extend([feat, feat])

        return (
            np.array(edge_list, dtype=np.int64).T,
            np.array(edge_features, dtype=np.float32),
        )

    def _compute_edges_full(
        self,
        edges_dict: Dict[Tuple[int, int], List[float]],
        threshold: float,
        verbose: bool,
    ) -> None:
        """Computes edges via full pairwise comparison for small problems.

        Args:
            edges_dict: Dictionary to store edges (modified in place).
            threshold: Minimum Jaccard similarity threshold.
            verbose: If True, print progress information.
        """
        for i in range(self.m):
            set_i = self.row_sets[i]
            if not set_i:
                continue

            for j in range(i + 1, self.m):
                set_j = self.row_sets[j]
                if not set_j:
                    continue

                # jaccard similarity of row indices
                intersection = len(set_i & set_j)
                if intersection == 0:
                    continue

                union = len(set_i | set_j)
                jaccard = intersection / union

                if jaccard >= threshold:
                    min_nnz = min(self.nnz_counts[i], self.nnz_counts[j])
                    edges_dict[(i, j)] = [jaccard, np.log(1.0 + min_nnz)]

            if verbose and (i + 1) % 200 == 0:
                print(f"      [full edges] {i+1}/{self.m}")

    def _compute_edges_sampled(
        self,
        edges_dict: Dict[Tuple[int, int], List[float]],
        max_neighbors: int,
        threshold: float,
        sample_size: int,
        verbose: bool,
    ) -> None:
        """Computes edges via sampling for large problems.

        Uses row-based grouping to find likely candidates efficiently,
        then samples additional random candidates to ensure coverage.

        Args:
            edges_dict: Dictionary to store edges (modified in place).
            max_neighbors: Maximum neighbors per node.
            threshold: Minimum Jaccard similarity threshold.
            sample_size: Number of candidates to consider per node.
            verbose: If True, print progress information.
        """
        # group constraints by representative row for efficient lookup
        row_to_constraints: Dict[int, List[int]] = {}
        for i, row_set in enumerate(self.row_sets):
            if row_set:
                rep_row = min(row_set)
                row_to_constraints.setdefault(rep_row, []).append(i)

        for i in range(self.m):
            set_i = self.row_sets[i]
            if not set_i:
                continue

            candidates = set()
            for row in list(set_i)[:20]:
                if row in row_to_constraints:
                    candidates.update(row_to_constraints[row])

            # sample additional random candidates if needed
            if len(candidates) < sample_size:
                remaining = sample_size - len(candidates)
                others = [x for x in range(self.m) if x != i and x not in candidates]
                if others and remaining > 0:
                    sampled = np.random.choice(
                        others, size=min(remaining, len(others)), replace=False
                    )
                    candidates.update(sampled)
            candidates.discard(i)

            # compute overlap
            neighbors = []
            for j in candidates:
                set_j = self.row_sets[j]
                if not set_j:
                    continue

                intersection = len(set_i & set_j)
                if intersection == 0:
                    continue

                union = len(set_i | set_j)
                jaccard = intersection / union

                if jaccard >= threshold:
                    min_nnz = min(self.nnz_counts[i], self.nnz_counts[j])
                    neighbors.append((j, jaccard, np.log(1.0 + min_nnz)))

            neighbors.sort(key=lambda x: -x[1])
            for j, jaccard, log_min_nnz in neighbors[:max_neighbors]:
                key = (min(i, j), max(i, j))
                if key not in edges_dict:
                    edges_dict[key] = [jaccard, log_min_nnz]

            if verbose and (i + 1) % 2000 == 0:
                print(
                    f"      Sampled edges: {i+1}/{self.m}, edges so far: {len(edges_dict)}"
                )

    def _knn_fallback(self, k: int, verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Constructs k-NN graph based on norm similarity as fallback.

        Used when no structural overlap edges are found (e.g., for diagonal
        constraint matrices like in MaxCut problems).

        Args:
            k: Number of nearest neighbors.
            verbose: If True, print progress information.

        Returns:
            Tuple of (edge_index, edge_attr).
        """
        if verbose:
            print("    using K-NN fallback (no overlap edges)...")

        k = min(k, self.m - 1)
        if k <= 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

        log_norms = np.log(1.0 + self.norms)
        edges_dict: Dict[Tuple[int, int], List[float]] = {}

        for i in range(self.m):
            dists = np.abs(log_norms - log_norms[i])
            dists[i] = np.inf  # exclude self
            nearest = np.argpartition(dists, k)[:k]

            for j in nearest:
                key = (min(i, j), max(i, j))
                if key not in edges_dict:
                    # similarity = inverse distance
                    sim = 1.0 / (1.0 + dists[j])
                    min_nnz = min(self.nnz_counts[i], self.nnz_counts[j])
                    edges_dict[key] = [sim, np.log(1.0 + min_nnz)]

        if not edges_dict:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

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
    max_neighbors: int = 10,
    overlap_threshold: float = 0.1,
    verbose: bool = False,
) -> Data:
    """Converts SDPA file to PyTorch Geometric Data object.

    Main entry point for converting SDP problems to graph format suitable
    for GNN-based rank prediction.

    Args:
        input_path: Path to input .dat-s file.
        output_path: Path to save output .pt file.
        max_neighbors: Maximum edges per node for large problems.
        overlap_threshold: Minimum Jaccard similarity for edges.
        verbose: If True, print progress information.

    Returns:
        PyG Data object with attributes:
            x: Node features of shape (m, 8).
            edge_index: Edge indices of shape (2, num_edges).
            edge_attr: Edge features of shape (num_edges, 2).
            global_attr: Global features of shape (5,).
            num_nodes: Number of nodes (m).

    Example:
        >>> data = process_sdpa_to_pyg("problem.dat-s", "output.pt")
        >>> print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")
    """
    if verbose:
        print(f"[1/4] parsing {input_path}...")

    parser = SDPAParser(input_path)
    parser.parse(verbose=verbose)
    C, A, b, m, n = parser.get_data()

    if verbose:
        print(f"[2/4] problem size: m={m}, n={n}")
    extractor = ScalableFeatureExtractor(C, A, b, m, n, verbose=verbose)

    if verbose:
        print("[3/4] computing features...")
    global_feats = extractor.compute_global_features()
    node_feats = extractor.compute_node_features(verbose=verbose)

    if verbose:
        print("[4/4] computing edges...")
    edge_index, edge_attr = extractor.compute_edges_fast(
        max_neighbors=max_neighbors,
        overlap_threshold=overlap_threshold,
        verbose=verbose,
    )

    # construct pyg data object
    data = Data(
        x=torch.tensor(node_feats, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        global_attr=torch.tensor(global_feats, dtype=torch.float32),
        num_nodes=m,
    )

    # save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)

    return data


def log_graph_info(data: Data, instance_name: str) -> None:
    """Print summary statistics of the generated graph.

    Args:
        data: PyG Data object.
        instance_name: Name of the instance for display.
    """
    print(f"p:{instance_name}")
    print("\n[global features]")
    print(f"  log(n):         {data.global_attr[0].item():.4f}")
    print(f"  log(m):         {data.global_attr[1].item():.4f}")
    print(f"  log(1+||C||_F): {data.global_attr[2].item():.4f}")
    print(f"  log(1+avg_nnz): {data.global_attr[3].item():.4f}")
    print(f"  log(1+C_nnz):   {data.global_attr[4].item():.4f}")

    n_approx = int(np.exp(data.global_attr[0].item()))
    m_approx = int(np.exp(data.global_attr[1].item()))
    print(f"  (n ≈ {n_approx}, m ≈ {m_approx})")

    print("\n[node features]")
    node_feats = data.x.numpy()
    feature_names = [
        "log(1+||A_i||_F)",
        "log(1+nnz)",
        "norm_trace",
        "diag_dominance",
        "norm_rhs",
        "C_overlap",
        "log(1+max_val)",
        "log(1+|lambda_max|)",
    ]

    print(f"{'feature':<18} {'min':>10} {'max':>10} {'mean':>10} {'std':>10}")
    print("-" * 60)
    for i, name in enumerate(feature_names):
        col = node_feats[:, i]
        nan_count = int(np.isnan(col).sum())
        if nan_count > 0:
            print(f"{name:<18} {'NaN':>10} {'NaN':>10} {'NaN':>10} {nan_count:>10}")
        else:
            print(
                f"{name:<18} {col.min():>10.4f} {col.max():>10.4f} "
                f"{col.mean():>10.4f} {col.std():>10.4f}"
            )

    print("\n[graph topology]")
    m = data.num_nodes
    num_edges = data.edge_index.shape[1] // 2
    avg_degree = num_edges * 2 / m if m > 0 else 0
    print(f"  node count (m):   {m}")
    print(f"  edge count:       {num_edges}")
    print(f"  average degree:   {avg_degree:.2f}")

    if num_edges == 0:
        print("  [warning] no edges created")
    elif avg_degree < 1.0:
        print("  [warning] very sparse graph (avg degree < 1)")

    if data.edge_attr.shape[0] > 0:
        print("\n[edge features]")
        edge_attr = data.edge_attr.numpy()
        print(
            f"  similarity:  min={edge_attr[:, 0].min():.4f}, "
            f"max={edge_attr[:, 0].max():.4f}, mean={edge_attr[:, 0].mean():.4f}"
        )
        print(
            f"  scale:       min={edge_attr[:, 1].min():.4f}, "
            f"max={edge_attr[:, 1].max():.4f}, mean={edge_attr[:, 1].mean():.4f}"
        )


if __name__ == "__main__":
    import argparse
    import time

    arg_parser = argparse.ArgumentParser(
        description="Convert SDPA files to PyG graphs for GNN rank prediction"
    )
    arg_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input SDPA file path (.dat-s)",
    )
    arg_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pt file path (auto-generated if not specified)",
    )
    arg_parser.add_argument(
        "--max-neighbors",
        type=int,
        default=10,
        help="Maximum neighbors per node for edge construction",
    )
    arg_parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.1,
        help="Minimum Jaccard similarity for edge creation",
    )
    arg_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose progress information",
    )

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
        overlap_threshold=args.overlap_threshold,
        verbose=args.verbose,
    )
    elapsed = time.time() - start_time

    log_graph_info(data, Path(args.input).stem)
    print(f"[p] successfully saved to {args.output}")
    print(f"  processing time: {elapsed:.2f}s")
