"""Training-only recommendation models for positive implicit TV interactions.

All model inputs use positional user/item indices, not raw source IDs. ``fit``
never edits its inputs. Candidate indices address the same complete metadata
universe as the columns of X; future-only items therefore have empty X columns.
Raw model scores have different scales and are not calibrated probabilities.
Seen items are deliberately not masked: TV repeat viewing is allowed by policy.
Cold/unknown users receive zeros; the caller owns one common popularity fallback.

Primary implementation documentation consulted:
https://benfred.github.io/implicit/api/models/cpu/als.html
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
import time
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _training_input(X: sparse.csr_matrix, metadata: Sequence[Mapping[str, Any]]) -> sparse.csr_matrix:
    """Own and canonicalize finite nonnegative CSR training weights."""
    if not sparse.isspmatrix_csr(X):
        raise TypeError("X must be a scipy.sparse.csr_matrix (users x items)")
    if X.ndim != 2 or X.shape[1] != len(metadata):
        raise ValueError("metadata length must equal the item-column count")
    if not all(isinstance(row, Mapping) for row in metadata):
        raise TypeError("Every metadata row must be a mapping")
    # Check BEFORE sum_duplicates so opposite signs cannot silently cancel.
    if not np.isfinite(X.data).all() or (X.data < 0).any():
        raise ValueError("Training weights must be finite and nonnegative")
    result = X.astype(np.float32, copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if not np.isfinite(result.data).all():
        raise ValueError("Training weights overflowed float32")
    return result


def _indices(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(values.dtype, np.integer):
        if values.size == 0:
            return values.astype(np.int64)
        raise TypeError(f"{name} must contain integers")
    return values.astype(np.int64, copy=False)


class _BaseModel:
    """Small explicit lifecycle/shape contract shared by the four models."""

    def _fit_input(self, X: sparse.csr_matrix, metadata: Sequence[Mapping[str, Any]]) -> sparse.csr_matrix:
        data = _training_input(X, metadata)
        self.shape_ = data.shape
        self.warm_users_ = np.diff(data.indptr) > 0
        self.warm_items_ = np.asarray(data.getnnz(axis=0)).ravel() > 0
        self.is_fitted_ = True
        return data

    def _score_input(self, user_indices: np.ndarray, candidate_indices: np.ndarray):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit before score")
        users = _indices(user_indices, "user_indices")
        items = _indices(candidate_indices, "candidate_indices")
        if ((items < 0) | (items >= self.shape_[1])).any():
            raise IndexError("candidate_indices outside the fitted metadata universe")
        valid = (users >= 0) & (users < self.shape_[0])
        rows = np.flatnonzero(valid)
        warm_rows = rows[self.warm_users_[users[rows]]]
        output = np.zeros((len(users), len(items)), dtype=np.float32)
        return users, items, warm_rows, output

    @staticmethod
    def _finite(scores: np.ndarray) -> np.ndarray:
        if not np.isfinite(scores).all():
            raise FloatingPointError("Model produced a non-finite score")
        return scores


class PopularityModel(_BaseModel):
    """Training-only baseline: distinct viewing users or summed positive weight.

    Cold user rows are zeros to keep the same model interface. ``item_scores_``
    is the global reference/fallback vector for the caller, including cold users.
    ``mode='users'`` counts each user-item pair once, independent of repeat count.
    """

    def __init__(self, mode: str = "users"):
        if mode not in {"users", "weight"}:
            raise ValueError("mode must be 'users' or 'weight'")
        self.mode = mode

    def fit(self, X: sparse.csr_matrix, metadata: Sequence[Mapping[str, Any]]):
        data = self._fit_input(X, metadata)
        if self.mode == "users":
            scores = np.asarray(data.getnnz(axis=0)).ravel()
        else:
            scores = np.asarray(data.sum(axis=0)).ravel()
        self.item_scores_ = scores.astype(np.float32)
        self.diagnostics_ = {"training_users": data.shape[0], "training_items": int(self.warm_items_.sum()),
                             "positive_pairs": data.nnz, "mode": self.mode}
        return self

    def score(self, user_indices: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
        _, items, rows, output = self._score_input(user_indices, candidate_indices)
        output[rows] = self.item_scores_[items]
        return self._finite(output)

    def get_config(self):
        return {"model": "popularity", "mode": self.mode}


def _tokens(value: Any) -> list[str]:
    """Unicode word tokens preserve Cyrillic; no English-only stemmer is used."""
    return re.findall(r"[^\W_]+", str(value or "").casefold(), flags=re.UNICODE)


def _names(value: Any) -> list[str]:
    """Keep full person names together instead of matching on first names."""
    if isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        parts = re.split(r"[,;|]+", str(value or ""))
    return ["_".join(words) for part in parts if (words := _tokens(part))]


def _identity(document):
    return document


class ContentModel(_BaseModel):
    """Attribute TF-IDF and weighted cosine profiles fitted on TRAIN items only.

    Only metadata items with nonzero TRAIN interaction columns contribute to the
    vocabulary or IDF. Metadata for all candidates is transformed with that fixed
    mapping. Thus a future item can be scored using previously seen attributes,
    while a future-only word cannot change the training feature statistics.
    """

    def __init__(self, max_features: int = 100_000, interaction_transform: str = "log1p",
                 title_weight: int = 1, category_weight: int = 3, genre_weight: int = 2,
                 people_weight: int = 1, year_weight: int = 1):
        if not isinstance(max_features, int) or max_features <= 0:
            raise ValueError("max_features must be a positive integer")
        if interaction_transform not in {"log1p", "identity", "binary"}:
            raise ValueError("Unknown interaction_transform")
        weights = (title_weight, category_weight, genre_weight, people_weight, year_weight)
        if any(not isinstance(w, int) or w < 0 for w in weights) or not any(weights):
            raise ValueError("Content field weights must be nonnegative integers with a positive sum")
        self.max_features = max_features
        self.interaction_transform = interaction_transform
        self.title_weight, self.category_weight, self.genre_weight = title_weight, category_weight, genre_weight
        self.people_weight, self.year_weight = people_weight, year_weight

    def _document(self, item: Mapping[str, Any]) -> list[str]:
        out = []
        out.extend(["title:" + token for token in _tokens(item.get("title", ""))] * self.title_weight)
        # Exact category/genre labels reduce false matches across field types.
        for name in _names(item.get("category", "")):
            out.extend(["category:" + name] * self.category_weight)
        for key in ("genre_1", "genre_2", "genre_3"):
            for name in _names(item.get(key, "")):
                out.extend(["genre:" + name] * self.genre_weight)
        for key in ("director", "actors"):
            for name in _names(item.get(key, "")):
                out.extend([key + ":" + name] * self.people_weight)
        years = re.findall(r"\b(?:18|19|20)\d{2}\b", str(item.get("year", "")))
        for year in years:
            out.extend(["decade:" + str(int(year) // 10 * 10)] * self.year_weight)
        return out

    def fit(self, X: sparse.csr_matrix, metadata: Sequence[Mapping[str, Any]]):
        start = time.perf_counter()
        data = self._fit_input(X, metadata)
        documents = [self._document(row) for row in metadata]
        fitting_docs = [documents[i] for i in np.flatnonzero(self.warm_items_)]
        self.vectorizer_ = None
        if any(fitting_docs):
            self.vectorizer_ = TfidfVectorizer(analyzer=_identity, lowercase=False,
                max_features=self.max_features, norm="l2", dtype=np.float32,
                sublinear_tf=True, smooth_idf=True)
            self.vectorizer_.fit(fitting_docs)
            self.item_features_ = self.vectorizer_.transform(documents).tocsr()
        else:
            self.item_features_ = sparse.csr_matrix((data.shape[1], 0), dtype=np.float32)
        weights = data.copy()
        if self.interaction_transform == "log1p":
            weights.data = np.log1p(weights.data)
        elif self.interaction_transform == "binary":
            weights.data.fill(1)
        self.user_profiles_ = normalize(weights @ self.item_features_, norm="l2", axis=1, copy=False).tocsr() \
            if self.item_features_.shape[1] else sparse.csr_matrix((data.shape[0], 0), dtype=np.float32)
        self.diagnostics_ = {"training_metadata_items": len(fitting_docs),
            "vocabulary_size": self.item_features_.shape[1], "feature_nnz": self.item_features_.nnz,
            "profile_nnz": self.user_profiles_.nnz, "fit_seconds": time.perf_counter() - start}
        return self

    def score(self, user_indices: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
        users, items, rows, output = self._score_input(user_indices, candidate_indices)
        if rows.size and items.size and self.item_features_.shape[1]:
            output[rows] = (self.user_profiles_[users[rows]] @ self.item_features_[items].T).toarray()
        return self._finite(output)

    def get_config(self):
        return {"model": "content_tfidf", "max_features": self.max_features,
            "interaction_transform": self.interaction_transform, "title_weight": self.title_weight,
            "category_weight": self.category_weight, "genre_weight": self.genre_weight,
            "people_weight": self.people_weight, "year_weight": self.year_weight,
            "fit_metadata_policy": "only items with positive TRAIN interactions", "normalization": "cosine"}


class ImplicitALSModel(_BaseModel):
    """Official implicit CPU ALS with an explicit implicit-feedback confidence.

    Positive observations receive c_ui = 1 + alpha * transform(weight). Unset
    entries follow implicit's P_ui=0, C_ui=1 convention; they are not explicit
    low star ratings. The library's own alpha is fixed to 1 to avoid multiplying
    confidence twice. Dot products remain raw, unmasked ranking scores.
    """

    def __init__(self, factors: int = 48, regularization: float = 0.1,
                 iterations: int = 25, alpha: float = 20.0, seed: int = 20261023,
                 num_threads: int = 2, confidence_transform: str = "log1p", use_cg: bool = True):
        if factors <= 0 or iterations <= 0 or num_threads <= 0:
            raise ValueError("factors, iterations and num_threads must be positive")
        if not np.isfinite([regularization, alpha]).all() or regularization <= 0 or alpha <= 0:
            raise ValueError("regularization and alpha must be finite and positive")
        if confidence_transform not in {"log1p", "identity", "binary"}:
            raise ValueError("Unknown confidence_transform")
        self.factors, self.regularization, self.iterations = int(factors), float(regularization), int(iterations)
        self.alpha, self.seed, self.num_threads = float(alpha), int(seed), int(num_threads)
        self.confidence_transform, self.use_cg = confidence_transform, bool(use_cg)

    def fit(self, X: sparse.csr_matrix, metadata: Sequence[Mapping[str, Any]]):
        from implicit.cpu.als import AlternatingLeastSquares
        from threadpoolctl import threadpool_limits
        start = time.perf_counter()
        data = self._fit_input(X, metadata)
        confidence = data.copy()
        if self.confidence_transform == "log1p":
            confidence.data = np.log1p(confidence.data)
        elif self.confidence_transform == "binary":
            confidence.data.fill(1)
        confidence.data = np.asarray(1 + self.alpha * confidence.data, dtype=np.float32)
        self.model_ = None
        if data.nnz:
            self.model_ = AlternatingLeastSquares(factors=self.factors,
                regularization=self.regularization, alpha=1.0, iterations=self.iterations,
                random_state=self.seed, num_threads=self.num_threads, use_cg=self.use_cg,
                calculate_training_loss=False, dtype=np.float32)
            # Avoid nested OpenBLAS threads inside the explicit ALS thread pool.
            with threadpool_limits(limits=1, user_api="blas"):
                self.model_.fit(confidence, show_progress=False)
            self.model_.user_factors[~self.warm_users_] = 0
            self.model_.item_factors[~self.warm_items_] = 0
            if not np.isfinite(self.model_.user_factors).all() or not np.isfinite(self.model_.item_factors).all():
                raise FloatingPointError("ALS produced non-finite factors")
        self.diagnostics_ = {"positive_pairs": data.nnz, "warm_users": int(self.warm_users_.sum()),
            "warm_items": int(self.warm_items_.sum()), "fit_seconds": time.perf_counter() - start,
            "confidence_min": float(confidence.data.min()) if data.nnz else None,
            "confidence_max": float(confidence.data.max()) if data.nnz else None}
        return self

    def score(self, user_indices: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
        users, items, rows, output = self._score_input(user_indices, candidate_indices)
        if rows.size and items.size and self.model_ is not None:
            output[rows] = self.model_.user_factors[users[rows]] @ self.model_.item_factors[items].T
        return self._finite(output)

    def get_config(self):
        return {"model": "implicit_als_cpu", "factors": self.factors, "regularization": self.regularization,
            "iterations": self.iterations, "alpha": self.alpha, "library_alpha": 1.0,
            "seed": self.seed, "num_threads": self.num_threads, "confidence_transform": self.confidence_transform,
            "positive_confidence": "1 + alpha * transformed positive weight", "use_cg": self.use_cg}


class CoViewPPRModel(_BaseModel):
    """Sparse directed programme co-view PageRank with personal TRAIN restart.

    The binary training history B defines programme co-view support. Each user's
    co-view contribution is divided by their number of distinct watched items,
    and item degrees are cosine-normalized. Graph construction is blockwise to
    avoid materializing the full dense item x item adjacency. The self diagonal
    is removed; each row keeps at most ``neighbors`` strongest outgoing edges.
    After pruning, rows are normalized and dangling mass returns to each user's
    own normalized history. No validation/test behaviour enters graph or restart.
    """

    def __init__(self, alpha: float = 0.7, neighbors: int = 64, tol: float = 1e-6,
                 max_iter: int = 300, graph_block_size: int = 128,
                 score_batch_size: int = 128, restart_transform: str = "log1p"):
        if not np.isfinite(alpha) or not 0 <= alpha < 1:
            raise ValueError("alpha must be finite in [0,1)")
        if not np.isfinite(tol) or tol <= 0 or min(neighbors, max_iter, graph_block_size, score_batch_size) <= 0:
            raise ValueError("tol and size/iteration settings must be positive")
        if restart_transform not in {"log1p", "identity", "binary"}:
            raise ValueError("Unknown restart_transform")
        self.alpha, self.neighbors, self.tol = float(alpha), int(neighbors), float(tol)
        self.max_iter, self.graph_block_size = int(max_iter), int(graph_block_size)
        self.score_batch_size, self.restart_transform = int(score_batch_size), restart_transform

    def fit(self, X: sparse.csr_matrix, metadata: Sequence[Mapping[str, Any]]):
        start_time = time.perf_counter()
        data = self._fit_input(X, metadata)
        binary = data.copy()
        binary.data.fill(1)
        u_degree = np.asarray(binary.sum(axis=1)).ravel()
        i_degree = np.asarray(binary.sum(axis=0)).ravel()
        # Each co-viewing user contributes 1/degree(user), then item cosine scale.
        user_scale = np.zeros_like(u_degree)
        np.divide(1, np.sqrt(u_degree), out=user_scale, where=u_degree > 0)
        item_scale = np.zeros_like(i_degree)
        np.divide(1, np.sqrt(i_degree), out=item_scale, where=i_degree > 0)
        normalized = binary.multiply(user_scale[:, None]).multiply(item_scale[None, :]).tocsr()
        count_items = data.shape[1]
        graph_rows, graph_cols, graph_weights = [], [], []
        for begin in range(0, count_items, self.graph_block_size):
            end = min(begin + self.graph_block_size, count_items)
            block = (normalized[:, begin:end].T @ normalized).tocsr()
            for offset in range(end - begin):
                item = begin + offset
                left, right = block.indptr[offset:offset + 2]
                ids = block.indices[left:right]
                weights = block.data[left:right]
                keep = (ids != item) & (weights > 0)
                ids, weights = ids[keep], weights[keep]
                if ids.size > self.neighbors:
                    # Exact deterministic boundary: equal scores keep lower IDs.
                    threshold = np.partition(weights, weights.size - self.neighbors)[weights.size - self.neighbors]
                    above = np.flatnonzero(weights > threshold)
                    ties = np.flatnonzero(weights == threshold)
                    ties = ties[np.argsort(ids[ties], kind="stable")]
                    take = np.concatenate([above, ties[:self.neighbors - len(above)]])
                    ids, weights = ids[take], weights[take]
                order = np.lexsort((ids, -weights))
                graph_rows.extend([item] * ids.size)
                graph_cols.extend(ids[order].tolist())
                graph_weights.extend(weights[order].tolist())
        graph = sparse.csr_matrix((graph_weights, (graph_rows, graph_cols)),
                                  shape=(count_items, count_items), dtype=np.float64)
        self.transition_ = normalize(graph, norm="l1", axis=1, copy=False).tocsr()
        self.dangling_ = np.diff(self.transition_.indptr) == 0
        restart = data.astype(np.float64)
        if self.restart_transform == "log1p":
            restart.data = np.log1p(restart.data)
        elif self.restart_transform == "binary":
            restart.data.fill(1)
        self.restart_ = normalize(restart, norm="l1", axis=1, copy=False).tocsr()
        self.last_convergence_ = []
        row_sums = np.asarray(self.transition_.sum(axis=1)).ravel()
        self.diagnostics_ = {"nodes": count_items, "directed_edges_after_pruning": self.transition_.nnz,
            "dangling_items": int(self.dangling_.sum()),
            "max_row_sum_error": float(np.max(np.abs(row_sums[~self.dangling_] - 1))) if (~self.dangling_).any() else 0.0,
            "graph_fit_seconds": time.perf_counter() - start_time}
        return self

    def score(self, user_indices: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
        users, items, rows, output = self._score_input(user_indices, candidate_indices)
        self.last_convergence_ = []
        for start in range(0, rows.size, self.score_batch_size):
            output_rows = rows[start:start + self.score_batch_size]
            ids = users[output_rows]
            v = self.restart_[ids].toarray()
            p = v.copy()
            active = np.ones(len(ids), dtype=bool)
            iterations = np.zeros(len(ids), dtype=np.int64)
            residuals = np.full(len(ids), np.inf)
            # Freeze each user's first converged iterate, independent of batches.
            for iteration in range(1, self.max_iter + 1):
                active_ids = np.flatnonzero(active)
                if not active_ids.size:
                    break
                current = p[active_ids]
                dangling_mass = current[:, self.dangling_].sum(axis=1, keepdims=True)
                incoming = (self.transition_.T @ current.T).T
                next_p = self.alpha * (incoming + dangling_mass * v[active_ids]) + (1 - self.alpha) * v[active_ids]
                change = np.abs(next_p - current).sum(axis=1)
                p[active_ids] = next_p
                residuals[active_ids] = change
                iterations[active_ids] = iteration
                active[active_ids[change < self.tol]] = False
            masses = p.sum(axis=1)
            if (p < -1e-12).any() or not np.allclose(masses, 1, atol=1e-9, rtol=0):
                raise FloatingPointError("PPR failed nonnegativity or probability conservation")
            for k, uid in enumerate(ids):
                self.last_convergence_.append({"user_index": int(uid), "iterations": int(iterations[k]),
                    "residual": float(residuals[k]), "converged": bool(residuals[k] < self.tol),
                    "mass": float(masses[k])})
            output[output_rows] = p[:, items].astype(np.float32)
        return self._finite(output)

    def get_config(self):
        return {"model": "programme_coview_ppr", "alpha": self.alpha, "neighbors": self.neighbors,
            "tol": self.tol, "max_iter": self.max_iter, "graph_block_size": self.graph_block_size,
            "score_batch_size": self.score_batch_size, "restart_transform": self.restart_transform,
            "graph_weights": "binary co-view weighted by inverse user degree and item cosine degree",
            "dangling_policy": "return dangling mass to the same personal training restart"}
