from typing import List, Union, Sequence, Optional
import time
import logging
import numpy as np
import pandas as pd
import pynndescent
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array, check_is_fitted


logger = logging.getLogger(__name__)


class HdLOF(BaseEstimator, OutlierMixin):
  
    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        metric: Union[str, callable] = "euclidean",
        random_state: Optional[int] = None,
        second_layer=False,
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.random_state = random_state
        self.second_layer = second_layer #HdLOF-2L

        # Attributes set by fit()
        self._fitting_time_: Optional[float] = None
        self._index_: Optional[pynndescent.NNDescent] = None
        self._relative_density_: Optional[np.ndarray] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> "HdLOF":

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_arr = check_array(X_arr, ensure_min_samples=2, estimator=self)

        # Build NN-Descent index
        start = time.time()
        self._index_ = pynndescent.NNDescent(
            X_arr,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            random_state=self.random_state,
            n_jobs=-1,
        )
        # Retrieve neighbors and distances
        neighbors, distances = self._index_.neighbor_graph

        # k-distance and reachability distance
        k_dist = distances[:, -1][:, None]  # shape (n_samples, 1)
        reach_dists = np.maximum(distances, k_dist)

        # local density = 1 / (average reachability distance)
        avg_reach = reach_dists.mean(axis=1) + 1e-12
        density = 1.0 / avg_reach

        # local-density of neighbors
        # vectorized: for each point i, average density over its neighbor set
        local_dens = np.take(density, neighbors).mean(axis=1)

        # relative density
        self._relative_density_ = density / (local_dens + 1e-12)
        if self.second_layer:
            local_dens_new = np.take(self._relative_density_, neighbors).mean(axis=1)
            self._relative_density_ = density / (local_dens_new + 1e-12)


        self._fitting_time_ = time.time() - start
        return self
        


    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
     
        check_is_fitted(self, ["_index_", "_relative_density_"])
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_arr = check_array(X_arr, estimator=self)

        # query neighbors in the fitted index
        neighbors, distances = self._index_.query(X_arr, k=self.n_neighbors)

        # compute density for new points
        avg_dist = distances.mean(axis=1) + 1e-12
        density_new = 1.0 / avg_dist

        # their neighbors’ densities (using fitted relative densities)
        local_dens_new = np.take(self._relative_density_, neighbors).mean(axis=1)

        # relative density score
        rel_density_new = density_new / (local_dens_new + 1e-12)
        return -(self._relative_density_)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict 1 for inliers and -1 for outliers.
        """
        scores = self.decision_function(X)
        # threshold by contamination
        n_samples = scores.shape[0]
        n_outliers = int(np.floor(self.contamination * n_samples))
        # lowest scores (most anomalous) become outliers
        threshold = np.partition(scores, n_outliers)[n_outliers]
        labels = np.ones(n_samples, dtype=int)
        labels[scores <= threshold] = -1
        return labels

    def fit_time(self) -> float:
        """
        Time (in seconds) to fit the model.
        """
        if self._fitting_time_ is None:
            raise ValueError("Fit must be called before retrieving fit_time.")
        return self._fitting_time_



class HdLOFEnsemble(BaseEstimator, OutlierMixin):
    """Multi‑scale HdLOF with statistical score fusion.

    One ANN graph at k_max is reused; reachability/density recalculated per k
    with correct k‑distance.  Aggregators:
        * 'max'        – strongest evidence (min LOF).
        * 'rank_sum'   – sum of ranks across k.
        * 'soft'       – weighted z‑score (w_k ∝ 1/k).
        * 'fisher'     – Fisher's method over per‑k empirical p‑values (default).
    """

    def __init__(
        self,
        k_grid: Sequence[int] = (5, 10, 15),
        agg: str = 'fisher',
        contamination: float = 0.1,
        metric: str = 'euclidean',
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ) -> None:
        self.k_grid = sorted(k_grid)
        self.agg = agg
        self.contamination = contamination
        self.metric = metric
        self.random_state = random_state
        self.n_jobs = n_jobs


    def _lof_matrix(self, D: np.ndarray, N: np.ndarray) -> np.ndarray:
        """
        Fully-vectorised LOF for all k in ``self.k_grid``.
        Returns ``(n_samples, len(k_grid))`` float32 matrix.
        """
        n, k_max          = D.shape
        k_grid            = np.asarray(self.k_grid, dtype=np.int32)          # (m,)
        m                 = k_grid.size
    
        # ---------------- 1) reachability distances  -----------------
        # RD_k(x, o) = max{ d(x, o), d_k(o) }  for every neighbour rank k
        reach             = np.maximum(D, D[N, np.arange(k_max)])            # (n, k_max)
    
        # ---------------- 2) cumulative sums of reachabilities --------
        reach_cum         = np.cumsum(reach, axis=1, dtype=np.float32)       # (n, k_max)
    
        # ---------------- 3) local reachability density ρ_k(x) --------
        avg_reach         = reach_cum[:, k_grid - 1] / k_grid                # (n, m)
        rho               = 1.0 / (avg_reach + 1e-12)                        # (n, m)
    
        # ---------------- 4) neighbour densities ρ_k(o), o ∈ N_k(x) ----
        # gather ρ_k for every neighbour, then prefix-sum so we can
        # read Σρ over the first k neighbours in O(1)
        rho_neigh_all     = rho[N]                                           # (n, k_max, m)
        rho_neigh_cum     = np.cumsum(rho_neigh_all, axis=1, dtype=np.float32)
    
        # mean neighbour density for each k in one shot
        rho_n             = rho_neigh_cum[:, k_grid - 1, np.arange(m)] / k_grid  # (n, m)
        
    
        # ---------------- 5) LOF_k(x) = ρ̄_n / ρ(x) --------------------
        lof               = rho_n / rho

       
        return lof.astype(np.float32)



    # -------------------------------------------------- aggregation
    def _aggregate(self, lof_mat: np.ndarray) -> np.ndarray:
        n, m = lof_mat.shape
        if self.agg == 'max':
            score = lof_mat.min(1)
        elif self.agg == 'rank_sum':
            ranks = lof_mat.argsort(0).argsort(0) + 1
            score = ranks.sum(1)
        elif self.agg == 'soft':
            scaled = (lof_mat.max(0) - lof_mat) / (lof_mat.ptp(0) + 1e-12)
            w = 1 / np.array(self.k_grid, dtype=np.float32)
            w /= w.sum()
            score = -(scaled * w).sum(1)
        elif self.agg == 'fisher':
            ranks = lof_mat.argsort(0).argsort(0) + 1  # 1..n lower LOF --> smaller rank
            pvals = (n + 1 - ranks) / (n + 1)          # empirical p‑values
            chi2 = -2.0 * np.log(pvals + 1e-12).sum(1) # Fisher statistic
            score = chi2                               # larger chi2 --> more anomalous
            return -score.astype(np.float32)
        else:
            raise ValueError("agg must be 'max', 'rank_sum', 'soft', or 'fisher'")
        return -score.astype(np.float32)

    # -------------------------------------------------- fit
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_arr = check_array(X_arr).astype(np.float32, copy=False)
        kmax = self.k_grid[-1]

        index = pynndescent.NNDescent(
            X_arr,
            n_neighbors=kmax,
            metric=self.metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        N, D = index.neighbor_graph  # sorted rows
        lof_mat = self._lof_matrix(D, N)
        self._scores_ = self._aggregate(lof_mat)
        self._index_ = index
        return self

    # -------------------------------------------------- predict/score
    def decision_function(self, X):
        return -self._scores_

    def predict(self, X):
        scores = self.decision_function(X)
        n_out = int(np.floor(self.contamination * len(scores)))
        thr = np.partition(scores, n_out)[n_out]
        labels = np.ones(len(scores), dtype=int)
        labels[scores <= thr] = -1
        return labels

