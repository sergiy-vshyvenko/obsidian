"""Optimizer wrapper adapters for unified benchmarking interface.

Each wrapper adapts an optimizer backend to a common interface::

    setup(param_bounds, minimize)  ->  None
    initialize(n_init, seed)       ->  pd.DataFrame
    fit(X, y)                      ->  None
    suggest(n)                     ->  pd.DataFrame
    is_available()                 ->  bool

All wrappers are purely stateful objects; create a new instance per benchmark
run to avoid state leaking between experiments.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ParamBounds = Dict[str, Tuple[float, float]]


class BaseOptimizerWrapper(ABC):
    """Common interface for all optimizer backends."""

    name: str = "base"
    label: str = "Base"

    @abstractmethod
    def setup(self, param_bounds: ParamBounds, minimize: bool = False) -> None:
        """Configure the optimizer with the parameter space."""

    @abstractmethod
    def initialize(self, n_init: int, seed: int = 0) -> pd.DataFrame:
        """Generate initial DoE points. Returns DataFrame with param_bounds keys."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the surrogate model to observed data."""

    @abstractmethod
    def suggest(self, n: int = 1) -> Optional[pd.DataFrame]:
        """Suggest next experiment(s). Returns DataFrame with param_bounds keys."""

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Return True if the backend library is importable."""


# ---------------------------------------------------------------------------
# Obsidian wrapper
# ---------------------------------------------------------------------------

class ObsidianWrapper(BaseOptimizerWrapper):
    """Wrapper for obsidian's BayesianOptimizer via Campaign."""

    name = "obsidian"
    label = "Obsidian (BoTorch)"

    def setup(self, param_bounds: ParamBounds, minimize: bool = False) -> None:
        from obsidian.parameters import ParamSpace, Param_Continuous, Target
        from obsidian.campaign import Campaign

        params = [Param_Continuous(k, v[0], v[1]) for k, v in param_bounds.items()]
        self._X_space = ParamSpace(params)
        self._target_name = "y"
        self._minimize = minimize
        target = Target(self._target_name, aim="min" if minimize else "max")
        self._campaign = Campaign(self._X_space, target)

    def initialize(self, n_init: int, seed: int = 0) -> pd.DataFrame:
        from obsidian.experiment import ExpDesigner
        designer = ExpDesigner(self._X_space, seed=seed)
        return designer.initialize(n_init, "LHS")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        df = X.copy()
        df[self._target_name] = y.values
        self._campaign.clear_data()
        self._campaign.add_data(df)
        self._campaign.fit()

    def suggest(self, n: int = 1) -> Optional[pd.DataFrame]:
        result = self._campaign.suggest(n_suggest=n)
        if result is None:
            return None
        X_suggest, _ = result
        return X_suggest[list(self._X_space.X_names)].reset_index(drop=True)

    @staticmethod
    def is_available() -> bool:
        try:
            import obsidian  # noqa: F401
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# BoFire wrapper
# ---------------------------------------------------------------------------

class BofireWrapper(BaseOptimizerWrapper):
    """Wrapper for BoFire's SoboStrategy (single-objective BO)."""

    name = "bofire"
    label = "BoFire (BoTorch)"

    def setup(self, param_bounds: ParamBounds, minimize: bool = False) -> None:
        from bofire.data_models.domain.api import Domain, Inputs, Outputs
        from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
        from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective

        inputs = Inputs(features=[
            ContinuousInput(key=k, bounds=(float(v[0]), float(v[1])))
            for k, v in param_bounds.items()
        ])
        obj = MinimizeObjective(w=1) if minimize else MaximizeObjective(w=1)
        outputs = Outputs(features=[ContinuousOutput(key="y", objective=obj)])
        self._domain = Domain(inputs=inputs, outputs=outputs)
        self._param_names = list(param_bounds.keys())
        self._minimize = minimize
        self._experiments: Optional[pd.DataFrame] = None

    def initialize(self, n_init: int, seed: int = 0) -> pd.DataFrame:
        from bofire.data_models.strategies.api import RandomStrategy as RandomDM
        from bofire.strategies.api import RandomStrategy

        dm = RandomDM(domain=self._domain, seed=seed)
        strategy = RandomStrategy(data_model=dm)
        candidates = strategy.ask(n_init)
        return candidates[self._param_names].reset_index(drop=True)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        df = X[self._param_names].copy()
        df["y"] = y.values
        df["valid_y"] = 1
        self._experiments = df

    def suggest(self, n: int = 1) -> Optional[pd.DataFrame]:
        from bofire.data_models.strategies.api import SoboStrategy as SoboDM
        from bofire.strategies.api import SoboStrategy

        if self._experiments is None:
            return None
        dm = SoboDM(domain=self._domain)
        strategy = SoboStrategy(data_model=dm)
        strategy.tell(self._experiments)
        candidates = strategy.ask(n)
        return candidates[self._param_names].reset_index(drop=True)

    @staticmethod
    def is_available() -> bool:
        try:
            import bofire  # noqa: F401
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# BayBe wrapper
# ---------------------------------------------------------------------------

class BaybeWrapper(BaseOptimizerWrapper):
    """Wrapper for BayBe's BotorchRecommender (stateless recommend API)."""

    name = "baybe"
    label = "BayBe (BoTorch)"

    def setup(self, param_bounds: ParamBounds, minimize: bool = False) -> None:
        self._param_bounds = param_bounds
        self._param_names = list(param_bounds.keys())
        self._minimize = minimize
        self._measurements: Optional[pd.DataFrame] = None
        self._searchspace = None
        self._objective = None
        self._recommender = None

    def _build_objects(self) -> None:
        from baybe.parameters import NumericalContinuousParameter
        from baybe.recommenders import BotorchRecommender
        from baybe.searchspace import SearchSpace
        from baybe.targets import NumericalTarget

        params = [
            NumericalContinuousParameter(name=k, bounds=(float(v[0]), float(v[1])))
            for k, v in self._param_bounds.items()
        ]
        self._searchspace = SearchSpace.from_product(parameters=params)
        self._objective = NumericalTarget(name="y", minimize=self._minimize).to_objective()
        self._recommender = BotorchRecommender()

    def initialize(self, n_init: int, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        data = {}
        for k, (lo, hi) in self._param_bounds.items():
            intervals = np.linspace(lo, hi, n_init + 1)
            samples = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(samples)
            data[k] = samples
        return pd.DataFrame(data)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        df = X[self._param_names].astype(float).copy()
        df["y"] = y.values
        self._measurements = df

    def suggest(self, n: int = 1) -> Optional[pd.DataFrame]:
        if self._searchspace is None:
            self._build_objects()
        candidates = self._recommender.recommend(
            batch_size=n,
            searchspace=self._searchspace,
            objective=self._objective,
            measurements=self._measurements,
        )
        return candidates[self._param_names].reset_index(drop=True)

    @staticmethod
    def is_available() -> bool:
        try:
            import baybe  # noqa: F401
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# EDBO+ wrapper
# ---------------------------------------------------------------------------

class EdboplusWrapper(BaseOptimizerWrapper):
    """EDBO+ wrapper for discrete-scope BO benchmarking.

    EDBO+ is designed for reaction optimization over a predefined scope.
    This wrapper materialises a fine grid over the continuous parameter space
    as the 'reaction scope' and uses EDBO+ to rank candidates each iteration.
    """

    name = "edboplus"
    label = "EDBO+ (BoTorch)"

    def setup(self, param_bounds: ParamBounds, minimize: bool = False) -> None:
        import os
        import tempfile

        self._param_bounds = param_bounds
        self._param_names = list(param_bounds.keys())
        self._minimize = minimize
        self._tmpdir = tempfile.mkdtemp(prefix="edboplus_")
        self._csv_path = os.path.join(self._tmpdir, "scope.csv")

        n_dims = len(param_bounds)
        if n_dims <= 2:
            n_grid = 30
            grids = [np.linspace(v[0], v[1], n_grid) for v in param_bounds.values()]
            from itertools import product as iproduct
            pts = list(iproduct(*grids))
            self._scope_df = pd.DataFrame(pts, columns=self._param_names)
        else:
            rng = np.random.default_rng(42)
            data = {k: rng.uniform(v[0], v[1], 1000) for k, v in param_bounds.items()}
            self._scope_df = pd.DataFrame(data)

        self._scope_df.to_csv(self._csv_path, index=False)

    def initialize(self, n_init: int, seed: int = 0) -> pd.DataFrame:
        from edbo.plus.optimizer_botorch import EDBOplus

        self._scope_df.to_csv(self._csv_path, index=False)
        optimizer = EDBOplus()
        result = optimizer.run(
            objectives=["y"],
            objective_mode=["min" if self._minimize else "max"],
            directory=self._tmpdir,
            filename="scope.csv",
            batch=n_init,
            seed=seed,
        )
        top = result[result["priority"] == 1].head(n_init)
        return top[self._param_names].reset_index(drop=True)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from scipy.spatial.distance import cdist

        scope_df = self._scope_df.copy()
        scope_df["y"] = "PENDING"
        X_np = X[self._param_names].values
        scope_np = scope_df[self._param_names].values
        for xi, yi in zip(X_np, y.values):
            d = cdist([xi], scope_np, metric="cityblock")
            idx = int(np.argmin(d))
            scope_df.at[idx, "y"] = float(yi)
        scope_df.to_csv(self._csv_path, index=False)

    def suggest(self, n: int = 1) -> Optional[pd.DataFrame]:
        from edbo.plus.optimizer_botorch import EDBOplus

        optimizer = EDBOplus()
        result = optimizer.run(
            objectives=["y"],
            objective_mode=["min" if self._minimize else "max"],
            directory=self._tmpdir,
            filename="scope.csv",
            batch=n,
        )
        top = result[result["priority"] == 1].head(n)
        return top[self._param_names].reset_index(drop=True)

    @staticmethod
    def is_available() -> bool:
        try:
            from edbo.plus.optimizer_botorch import EDBOplus  # noqa: F401
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

ALL_WRAPPERS: List[type] = [
    ObsidianWrapper,
    BofireWrapper,
    BaybeWrapper,
    EdboplusWrapper,
]


def get_available_optimizers() -> List[type]:
    """Return wrapper classes whose backend library is importable."""
    return [w for w in ALL_WRAPPERS if w.is_available()]
