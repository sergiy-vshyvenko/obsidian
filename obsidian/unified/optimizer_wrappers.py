"""Optimizer wrapper adapters for unified benchmarking interface.

Each wrapper adapts an optimizer backend to a common interface::

    setup(param_bounds, objectives, minimize)  ->  None
    initialize(n_init, seed)                   ->  pd.DataFrame
    fit(X, y)                                  ->  None
    suggest(n)                                 ->  pd.DataFrame
    is_available()                             ->  bool

``objectives`` is a list of ``(name, minimize)`` tuples — one per response.
For single-objective use, pass ``objectives=None`` and ``minimize=bool``
(backward-compatible with the benchmark runner).

All wrappers are purely stateful objects; create a new instance per benchmark
run to avoid state leaking between experiments.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ParamBounds = Dict[str, Tuple[float, float]]
Objectives = List[Tuple[str, bool]]   # [(name, minimize), ...]


def _resolve_objectives(objectives: Optional[Objectives], minimize: bool) -> Objectives:
    """Return objectives list, falling back to single-objective with name 'y'."""
    if objectives is not None:
        return objectives
    return [("y", minimize)]


class BaseOptimizerWrapper(ABC):
    """Common interface for all optimizer backends."""

    name: str = "base"
    label: str = "Base"

    @abstractmethod
    def setup(
        self,
        param_bounds: ParamBounds,
        objectives: Optional[Objectives] = None,
        minimize: bool = False,
    ) -> None:
        """Configure the optimizer with the parameter space and objectives.

        Parameters
        ----------
        param_bounds:
            ``{param_name: (min, max)}`` for each continuous parameter.
        objectives:
            List of ``(response_name, minimize)`` tuples.  Pass ``None``
            for single-objective, in which case ``minimize`` is used.
        minimize:
            Used only when ``objectives`` is ``None`` (backward compat).
        """

    @abstractmethod
    def initialize(self, n_init: int, seed: int = 0) -> pd.DataFrame:
        """Generate initial DoE points. Returns DataFrame with param_bounds keys."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> None:
        """Fit the surrogate model to observed data.

        ``y`` may be a Series (single objective) or DataFrame (multi-objective).
        """

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

    def setup(
        self,
        param_bounds: ParamBounds,
        objectives: Optional[Objectives] = None,
        minimize: bool = False,
    ) -> None:
        from obsidian.parameters import ParamSpace, Param_Continuous, Target
        from obsidian.campaign import Campaign

        self._objectives = _resolve_objectives(objectives, minimize)
        params = [Param_Continuous(k, v[0], v[1]) for k, v in param_bounds.items()]
        self._X_space = ParamSpace(params)
        targets = [Target(name, aim="min" if min_ else "max") for name, min_ in self._objectives]
        self._campaign = Campaign(self._X_space, targets if len(targets) > 1 else targets[0])

    def initialize(self, n_init: int, seed: int = 0) -> pd.DataFrame:
        from obsidian.experiment import ExpDesigner
        designer = ExpDesigner(self._X_space, seed=seed)
        return designer.initialize(n_init, "LHS")

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> None:
        df = X.copy()
        if isinstance(y, pd.DataFrame):
            for col in y.columns:
                df[col] = y[col].values
        else:
            df[self._objectives[0][0]] = y.values
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
    """Wrapper for BoFire (single- and multi-objective BO).

    Single-objective uses ``SoboStrategy``; multi-objective uses
    ``qNEHVIStrategy`` (noisy expected hypervolume improvement).
    """

    name = "bofire"
    label = "BoFire (BoTorch)"

    def setup(
        self,
        param_bounds: ParamBounds,
        objectives: Optional[Objectives] = None,
        minimize: bool = False,
    ) -> None:
        from bofire.data_models.domain.api import Domain, Inputs, Outputs
        from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
        from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective

        self._objectives = _resolve_objectives(objectives, minimize)
        self._param_names = list(param_bounds.keys())
        self._experiments: Optional[pd.DataFrame] = None

        inputs = Inputs(features=[
            ContinuousInput(key=k, bounds=(float(v[0]), float(v[1])))
            for k, v in param_bounds.items()
        ])
        output_features = []
        for name, min_ in self._objectives:
            obj = MinimizeObjective(w=1) if min_ else MaximizeObjective(w=1)
            output_features.append(ContinuousOutput(key=name, objective=obj))
        outputs = Outputs(features=output_features)
        self._domain = Domain(inputs=inputs, outputs=outputs)

    def initialize(self, n_init: int, seed: int = 0) -> pd.DataFrame:
        from bofire.data_models.strategies.api import RandomStrategy as RandomDM
        from bofire.strategies.api import RandomStrategy

        dm = RandomDM(domain=self._domain, seed=seed)
        strategy = RandomStrategy(data_model=dm)
        candidates = strategy.ask(n_init)
        return candidates[self._param_names].reset_index(drop=True)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> None:
        df = X[self._param_names].copy()
        if isinstance(y, pd.DataFrame):
            for col in y.columns:
                df[col] = y[col].values
                df[f"valid_{col}"] = 1
        else:
            name = self._objectives[0][0]
            df[name] = y.values
            df[f"valid_{name}"] = 1
        self._experiments = df

    def suggest(self, n: int = 1) -> Optional[pd.DataFrame]:
        if self._experiments is None:
            return None

        if len(self._objectives) > 1:
            from bofire.data_models.strategies.api import qNEHVIStrategy as qNEHVIDM
            from bofire.strategies.api import qNEHVIStrategy
            dm = qNEHVIDM(domain=self._domain)
            strategy = qNEHVIStrategy(data_model=dm)
        else:
            from bofire.data_models.strategies.api import SoboStrategy as SoboDM
            from bofire.strategies.api import SoboStrategy
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
    """Wrapper for BayBe (single- and multi-objective BO).

    Single-objective uses ``NumericalTarget.to_objective()``;
    multi-objective uses ``ParetoObjective`` with ``BotorchRecommender``.
    """

    name = "baybe"
    label = "BayBe (BoTorch)"

    def setup(
        self,
        param_bounds: ParamBounds,
        objectives: Optional[Objectives] = None,
        minimize: bool = False,
    ) -> None:
        self._objectives = _resolve_objectives(objectives, minimize)
        self._param_bounds = param_bounds
        self._param_names = list(param_bounds.keys())
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

        targets = [NumericalTarget(name=name, minimize=min_) for name, min_ in self._objectives]
        if len(targets) == 1:
            self._objective = targets[0].to_objective()
        else:
            from baybe.objectives import ParetoObjective
            self._objective = ParetoObjective(targets=targets)

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

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> None:
        df = X[self._param_names].copy()
        if isinstance(y, pd.DataFrame):
            for col in y.columns:
                df[col] = y[col].values
        else:
            df[self._objectives[0][0]] = y.values
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
    """EDBO+ wrapper for discrete-scope BO benchmarking (single-objective only).

    EDBO+ is designed for reaction optimization over a predefined scope.
    This wrapper materialises a fine grid over the continuous parameter space
    as the 'reaction scope' and uses EDBO+ to rank candidates each iteration.
    """

    name = "edboplus"
    label = "EDBO+ (BoTorch)"

    def setup(
        self,
        param_bounds: ParamBounds,
        objectives: Optional[Objectives] = None,
        minimize: bool = False,
    ) -> None:
        import os
        import tempfile

        self._objectives = _resolve_objectives(objectives, minimize)
        self._param_names = list(param_bounds.keys())
        self._minimize = self._objectives[0][1]
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

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> None:
        from scipy.spatial.distance import cdist

        y_series = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y

        scope_df = self._scope_df.copy()
        scope_df["y"] = "PENDING"
        X_np = X[self._param_names].values
        scope_np = scope_df[self._param_names].values
        for xi, yi in zip(X_np, y_series.values):
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
