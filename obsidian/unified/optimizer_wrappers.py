"""Optimizer wrapper adapters for unified interface.

Each wrapper adapts an optimizer backend to a common interface::

    setup(param_bounds, param_categories, objectives, minimize)  ->  None
    initialize(n_init, seed)                                      ->  pd.DataFrame
    fit(X, y)                                                     ->  None
    suggest(n)                                                    ->  pd.DataFrame
    is_available()                                                ->  bool

Parameters
----------
param_bounds : dict {name: (min, max)}
    Continuous parameters.
param_categories : dict {name: [cat1, cat2, ...]}, optional
    Categorical / ordinal parameters.
objectives : list of (name, minimize) tuples, optional
    Multi-objective targets.  Pass None + minimize=bool for single-objective.
minimize : bool
    Used only when objectives is None (backward compat with benchmark runner).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ParamBounds = Dict[str, Tuple[float, float]]
ParamCategories = Dict[str, List[str]]
Objectives = List[Tuple[str, bool]]   # [(name, minimize), ...]


def _resolve_objectives(objectives: Optional[Objectives], minimize: bool, y_name: str = "y") -> Objectives:
    if objectives is not None:
        return objectives
    return [(y_name, minimize)]


class BaseOptimizerWrapper(ABC):
    """Common interface for all optimizer backends."""

    name: str = "base"
    label: str = "Base"

    @abstractmethod
    def setup(
        self,
        param_bounds: ParamBounds,
        param_categories: Optional[ParamCategories] = None,
        objectives: Optional[Objectives] = None,
        minimize: bool = False,
    ) -> None:
        """Configure the optimizer with the parameter space and objectives."""

    @abstractmethod
    def initialize(self, n_init: int, seed: int = 0) -> pd.DataFrame:
        """Generate initial DoE points."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> None:
        """Fit the surrogate model. y may be a Series (SOO) or DataFrame (MOO)."""

    @abstractmethod
    def suggest(self, n: int = 1) -> Optional[pd.DataFrame]:
        """Suggest next experiment(s)."""

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
        param_categories: Optional[ParamCategories] = None,
        objectives: Optional[Objectives] = None,
        minimize: bool = False,
    ) -> None:
        from obsidian.parameters import ParamSpace, Param_Continuous, Param_Categorical, Target
        from obsidian.campaign import Campaign

        self._objectives = _resolve_objectives(objectives, minimize)
        params = [Param_Continuous(k, v[0], v[1]) for k, v in param_bounds.items()]
        for k, cats in (param_categories or {}).items():
            params.append(Param_Categorical(k, cats))
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
        result = self._campaign.suggest(m_batch=n)
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

    Supports continuous and categorical parameters.
    Single-objective uses SoboStrategy; multi-objective uses qNEHVIStrategy.
    """

    name = "bofire"
    label = "BoFire (BoTorch)"

    def setup(
        self,
        param_bounds: ParamBounds,
        param_categories: Optional[ParamCategories] = None,
        objectives: Optional[Objectives] = None,
        minimize: bool = False,
    ) -> None:
        from bofire.data_models.domain.api import Domain, Inputs, Outputs
        from bofire.data_models.features.api import (
            ContinuousInput, CategoricalInput, ContinuousOutput,
        )
        from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective

        self._objectives = _resolve_objectives(objectives, minimize)
        self._param_names = list(param_bounds.keys()) + list((param_categories or {}).keys())
        self._experiments: Optional[pd.DataFrame] = None

        input_features = [
            ContinuousInput(key=k, bounds=(float(v[0]), float(v[1])))
            for k, v in param_bounds.items()
        ]
        for k, cats in (param_categories or {}).items():
            input_features.append(CategoricalInput(key=k, categories=list(cats)))

        output_features = []
        for name, min_ in self._objectives:
            obj = MinimizeObjective(w=1) if min_ else MaximizeObjective(w=1)
            output_features.append(ContinuousOutput(key=name, objective=obj))

        self._domain = Domain(
            inputs=Inputs(features=input_features),
            outputs=Outputs(features=output_features),
        )

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
            from bofire.data_models.strategies.api import MoboStrategy as MoboDM
            from bofire.strategies.api import MoboStrategy
            dm = MoboDM(domain=self._domain)
            strategy = MoboStrategy(data_model=dm)
        else:
            from bofire.data_models.strategies.api import SoboStrategy as SoboDM
            from bofire.strategies.predictives.sobo import SoboStrategy
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

    Supports continuous and categorical parameters.
    Single-objective uses NumericalTarget; multi-objective uses ParetoObjective.
    Uses baybe.Campaign so the fitted surrogate is accessible via predict().
    """

    name = "baybe"
    label = "BayBe (BoTorch)"

    def setup(
        self,
        param_bounds: ParamBounds,
        param_categories: Optional[ParamCategories] = None,
        objectives: Optional[Objectives] = None,
        minimize: bool = False,
    ) -> None:
        from baybe.parameters import NumericalContinuousParameter, CategoricalParameter
        from baybe.searchspace import SearchSpace
        from baybe.targets import NumericalTarget

        self._objectives = _resolve_objectives(objectives, minimize)
        self._param_bounds = param_bounds
        self._param_categories = param_categories or {}
        self._param_names = list(param_bounds.keys()) + list(self._param_categories.keys())

        params = [
            NumericalContinuousParameter(name=k, bounds=(float(v[0]), float(v[1])))
            for k, v in param_bounds.items()
        ]
        for k, cats in self._param_categories.items():
            params.append(CategoricalParameter(name=k, values=list(cats)))

        self._searchspace = SearchSpace.from_product(parameters=params)

        targets = [NumericalTarget(name=name, minimize=min_) for name, min_ in self._objectives]
        if len(targets) == 1:
            self._baybe_objective = targets[0].to_objective()
        else:
            from baybe.objectives import ParetoObjective
            self._baybe_objective = ParetoObjective(targets=targets)

        self._campaign = None

    def initialize(self, n_init: int, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        data = {}
        for k, (lo, hi) in self._param_bounds.items():
            intervals = np.linspace(lo, hi, n_init + 1)
            samples = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(samples)
            data[k] = samples
        for k, cats in self._param_categories.items():
            data[k] = rng.choice(cats, size=n_init)
        return pd.DataFrame(data)

    def _build_campaign(self) -> None:
        from baybe.recommenders import BotorchRecommender
        from baybe import Campaign as BaybeCampaign

        self._campaign = BaybeCampaign(
            searchspace=self._searchspace,
            objective=self._baybe_objective,
            recommender=BotorchRecommender(),
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> None:
        df = X[self._param_names].copy()
        float_cols = list(self._param_bounds.keys())
        df[float_cols] = df[float_cols].astype(float)
        if isinstance(y, pd.DataFrame):
            for col in y.columns:
                df[col] = y[col].values
        else:
            df[self._objectives[0][0]] = y.values
        # Rebuild campaign each fit so measurements don't accumulate across calls
        self._build_campaign()
        self._campaign.add_measurements(df)

    def suggest(self, n: int = 1) -> Optional[pd.DataFrame]:
        if self._campaign is None:
            return None
        candidates = self._campaign.recommend(batch_size=n)
        return candidates[self._param_names].reset_index(drop=True)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return posterior mean and std for each objective at the given X points."""
        if self._campaign is None:
            return pd.DataFrame()
        surrogate = self._campaign.get_surrogate()
        cols = X[self._param_names].copy()
        cols[list(self._param_bounds.keys())] = cols[list(self._param_bounds.keys())].astype(float)
        stats = surrogate.posterior_stats(cols, stats=["mean", "std"])
        return stats.reset_index(drop=True)

    @staticmethod
    def is_available() -> bool:
        try:
            import baybe  # noqa: F401
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# EDBO+ wrapper (continuous-only, single-objective, discrete grid)
# ---------------------------------------------------------------------------

class EdboplusWrapper(BaseOptimizerWrapper):
    """EDBO+ wrapper for discrete-scope BO benchmarking."""

    name = "edboplus"
    label = "EDBO+ (BoTorch)"

    def setup(
        self,
        param_bounds: ParamBounds,
        param_categories: Optional[ParamCategories] = None,
        objectives: Optional[Objectives] = None,
        minimize: bool = False,
    ) -> None:
        import os, tempfile

        self._objectives = _resolve_objectives(objectives, minimize)
        self._param_names = list(param_bounds.keys())
        self._minimize = self._objectives[0][1]
        self._tmpdir = tempfile.mkdtemp(prefix="edboplus_")
        self._csv_path = os.path.join(self._tmpdir, "scope.csv")

        n_dims = len(param_bounds)
        if n_dims <= 2:
            grids = [np.linspace(v[0], v[1], 30) for v in param_bounds.values()]
            from itertools import product as iproduct
            self._scope_df = pd.DataFrame(list(iproduct(*grids)), columns=self._param_names)
        else:
            rng = np.random.default_rng(42)
            self._scope_df = pd.DataFrame(
                {k: rng.uniform(v[0], v[1], 1000) for k, v in param_bounds.items()}
            )
        self._scope_df.to_csv(self._csv_path, index=False)

    def initialize(self, n_init: int, seed: int = 0) -> pd.DataFrame:
        from edbo.plus.optimizer_botorch import EDBOplus
        self._scope_df.to_csv(self._csv_path, index=False)
        optimizer = EDBOplus()
        result = optimizer.run(
            objectives=["y"],
            objective_mode=["min" if self._minimize else "max"],
            directory=self._tmpdir, filename="scope.csv",
            batch=n_init, seed=seed,
        )
        return result[result["priority"] == 1].head(n_init)[self._param_names].reset_index(drop=True)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> None:
        from scipy.spatial.distance import cdist
        y_series = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
        scope_df = self._scope_df.copy()
        scope_df["y"] = "PENDING"
        X_np = X[self._param_names].values
        scope_np = scope_df[self._param_names].values
        for xi, yi in zip(X_np, y_series.values):
            idx = int(np.argmin(cdist([xi], scope_np, metric="cityblock")))
            scope_df.at[idx, "y"] = float(yi)
        scope_df.to_csv(self._csv_path, index=False)

    def suggest(self, n: int = 1) -> Optional[pd.DataFrame]:
        from edbo.plus.optimizer_botorch import EDBOplus
        optimizer = EDBOplus()
        result = optimizer.run(
            objectives=["y"],
            objective_mode=["min" if self._minimize else "max"],
            directory=self._tmpdir, filename="scope.csv", batch=n,
        )
        return result[result["priority"] == 1].head(n)[self._param_names].reset_index(drop=True)

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
