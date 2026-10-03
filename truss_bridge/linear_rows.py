"""truss_bridge/linear_rows.py

Incremental construction of sparse linear constraints for scipy's MILP solver.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.sparse import coo_matrix


class LinearRows:
    """A growing list of constraints ``lower <= sum(coefficient * x[variable]) <= upper``."""

    def __init__(self) -> None:
        self.rows: list[dict[int, float]] = []
        self.lower_bounds: list[float] = []
        self.upper_bounds: list[float] = []

    def add(self, coefficients: dict[int, float], lower: float = -np.inf, upper: float = np.inf) -> None:
        """Append one constraint row.

        Args:
            coefficients: Mapping from variable index to coefficient.
            lower: Lower bound of the row sum.
            upper: Upper bound of the row sum.
        """
        self.rows.append(coefficients)
        self.lower_bounds.append(lower)
        self.upper_bounds.append(upper)

    def at_most_one(self, first: int, second: int) -> None:
        """Forbid both binary variables from being 1."""
        self.add({first: 1, second: 1}, upper=1)

    def to_constraint(self, variable_count: int) -> LinearConstraint:
        """Sparse scipy constraint over ``variable_count`` variables."""
        row_indices, column_indices, values = [], [], []
        for row_index, coefficients in enumerate(self.rows):
            for variable, coefficient in coefficients.items():
                row_indices.append(row_index)
                column_indices.append(variable)
                values.append(coefficient)
        matrix = coo_matrix((values, (row_indices, column_indices)), shape=(len(self.rows), variable_count))
        return LinearConstraint(matrix.tocsr(), self.lower_bounds, self.upper_bounds)
