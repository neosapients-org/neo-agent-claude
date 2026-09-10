"""
Budget guard exceptions for ns_probe.
"""


class CostBudgetExceeded(Exception):
    """Raised when cost budget is exceeded."""

    pass


class LoopBudgetExceeded(Exception):
    """Raised when iteration/loop budget is exceeded."""

    pass
