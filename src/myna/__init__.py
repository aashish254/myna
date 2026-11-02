__all__ = ["Myna", "Observation", "MynaConfig", "MynaModel", "fetch_weights", "MynaV6"]

from .engine import Myna, Observation
from .model import MynaConfig, MynaModel
from .weights import fetch_weights


def __getattr__(name):
    """`MynaV6` on demand, not on import (PEP 562).

    It lives in `v6_engine`, which needs `transformers` — an optional extra that CI does not
    install. A plain `from .v6_engine import MynaV6` here would put that extra on the path of
    every `import myna`, turning a missing optional dependency into a collection error for
    the whole suite. `from myna import MynaV6` still works, and the ImportError names the
    extra that fixes it.
    """
    if name == "MynaV6":
        try:
            from .v6_engine import MynaV6
        except ImportError as e:
            raise ImportError(
                "MynaV6 needs the v6 extra: pip install 'myna[v6]' "
                "(in this checkout, `uv sync --extra v6`)") from e
        return MynaV6
    raise AttributeError(f"module 'myna' has no attribute {name!r}")
