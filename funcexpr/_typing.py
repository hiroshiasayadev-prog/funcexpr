from typing import Protocol
import numpy as np

# Type alias for values accepted in ctx and returned by Callable.
class SupportsArray(Protocol):
    def __array__(self) -> np.ndarray: ...
Scalar = int | float | complex | np.generic
CtxValue = SupportsArray | Scalar