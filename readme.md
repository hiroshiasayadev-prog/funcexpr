# funcexpr

A lightweight wrapper around [numexpr](https://github.com/pydata/numexpr) that adds support for registered callables and automatic type normalization.

```python
import numpy as np
import funcexpr as fe

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

def clip_positive(x):
    return np.clip(x, 0, None)

result = fe.evaluate("clip_positive(a - b) + b", ctx={"a": a, "b": b}, funcs={"clip_positive": clip_positive})
# array([4., 5., 6.])
```

## Motivation

numexpr is fast, but it only accepts plain ndarrays and supports a limited set of built-in functions. funcexpr solves this by pre-processing the expression AST before handing it off to numexpr:

- **Registered callables** are evaluated eagerly via NumPy and replaced with temporary variables before the expression reaches numexpr
- **numexpr built-ins** (`sin`, `cos`, `exp`, etc.) at the top level of an expression are passed through to numexpr as-is; when they appear as arguments to a registered callable, they are resolved via NumPy during eager argument evaluation
- **Type normalization** handles Python scalars, numpy scalars, and any object implementing `__array__`

## Installation

```bash
pip install funcexpr
```

## Usage

### Basic

```python
import numpy as np
import funcexpr as fe

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

fe.evaluate("a + b * 2", ctx={"a": a, "b": b})
# array([ 9., 12., 15.])
```

### Registered callables

Any Python callable can be registered via `funcs`. Arguments are evaluated eagerly before the callable is invoked, so nested calls and expressions as arguments work naturally.

```python
def double(x):
    return x * 2

fe.evaluate("double(a) + b", ctx={"a": a, "b": b}, funcs={"double": double})
# array([ 6.,  9., 12.])
```

Nested calls:

```python
fe.evaluate("double(double(a))", ctx={"a": a}, funcs={"double": double})
# array([ 4.,  8., 12.])
```

Expression as argument:

```python
fe.evaluate("double(a + b)", ctx={"a": a, "b": b}, funcs={"double": double})
# array([10., 14., 18.])
```

### numexpr built-ins

numexpr built-ins used at the top level of an expression are passed through to numexpr directly and require no registration.

```python
fe.evaluate("sin(a) + cos(b)", ctx={"a": a, "b": b})
```

When a built-in appears as an argument to a registered callable, it is resolved via NumPy during eager argument evaluation.

```python
def my_func(x):
    return x * 2

fe.evaluate("my_func(sin(a))", ctx={"a": a}, funcs={"my_func": my_func})
# equivalent to my_func(np.sin(a))
```

### Type normalization

`ctx` values and callable return values are normalized automatically.

| Type | Behavior |
|---|---|
| `int`, `float`, `complex` | passed through as-is |
| `np.generic` (numpy scalar) | passed through as-is |
| `np.ndarray` | passed through as-is |
| object with `__array__` | converted via `np.asarray()` |
| anything else | `TypeError` |

## API reference

```python
def evaluate(
    expr: str,
    ctx: dict[str, np.ndarray | int | float | complex | np.generic],
    funcs: dict[str, Callable] | None = None,
) -> np.ndarray:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `expr` | `str` | — | Python expression string to evaluate |
| `ctx` | `dict` | — | Variable context |
| `funcs` | `dict[str, Callable] \| None` | `None` | Registered callables |

**Returns** `np.ndarray`.

## Error handling

| Condition | Exception |
|---|---|
| Invalid expression | `SyntaxError` |
| Variable missing from `ctx` | `KeyError` (raised by numexpr) |
| Unnormalizable type in `ctx` or callable return value | `TypeError` |
| Unrecognized function name | `TypeError` (raised by numexpr) |
| Any other numexpr or callable error | propagated as-is |

## Limitations

Callable arguments support a subset of AST node types: `Constant`, `Name`, `BinOp`, `UnaryOp`, and `Call`. Passing unsupported node types (e.g. `Compare`, `BoolOp`, `IfExp`) as callable arguments will raise `TypeError`. This may be lifted in a future version.

## Design

funcexpr is intentionally minimal. It does one thing: let you use arbitrary callables inside numexpr expressions. More advanced features such as xarray DataArray support and axis alignment are out of scope and belong in a higher-level layer.

## License

MIT