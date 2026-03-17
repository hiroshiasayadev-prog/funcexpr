import ast
from typing import Callable, Any
import operator

import numpy as np
import numexpr as ne

from ._typing import CtxValue

# Mapping from AST binary operator node type to the corresponding callable.
# Used in _eval_node to evaluate BinOp nodes without rebuilding the dict on
# every call.
OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Mapping from function name to its NumPy equivalent, covering all functions
# supported by numexpr. Used as a fallback in _eval_node when a Call node is
# not registered in funcs, allowing numexpr built-ins to be resolved during
# eager argument evaluation.
BUILTIN_FUNCS: dict[str, Callable] = {
    # trigonometric
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    # hyperbolic
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "arcsinh": np.arcsinh,
    "arccosh": np.arccosh,
    "arctanh": np.arctanh,
    # exponential / logarithmic
    "exp": np.exp,
    "expm1": np.expm1,
    "log": np.log,
    "log10": np.log10,
    "log1p": np.log1p,
    # basic math
    "sqrt": np.sqrt,
    "abs": np.abs,
    # complex
    "conj": np.conj,
    "real": np.real,
    "imag": np.imag,
}


def _normalize(value: object) -> CtxValue:
    """Normalize *value* to a type accepted by numexpr.

    Resolution order:
    1. ``int``, ``float``, ``complex`` — passed through as-is; numexpr handles
       Python scalars natively.
    2. ``np.generic`` (any numpy scalar) — passed through as-is.
    3. ``np.ndarray`` — passed through as-is.
    4. Objects that implement the ``__array__`` protocol — converted via
       ``np.asarray(value)``.

    Args:
        value: The value to normalize.

    Returns:
        The normalized value, ready to be placed in a numexpr context dict.

    Raises:
        TypeError: if *value* cannot be converted to a numexpr-compatible type.
    """
    if isinstance(value, (int, float, complex)):
        return value
    if isinstance(value, np.generic):
        return value
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "__array__"):
        return np.asarray(value)
    raise TypeError(
        f"Cannot normalize value of type {type(value).__name__!r} for use in "
        "numexpr. Expected int, float, complex, np.generic, np.ndarray, or an "
        "object implementing the __array__ protocol."
    )


def _eval_node(
    node: ast.expr,
    ctx: dict[str, CtxValue],
    funcs: dict[str, Callable],
) -> CtxValue:
    """Recursively evaluate an AST node to a concrete value.

    Used to evaluate the arguments of a ``Call`` node before passing them to
    the registered callable. Mirrors the structure of a simple AST interpreter:
    leaf nodes return immediately, compound nodes recurse into their children.

    Only the node types needed to evaluate callable arguments are supported:
    ``Constant``, ``Name``, ``BinOp``, ``UnaryOp``, and ``Call``.

    Args:
        node: The AST node to evaluate.
        ctx: The current variable context, including any intermediate variables
            accumulated so far.
        funcs: Registered callables, forwarded for nested ``Call`` nodes.

    Returns:
        The evaluated value as a ``CtxValue``.

    Raises:
        ValueError: if a ``Call`` node references a function not in *funcs*.
        KeyError: if a ``Name`` node references a variable not in *ctx*.

    Note:
        Functions not present in *funcs* are passed through to numexpr as-is,
        allowing numexpr built-ins (e.g. ``sin``, ``cos``, ``exp``) to be used
        without registration. If numexpr does not recognize the function name,
        it will raise. Callable arguments support a subset of AST node types
        (``Constant``, ``Name``, ``BinOp``, ``UnaryOp``, ``Call``). Passing
        unsupported node types (e.g. ``Compare``, ``BoolOp``, ``IfExp``) as
        callable arguments will raise ``TypeError``. This limitation may be
        lifted in a future version.
    """
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        return ctx[node.id]

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, ctx, funcs)
        if isinstance(node.op, ast.USub):
            return -operand
        return operand

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, ctx, funcs)
        right = _eval_node(node.right, ctx, funcs)
        return OPS[type(node.op)](left, right)

    if isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in funcs:
            raise ValueError(
                f"Function {func_name!r} is not registered in funcs."
            )
        args = [_eval_node(a, ctx, funcs) for a in node.args]
        result = funcs[func_name](*args)
        return _normalize(result)

    raise TypeError(f"Unsupported AST node type: {type(node).__name__!r}")


def _substitute_calls(
    node: ast.expr,
    funcs: dict[str, Callable],
    ctx: dict[str, CtxValue],
    tmp_ctx: dict[str, CtxValue],
    counter: list[int],
) -> str:
    """Recursively convert an AST node to an expression string, substituting
    ``Call`` nodes with temporary variable names.

    When a ``Call`` node is encountered, its arguments are evaluated eagerly
    via :func:`_eval_node`, the callable is invoked, and the result is stored
    in *tmp_ctx* under a generated name (``__tmp0``, ``__tmp1``, …). The
    generated name is then spliced into the reconstructed expression string so
    that numexpr can reference the pre-computed value.

    Args:
        node: The current AST node to process.
        funcs: Registered callables. A ``Call`` referencing a name absent from
            this dict raises ``ValueError``.
        ctx: The original variable context passed to :func:`evaluate`.
        tmp_ctx: Accumulator for intermediate variables produced by ``Call``
            substitution. Modified in place.
        counter: A single-element list used as a mutable integer counter for
            generating unique ``__tmpN`` names. Modified in place.

    Returns:
        A string fragment of the reconstructed expression with all ``Call``
        nodes replaced by ``__tmpN`` references.

    Raises:
        KeyError: if a ``Name`` node references a variable not in *ctx*.
    """
    if isinstance(node, ast.Constant):
        return str(node.value)

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.UnaryOp):
        operand = _substitute_calls(node.operand, funcs, ctx, tmp_ctx, counter)
        if isinstance(node.op, ast.USub):
            return f"-{operand}"
        return operand

    if isinstance(node, ast.BinOp):
        left = _substitute_calls(node.left, funcs, ctx, tmp_ctx, counter)
        right = _substitute_calls(node.right, funcs, ctx, tmp_ctx, counter)
        op_symbols = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
            ast.Pow: "**",
        }
        return f"({left}{op_symbols[type(node.op)]}{right})"

    if isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None

        if funcs and func_name in funcs:
            # Registered callable: evaluate arguments eagerly, invoke the
            # function, normalize the result, and replace the call with a
            # temporary variable so numexpr can reference the pre-computed value.
            merged_ctx = {**ctx, **tmp_ctx}
            args = [_eval_node(a, merged_ctx, funcs) for a in node.args]
            result = funcs[func_name](*args)
            normalized = _normalize(result)
            tmp_name = f"__tmp{counter[0]}"
            counter[0] += 1
            tmp_ctx[tmp_name] = normalized
            return tmp_name
        else:
            # Unregistered callable: pass through to numexpr as-is.
            # numexpr built-ins (e.g. sin, cos) are handled here.
            # If numexpr does not recognize the function, it will raise.
            args = [_substitute_calls(a, funcs, ctx, tmp_ctx, counter) for a in node.args]
            return f"{func_name}({', '.join(args)})"

    raise TypeError(f"Unsupported AST node type: {type(node).__name__!r}")


def evaluate(
    expr: str,
    ctx: dict[str, CtxValue],
    funcs: dict[str, Callable] | None = None,
) -> np.ndarray:
    """Evaluate a Python expression string using numexpr, with support for
    registered callables and automatic type normalization.

    Extends ``numexpr.evaluate`` by pre-processing the expression AST:
    any ``Call`` nodes referencing functions in *funcs* are evaluated eagerly
    and replaced with temporary variables before the expression is handed off
    to numexpr. This allows arbitrary Python callables (e.g. custom
    interpolation functions) to be used inside numexpr expressions.

    Type normalization is applied to all values in *ctx* and to the return
    values of callables, converting Python scalars, numpy scalars, and objects
    implementing ``__array__`` to types accepted by numexpr.

    Args:
        expr:
            A Python expression string (e.g. ``"my_func(a) + b * c"``).
        ctx:
            A mapping from variable name to value. Accepts ``int``, ``float``,
            ``complex``, ``np.generic``, ``np.ndarray``, or any object
            implementing the ``__array__`` protocol.
        funcs:
            Optional mapping from function name to callable. Each callable
            receives evaluated arguments and must return a value normalizable
            by :func:`_normalize`. Defaults to ``None`` (no custom functions).

    Returns:
        The result of the expression as a ``np.ndarray``.

    Raises:
        SyntaxError: if *expr* is not a valid Python expression.
        TypeError: if a value in *ctx* or a callable return value cannot be
            normalized to a numexpr-compatible type.
        KeyError: if *expr* references a variable not present in *ctx*.
        Exception: any exception raised by numexpr or a registered callable
            is propagated as-is.

    Examples:
        >>> import numpy as np
        >>> import funcexpr as fe
        >>> a = np.array([1.0, 2.0, 3.0])
        >>> b = np.array([4.0, 5.0, 6.0])
        >>> fe.evaluate("a + b * 2", ctx={"a": a, "b": b})
        array([ 9., 12., 15.])

        >>> def double(x):
        ...     return x * 2
        >>> fe.evaluate("double(a) + b", ctx={"a": a, "b": b}, funcs={"double": double})
        array([ 6., 9., 12.])

    Note:
        Functions not present in *funcs* are passed through to numexpr as-is,
        allowing numexpr built-ins (e.g. ``sin``, ``cos``, ``exp``) to be used
        without registration. If numexpr does not recognize the function name,
        it will raise.
        When a built-in appears as an argument to a registered callable (e.g.
        ``my_func(sin(a))``), it is resolved via ``BUILTIN_FUNCS`` using NumPy
        during eager argument evaluation.
        Callable arguments support a subset of AST node types (``Constant``,
        ``Name``, ``BinOp``, ``UnaryOp``, ``Call``). Passing unsupported node
        types (e.g. ``Compare``, ``BoolOp``, ``IfExp``) as callable arguments
        will raise ``TypeError``. This limitation may be lifted in a future
        version.
    """
    # Merge user-provided funcs with BUILTIN_FUNCS. BUILTIN_FUNCS serves as a
    # fallback so that numexpr built-ins (e.g. sin, cos) can be resolved during
    # eager argument evaluation in _eval_node. User-provided funcs take precedence.
    merged_funcs = {**BUILTIN_FUNCS, **(funcs or {})}

    # Parse and validate the expression first.
    tree = ast.parse(expr, mode="eval")

    # Normalize all ctx values up front.
    normalized_ctx: dict[str, CtxValue] = {
        k: _normalize(v) for k, v in ctx.items()
    }

    # Substitute Call nodes and accumulate intermediate variables.
    tmp_ctx: dict[str, CtxValue] = {}
    counter = [0]
    substituted_expr = _substitute_calls(
        tree.body, merged_funcs or {}, normalized_ctx, tmp_ctx, counter
    )

    # Merge ctx and intermediate variables for numexpr.
    eval_ctx = {**normalized_ctx, **tmp_ctx}

    return ne.evaluate(substituted_expr, local_dict=eval_ctx, global_dict={})