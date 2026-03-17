import unittest
import numpy as np
from unittest.mock import MagicMock
from funcexpr import evaluate, _normalize


class TestNormalize(unittest.TestCase):
    """Tests for the _normalize helper."""

    def test_int_passthrough(self):
        result = _normalize(1)
        self.assertEqual(result, 1)
        self.assertIsInstance(result, int)

    def test_float_passthrough(self):
        result = _normalize(1.0)
        self.assertEqual(result, 1.0)
        self.assertIsInstance(result, float)

    def test_complex_passthrough(self):
        result = _normalize(1 + 2j)
        self.assertEqual(result, 1 + 2j)

    def test_numpy_scalar_passthrough(self):
        x = np.float64(3.14)
        self.assertIs(_normalize(x), x)

    def test_numpy_int_scalar_passthrough(self):
        x = np.int32(5)
        self.assertIs(_normalize(x), x)

    def test_ndarray_passthrough(self):
        arr = np.array([1.0, 2.0])
        self.assertIs(_normalize(arr), arr)

    def test_array_protocol_object(self):
        class ArrayLike:
            def __array__(self, dtype=None):
                return np.array([1.0, 2.0, 3.0])

        result = _normalize(ArrayLike())
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_unsupported_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            _normalize("not_a_number")

    def test_unsupported_object_raises_type_error(self):
        with self.assertRaises(TypeError):
            _normalize(object())


class TestEvaluateBasic(unittest.TestCase):
    """Tests for evaluate with no custom functions."""

    def setUp(self):
        self.a = np.array([1.0, 2.0, 3.0])
        self.b = np.array([4.0, 5.0, 6.0])

    def test_addition(self):
        result = evaluate("a + b", {"a": self.a, "b": self.b})
        np.testing.assert_array_equal(result, self.a + self.b)

    def test_subtraction(self):
        result = evaluate("a - b", {"a": self.a, "b": self.b})
        np.testing.assert_array_equal(result, self.a - self.b)

    def test_multiplication(self):
        result = evaluate("a * b", {"a": self.a, "b": self.b})
        np.testing.assert_array_equal(result, self.a * self.b)

    def test_division(self):
        result = evaluate("a / b", {"a": self.a, "b": self.b})
        np.testing.assert_array_almost_equal(result, self.a / self.b)

    def test_power(self):
        result = evaluate("a ** 2", {"a": self.a})
        np.testing.assert_array_equal(result, self.a ** 2)

    def test_scalar_ctx(self):
        result = evaluate("a * 2.0", {"a": self.a})
        np.testing.assert_array_equal(result, self.a * 2.0)

    def test_complex_expr(self):
        result = evaluate("a + b * 2", {"a": self.a, "b": self.b})
        np.testing.assert_array_equal(result, self.a + self.b * 2)


class TestEvaluateWithFuncs(unittest.TestCase):
    """Tests for evaluate with registered callables."""

    def setUp(self):
        self.a = np.array([1.0, 2.0, 3.0])
        self.b = np.array([4.0, 5.0, 6.0])

    def test_builtin_func_in_callable_arg(self):
        # sin is a numexpr built-in; it should resolve via BUILTIN_FUNCS
        # when used as an argument to a registered callable.
        result = evaluate("sain(sin(a))", {"a": self.a}, funcs={"sain": np.sin})
        np.testing.assert_array_almost_equal(result, np.sin(np.sin(self.a)))

    def test_builtin_func_passthrough(self):
        # sin with no funcs registered should pass through to numexpr.
        result = evaluate("sin(a)", {"a": self.a})
        np.testing.assert_array_almost_equal(result, np.sin(self.a))

    def test_simple_callable(self):
        def double(x):
            return x * 2

        result = evaluate("double(a) + b", {"a": self.a, "b": self.b}, funcs={"double": double})
        np.testing.assert_array_equal(result, self.a * 2 + self.b)

    def test_callable_with_multiple_args(self):
        def weighted(x, y):
            return x * 0.3 + y * 0.7

        result = evaluate("weighted(a, b)", {"a": self.a, "b": self.b}, funcs={"weighted": weighted})
        np.testing.assert_array_almost_equal(result, self.a * 0.3 + self.b * 0.7)

    def test_nested_callable(self):
        def double(x):
            return x * 2

        result = evaluate("double(double(a))", {"a": self.a}, funcs={"double": double})
        np.testing.assert_array_equal(result, self.a * 4)

    def test_callable_with_binop_arg(self):
        def double(x):
            return x * 2

        result = evaluate("double(a + b)", {"a": self.a, "b": self.b}, funcs={"double": double})
        np.testing.assert_array_equal(result, (self.a + self.b) * 2)

    def test_callable_returning_scalar(self):
        def const(_):
            return 5.0

        result = evaluate("const(a) + a", {"a": self.a}, funcs={"const": const})
        np.testing.assert_array_equal(result, 5.0 + self.a)

    def test_callable_is_invoked_once(self):
        mock_fn = MagicMock(return_value=self.a)
        evaluate("f(a)", {"a": self.a}, funcs={"f": mock_fn})
        mock_fn.assert_called_once()

    def test_unregistered_function_raises_value_error(self):
        with self.assertRaises(TypeError):
            evaluate("foo(a)", {"a": self.a}, funcs={})

    def test_unregistered_function_no_funcs_raises_value_error(self):
        with self.assertRaises(TypeError):
            evaluate("foo(a)", {"a": self.a})


class TestEvaluateErrors(unittest.TestCase):
    """Tests for error handling in evaluate."""

    def setUp(self):
        self.a = np.array([1.0, 2.0, 3.0])

    def test_syntax_error_raised(self):
        with self.assertRaises(SyntaxError):
            evaluate("a + (b", {"a": self.a})

    def test_missing_ctx_variable_raises_key_error(self):
        with self.assertRaises(KeyError):
            evaluate("a + b", {"a": self.a})

    def test_invalid_ctx_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            evaluate("a + 1", {"a": "not_an_array"})


class TestEvaluateCtxNormalization(unittest.TestCase):
    """Tests for automatic ctx value normalization."""

    def test_array_protocol_in_ctx(self):
        class ArrayLike:
            def __array__(self, dtype=None):
                return np.array([1.0, 2.0, 3.0])

        result = evaluate("a * 2", {"a": ArrayLike()})
        np.testing.assert_array_equal(result, np.array([2.0, 4.0, 6.0]))

    def test_numpy_scalar_in_ctx(self):
        a = np.array([1.0, 2.0, 3.0])
        result = evaluate("a * c", {"a": a, "c": np.float64(3.0)})
        np.testing.assert_array_equal(result, a * 3.0)