# Curated set of builtins exposed to model-generated code. Dangerous entry
# points (open, exec, eval, __import__, compile, input, ...) are intentionally
# left out so the REPL is restricted rather than full Python.
import builtins


_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "len", "list", "map", "max", "min", "print", "range", "repr",
    "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
]
SAFE_BUILTINS = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}