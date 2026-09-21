from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from collections import defaultdict

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    right_eps_shift  = list(vals)
    right_eps_shift[arg] += epsilon
    left_eps_shift = list(vals)
    left_eps_shift[arg] -= epsilon

    return (f(*right_eps_shift) - f(*left_eps_shift)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def dfs(variable: Variable, used: set) -> list[Variable]:
    """
    Returns
    """
    if variable.is_constant() or variable.unique_id in used:
        return []

    used.add(variable.unique_id)
    if variable.is_leaf():
        return [variable]

    dfs_pass = []
    for ancestor in variable.parents:
        dfs_pass.extend(dfs(ancestor, used))

    dfs_pass.append(variable)

    return dfs_pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    return dfs(variable, set())[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    cur_var_der = defaultdict(int)
    cur_var_der[variable.unique_id] = deriv

    q = topological_sort(variable)

    for var in q:
        if var.is_leaf():
            var.accumulate_derivative(cur_var_der[var.unique_id])
        else:
            for parent_var, parent_der in var.chain_rule(cur_var_der[var.unique_id]):
                cur_var_der[parent_var.unique_id] += parent_der


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
