import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp


def equation_in_latex(equation):
    return f'{sp.latex(equation)}'


def evaluate_function(equation, variable, x_values, constants, threshold=None):
    eq = equation.subs(constants)
    lambdified_eq = sp.lambdify(variable, eq.rhs, 'numpy')
    y_values = lambdified_eq(x_values)
    if threshold is not None:
        y_values[y_values < threshold[0]] = np.nan
        y_values[y_values > threshold[1]] = np.nan
    return y_values


def plot_function(equation, variable, x_values, constants, axes, threshold=None):
    y_values = evaluate_function(equation, variable, x_values, constants, threshold=threshold)
    axes.plot(x_values, y_values,
              label=', '.join([f'${sp.latex(c_name)} = {c_val}$' for c_name, c_val in constants]))


def plot_many_functions(equation, variable, diff_equation, x_values, constants_list, axes, legend=True, title=True,
                        threshold=None):
    for constants in constants_list:
        plot_function(equation, variable, x_values, constants, axes, threshold=threshold)
    if legend: axes.legend()
    axes.grid()
    if title:
        axes.set_title(f'Differential equation:\n${equation_in_latex(diff_equation)}$\n'
                       f'Solution:\n${equation_in_latex(equation)}$')


def convert_equation_to_meshgrid(x_var, y_var, equation, x_0, x_k, y_0, y_k, step):
    lambdified_eq = sp.lambdify([x_var, y_var], equation, 'numpy')
    # x_0, x_k, y_0, y_k, step = params
    X, Y = np.meshgrid(np.arange(x_0, x_k, step), np.arange(y_0, y_k, step))
    dy = lambdified_eq(X, Y)
    dx = np.ones(dy.shape)
    norm = np.sqrt(dx ** 2 + dy ** 2)
    return X, Y, dx / norm, dy / norm


def plot_direction_field(x_var, y_var, equation, params, axes, title=True):
    meshgrid = convert_equation_to_meshgrid(x_var, y_var, equation, *params)
    axes.quiver(*meshgrid, pivot='mid')
    axes.grid()
    diff_func_latex = r'\frac{dy}{dx}'
    if title:
        axes.set_title(f'Direction field for equation:\n'
                       f'${diff_func_latex} = {equation_in_latex(equation)}$')


def solve_differential_equation(function, lower_bound, upper_bound, initial_x, initial_y_list, output_size):
    if lower_bound <= initial_x <= upper_bound:
        span_backward = (initial_x, lower_bound)
        span_forward = (initial_x, upper_bound)
        x_range_backward, sol_backward = solve_equation_with_scipy(function, span_backward, initial_y_list, output_size // 2)
        x_range_forward, sol_forward = solve_equation_with_scipy(function, span_forward, initial_y_list, output_size // 2)
        x_range = np.concatenate((x_range_backward[::-1], x_range_forward), axis=0)
        sol = np.concatenate((sol_backward[::, ::-1], sol_forward), axis=1)
    elif initial_x < lower_bound:
        span = (initial_x, lower_bound)
        x_range, sol = solve_equation_with_scipy(function, span, initial_y_list, output_size)
    elif initial_x > upper_bound:
        span = (initial_x, upper_bound)
        x_range, sol = solve_equation_with_scipy(function, span, initial_y_list, output_size)
    else:
        x_range, sol = np.array([]), np.array([])
    return x_range, sol


def solve_equation_with_scipy(function, span, initial_y_list, output_size):
    x_range = np.linspace(*span, output_size)
    solution = solve_ivp(function, span, initial_y_list, t_eval=x_range, dense_output=True)
    return x_range, solution.y


def plot_numeric_solutions(diff_equation, x, f, initial_conditions_y, initial_x, span, size, axes, legend=True,
                          title=True, threshold=None):
    lambdified_eq = sp.lambdify((x, f), diff_equation, 'numpy')

    def first_order_equation(x, f):
        return lambdified_eq(x, f)

    x_range, sol = solve_differential_equation(first_order_equation, *span, initial_x, initial_conditions_y, size)
    for index, y0 in enumerate(initial_conditions_y):
        y_values = sol[index]
        if threshold is not None:
            y_values[y_values < threshold[0]] = np.nan
            y_values[y_values > threshold[1]] = np.nan
        axes.plot(x_range, y_values, label=f'$y({initial_x}) = {y0}$')

    if legend:
        axes.legend()
    axes.grid()
    diff_func_latex = r'\frac{dy}{dx}'
    if title:
        axes.set_title(f'Numerical solution for equation:\n'
                       f'${diff_func_latex} = {equation_in_latex(diff_equation)}$')
