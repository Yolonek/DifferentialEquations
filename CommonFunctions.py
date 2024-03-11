import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp


def plot_function(equation, variable, x_values, constants, axes, threshold=None):
    eq = equation.subs(constants)
    lambdified_eq = sp.lambdify(variable, eq.rhs, 'numpy')
    y_values = lambdified_eq(x_values)
    if threshold is not None:
        y_values[y_values < threshold[0]] = np.nan
        y_values[y_values > threshold[1]] = np.nan
    axes.plot(x_values, y_values,
              label=', '.join([f'${sp.latex(c_name)} = {c_val}$' for c_name, c_val in constants]))


def plot_many_functions(equation, variable, diff_equation, x_values, constants_list, axes, legend=True, title=True,
                        threshold=None):
    for constants in constants_list:
        plot_function(equation, variable, x_values, constants, axes, threshold=threshold)
    if legend: axes.legend()
    axes.grid()
    if title:
        axes.set_title(f'Differential equation:\n${sp.latex(diff_equation)}$\n'
                       f'Solution:\n${sp.latex(equation)}$')


def plot_direction_field(x_var, y_var, equation, params, axes, title=True):
    lambdified_eq = sp.lambdify([x_var, y_var], equation, 'numpy')
    x_0, x_k, y_0, y_k, step = params
    X, Y = np.meshgrid(np.arange(x_0, x_k, step), np.arange(y_0, y_k, step))
    dy = lambdified_eq(X, Y)
    dx = np.ones(dy.shape)
    norm = np.sqrt(dx**2 + dy**2)
    axes.quiver(X, Y, dx / norm, dy / norm)
    axes.grid()
    diff_func_latex = r'\frac{dy}{dx}'
    if title:
        axes.set_title(f'Direction field for equation:\n'
                       f'${diff_func_latex} = {sp.latex(equation)}$')


def plot_numeric_solutions(diff_equation, x, f, initial_conditions_y, initial_x, span, size, axes, legend=True,
                          title=True, threshold=None):
    lambdified_eq = sp.lambdify((x, f), diff_equation, 'numpy')

    def first_order_equation(x, f):
        return lambdified_eq(x, f)

    if span[0] <= initial_x <= span[1]:
        span_backward = (initial_x, span[0])
        span_forward = (initial_x, span[1])
        x_range_backward = np.linspace(*span_backward, size // 2)
        x_range_forward = np.linspace(*span_forward, size // 2)
        x_range = np.concatenate((x_range_backward[::-1], x_range_forward), axis=0)
        sol_backward = solve_ivp(first_order_equation, span_backward, initial_conditions_y,
                                 t_eval=x_range_backward, dense_output=True)
        sol_forward = solve_ivp(first_order_equation, span_forward, initial_conditions_y,
                                t_eval=x_range_forward, dense_output=True)
        for index, y0 in enumerate(initial_conditions_y):
            y_values_backward = sol_backward.y[index]
            y_values_forward = sol_forward.y[index]
            if threshold is not None:
                y_values_backward[y_values_backward < threshold[0]] = np.nan
                y_values_backward[y_values_backward > threshold[1]] = np.nan
                y_values_forward[y_values_forward < threshold[0]] = np.nan
                y_values_forward[y_values_forward > threshold[1]] = np.nan
            y_values = np.concatenate((y_values_backward[::-1], y_values_forward), axis=0)
            axes.plot(x_range, y_values, label=f'$y({initial_x}) = {y0}$')

    if legend:
        axes.legend()
    axes.grid()
    diff_func_latex = r'\frac{dy}{dx}'
    if title:
        axes.set_title(f'Numerical solution for equation:\n'
                       f'${diff_func_latex} = {sp.latex(diff_equation)}$')
