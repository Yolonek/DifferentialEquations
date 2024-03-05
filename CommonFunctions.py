import sympy as sp
import numpy as np


def plot_function(equation, variable, diff_equation, x_values, constants_list, axes, legend=True, title=True,
                  threshold=None):
    for constants in constants_list:
        eq = equation.subs(constants)
        lambdified_eq = sp.lambdify(variable, eq.rhs, 'numpy')
        y_values = lambdified_eq(x_values)
        if threshold is not None:
            y_values[y_values < threshold[0]] = np.nan
            y_values[y_values > threshold[1]] = np.nan
        axes.plot(x_values, y_values,
                  label=', '.join([f'${sp.latex(c_name)} = {c_val}$' for c_name, c_val in constants]))
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
