import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp


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


def plot_numeric_solution(diff_equation, x, f, initial_conditions, span, size, axes, legend=True,
                          title=True, threshold=None):
    lambdified_eq = sp.lambdify((x, f), diff_equation, 'numpy')

    def first_order_equation(x, f):
        return lambdified_eq(x, f)

    x_range = np.linspace(*span, size)
    sol = solve_ivp(first_order_equation, span, initial_conditions, t_eval=x_range)
    for index, y0 in enumerate(initial_conditions):
        y_values = sol.y[index]
        if threshold is not None:
            y_values[y_values < threshold[0]] = np.nan
            y_values[y_values > threshold[1]] = np.nan
        axes.plot(x_range, y_values, label=f'$y(0) = {y0}$')
    if legend: axes.legegnd()
    axes.grid()
    diff_func_latex = r'\frac{dy}{dx}'
    if title:
        axes.set_title(f'Numerical solution for equation:\n'
                       f'${diff_func_latex} = {sp.latex(diff_equation)}$')
