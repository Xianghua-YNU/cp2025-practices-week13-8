import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# 物理常数
G = 6.674e-11  # 万有引力常数 (m^3 kg^-1 s^-2)
M = 5.974e24   # 地球质量 (kg)
m = 7.348e22   # 月球质量 (kg)
R = 3.844e8    # 地月距离 (m)
omega = 2.662e-6  # 月球角速度 (s^-1)

# L1拉格朗日点位置方程
def lagrange_equation(r):
    """
    在L1点，卫星受到的地球引力、月球引力和离心力平衡。
    方程形式为：G*M/r^2 - G*m/(R-r)^2 - omega^2*r = 0
    """
    earth_gravity = G * M / (r**2)
    moon_gravity = G * m / ((R - r)**2)
    centrifugal_force = omega**2 * r
    return earth_gravity - moon_gravity - centrifugal_force

# L1拉格朗日点位置方程的导数，用于牛顿法
def lagrange_equation_derivative(r):
    earth_gravity_derivative = -2 * G * M / (r**3)
    moon_gravity_derivative = -2 * G * m / ((R - r)**3)
    centrifugal_force_derivative = omega**2
    return earth_gravity_derivative + moon_gravity_derivative - centrifugal_force_derivative

# 使用牛顿法（切线法）求解方程f(x)=0
def newton_method(f, df, x0, tol=1e-8, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = df(x)
        if abs(dfx) < 1e-14:  # 避免除以接近零的数
            break
        delta = fx / dfx
        x = x - delta
    return x

# 使用弦截法求解方程f(x)=0
def secant_method(f, a, b, tol=1e-8, max_iter=100):
    fa = f(a)
    fb = f(b)
    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b
    if fa * fb > 0:  # 确保区间端点函数值异号
        print("警告: 区间端点函数值同号，弦截法可能不收敛")
    for i in range(max_iter):
        x = b - fb * (b - a) / (fb - fa)
        fx = f(x)
        if abs(fx) < tol:  # 函数值接近零
            return x
        if fa * fx < 0:  # 更新区间
            a, b = b, x
            fa, fb = fb, fx
        else:
            a, b = x, b
            fa, fb = fx, fb
    return x

# 使用SciPy的fsolve求解
def fsolve_method(f, x0):
    return fsolve(f, x0)[0]

# 初始猜测值
r0_newton = 3.5e8  # 牛顿法初始猜测值
a, b = 3.2e8, 3.7e8  # 弦截法初始区间
r0_fsolve = 3.5e8  # fsolve初始猜测值

# 求解
r_newton = newton_method(lagrange_equation, lagrange_equation_derivative, r0_newton)
r_secant = secant_method(lagrange_equation, a, b)
r_fsolve = fsolve_method(lagrange_equation, r0_fsolve)

# 输出结果
print(f"牛顿法求解结果：{r_newton:.8e} m")
print(f"弦截法求解结果：{r_secant:.8e} m")
print(f"SciPy fsolve求解结果：{r_fsolve:.8e} m")

# 绘制L1拉格朗日点位置方程的函数图像
def plot_lagrange_equation(r_min, r_max, num_points=1000):
    r_values = np.linspace(r_min, r_max, num_points)
    f_values = np.array([lagrange_equation(r) for r in r_values])
    zero_crossings = np.where(np.diff(np.signbit(f_values)))[0]
    r_zeros = []
    for idx in zero_crossings:
        r1, r2 = r_values[idx], r_values[idx + 1]
        f1, f2 = f_values[idx], f_values[idx + 1]
        r_zero = r1 - f1 * (r2 - r1) / (f2 - f1)
        r_zeros.append(r_zero)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_values / 1e8, f_values, 'b-', label='L1 point equation')
    for r_zero in r_zeros:
        ax.plot(r_zero / 1e8, 0, 'ro', label=f'Zero point: {r_zero:.4e} m')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Distance from Earth center (10^8 m)')
    ax.set_ylabel('Equation value')
    ax.set_title('L1 Lagrange Point Equation')
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())
    ax.grid(True, alpha=0.3)
    return fig

# 绘制方程图像
fig = plot_lagrange_equation(3.0e8, 3.8e8)
plt.savefig('lagrange_equation.png', dpi=300)
plt.show()


