import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# 物理常数
h = 6.626e-34  # 普朗克常数 (J·s)
c = 2.998e8    # 光速 (m/s)
kB = 1.381e-23 # 玻尔兹曼常数 (J/K)
lambda1 = 390e-9  # 可见光下限 (m)
lambda2 = 750e-9  # 可见光上限 (m)

def planck_integrand(x):
    """普朗克辐射定律积分内核函数"""
    return x**3 / (np.exp(x) - 1)

def efficiency(T):
    """计算给定温度T下的发光效率"""
    # 计算积分上下限
    x1 = h * c / (lambda2 * kB * T)
    x2 = h * c / (lambda1 * kB * T)
    
    # 数值积分
    integral, _ = quad(planck_integrand, x1, x2)
    
    # 计算效率 (单位: %)
    eta = (15 / np.pi**4) * integral * 100
    return eta

def find_optimal_temperature(bracket=(2000, 4000)):
    """使用黄金分割法寻找最优温度"""
    # 定义目标函数 (添加负号以便使用最小化算法)
    objective = lambda T: -efficiency(T)
    
    # 使用黄金分割法进行优化
    result = minimize_scalar(objective, method='golden', bracket=bracket, tol=1e-1)
    
    # 计算最优温度和最大效率
    T_opt = result.x
    eta_max = -result.fun
    
    return T_opt, eta_max

def plot_efficiency_vs_temperature(T_range=(300, 10000), num_points=500):
    """绘制效率随温度变化的曲线"""
    # 生成温度数组
    T_array = np.linspace(T_range[0], T_range[1], num_points)
    eta_array = np.array([efficiency(T) for T in T_array])
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(T_array, eta_array, 'b-', linewidth=2)
    
    # 标记实际工作温度点
    T_actual = 2700
    eta_actual = efficiency(T_actual)
    plt.scatter(T_actual, eta_actual, color='red', s=100, zorder=5)
    plt.annotate(f'实际工作点: {T_actual}K, {eta_actual:.2f}%',
                xy=(T_actual, eta_actual),
                xytext=(T_actual + 500, eta_actual + 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # 标记理论最优温度点
    T_opt, eta_max = find_optimal_temperature()
    plt.scatter(T_opt, eta_max, color='green', s=100, zorder=5)
    plt.annotate(f'理论最优点: {T_opt:.0f}K, {eta_max:.2f}%',
                xy=(T_opt, eta_max),
                xytext=(T_opt - 2000, eta_max - 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # 设置图表属性
    plt.title('白炽灯发光效率 vs 温度', fontsize=16)
    plt.xlabel('温度 (K)', fontsize=14)
    plt.ylabel('发光效率 (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(T_range)
    plt.ylim(0, 12)
    
    # 标记钨丝熔点
    plt.axvline(x=3695, color='gray', linestyle='--', alpha=0.5)
    plt.text(3700, 10, '钨丝熔点 (3695K)', rotation=90, fontsize=12)
    
    return plt.gcf()

def main():
    """主函数：执行计算和分析"""
    # 计算理论最优温度
    T_opt, eta_max = find_optimal_temperature()
    print(f"理论最优温度 T_opt = {T_opt:.0f} K")
    print(f"最大发光效率 η_max = {eta_max:.2f} %")
    
    # 计算实际工作温度下的效率
    T_actual = 2700
    eta_actual = efficiency(T_actual)
    print(f"实际工作温度 T_actual = {T_actual} K")
    print(f"实际发光效率 η_actual = {eta_actual:.2f} %")
    
    # 绘制效率-温度曲线
    fig = plot_efficiency_vs_temperature()
    plt.tight_layout()
    plt.savefig('efficiency_vs_temperature.png', dpi=300)
    plt.show()
    
    # 可行性分析
    print("\n=== 可行性分析 ===")
    print(f"钨丝熔点: 3695 K")
    print(f"理论最优温度与熔点差距: {3695 - T_opt:.0f} K")
    print(f"实际温度与理论最优温度差距: {T_opt - T_actual:.0f} K")
    
    # 效率提升潜力
    improvement = (eta_max / eta_actual - 1) * 100
    print(f"理论上效率可提升: {improvement:.1f}%")

if __name__ == "__main__":
    main()
