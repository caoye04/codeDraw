import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 解析数据
accept_rounds = [1, 3, 8, 13, 24, 26, 27, 29, 30, 31, 32, 33, 36, 38, 39, 40, 41, 43, 44, 46, 49]
wins = [16, 14, 7, 13, 15, 14, 14, 14, 18, 15, 18, 16, 15, 16, 14, 13, 15, 13, 14, 13, 15]
loses = [3, 6, 13, 7, 5, 5, 6, 5, 2, 5, 2, 4, 5, 4, 6, 6, 4, 7, 6, 7, 5]
draws = [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]

# 计算胜率和不输率
win_rates = [wins[i]/(wins[i]+loses[i]+draws[i]) for i in range(len(wins))]
non_lose_rates = [(wins[i]+draws[i])/(wins[i]+loses[i]+draws[i]) for i in range(len(wins))]

def plot_training_progress(rounds, win_rates, non_lose_rates, accept_count):
    # 创建图表
    plt.figure(figsize=(12, 7), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # 绘制不败率曲线和点
    plt.plot(rounds, non_lose_rates,
             color='#228B22', 
             label='不败率',
             linewidth=2,
             marker='s',
             markersize=4,
             markerfacecolor='#228B22',
             markeredgecolor='#228B22')
    
    # 绘制胜率曲线和点
    plt.plot(rounds, win_rates, 
             color='#0000FF',  # 蓝色
             label='胜率', 
             linewidth=2,
             marker='o',
             markersize=6,
             markerfacecolor='#0000FF',
             markeredgecolor='#0000FF')
    
    # 添加胜率数据标签
    for i, (round_num, rate) in enumerate(zip(rounds, win_rates)):
        plt.annotate(f'{rate:.2f}', 
                    (round_num, rate),
                    xytext=(0, 7),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color='#0000FF')
    
    # 计算平均胜率和最高胜率
    avg_win_rate = np.mean(win_rates)
    max_win_rate = max(win_rates)
    
    # 添加平均胜率虚线和标签
    plt.axhline(y=avg_win_rate, 
                color='red', 
                linestyle='--', 
                alpha=0.8)
    
    # 在红色虚线上添加文字
    plt.text(5, avg_win_rate + 0.01, 
             f'平均胜率: {avg_win_rate:.2f}', 
             color='red',
             fontsize=10)
    
    # 添加最高胜率虚线和标签
    plt.axhline(y=max_win_rate, 
                color='black', 
                linestyle='--', 
                alpha=0.8)
    
    # 在黑色虚线上添加文字
    plt.text(5, max_win_rate + 0.01, 
             f'最高胜率: {max_win_rate:.2f}', 
             color='black',
             fontsize=10)
    
    # 设置标题和标签
    plt.title('AlphaZero VS Random 对弈胜率与不败率变化',
              fontsize=16, 
              pad=20)
    
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('比率', fontsize=12)
    
    # 添加图例
    plt.legend(loc='lower right', fontsize=10)
    
    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.2)
    
    # 在左下角添加统计信息
    avg_non_lose_rate = np.mean(non_lose_rates)
    info_text = (f'总接受模型数(ACCEPT次数): {accept_count}\n'
                f'平均胜率: {avg_win_rate:.2f}\n'
                f'最高胜率: {max(win_rates):.2f}\n'
                f'平均不败率: {avg_non_lose_rate:.2f}')
    plt.text(0.02, 0.02, info_text,
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.8,edgecolor='gray'),
             fontsize=10)
    
    # 在右下角添加参数信息
    param_text = ('参数设置:\n'
                 'update_threshold=0.490\n'
                 'temperature=1.9\n'
                 'lr=0.000001')
    plt.text(0.24, 0.02, param_text,
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=10,)  # 右对齐
    
    # 设置坐标轴范围
    plt.xlim(0, 50)
    plt.ylim(0, 1.0)  # 调整y轴范围到1.0，因为不败率可能超过0.95
    
    # 设置x轴刻度
    plt.xticks(range(0, 51, 5))
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    # 设置y轴显示为小数
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('hw2_lms_1.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    
    plt.show()

# 绘制图表
plot_training_progress(accept_rounds, win_rates, non_lose_rates, len(accept_rounds))