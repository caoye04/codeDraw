import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# 设置中文字体
plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 解析数据函数
def parse_experiment_data(data_str):
    lines = data_str.strip().split("\n")
    steps = []
    train_losses = []
    val_losses = []
    
    for line in lines:
        if line.startswith("step "):
            parts = line.split()
            step = int(parts[1].strip(":"))
            train_loss = float(parts[4].strip(","))
            val_loss = float(parts[7])
            
            steps.append(step)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
    
    return steps, train_losses, val_losses

# 实验2：原始注意力机制
naive_attention_experiment = """
step 0: train loss 10.8687, val loss 10.8612
step 50: train loss 7.7211, val loss 7.3998
step 100: train loss 6.6847, val loss 5.9085
step 150: train loss 6.3893, val loss 5.6637
step 200: train loss 6.2696, val loss 5.5327
step 250: train loss 6.1367, val loss 5.4427
step 300: train loss 6.1035, val loss 5.3920
step 350: train loss 6.0469, val loss 5.3374
step 400: train loss 5.9526, val loss 5.3672
step 450: train loss 5.9053, val loss 5.2776
step 500: train loss 5.8906, val loss 5.3069
step 550: train loss 5.8820, val loss 5.3083
step 600: train loss 5.8300, val loss 5.3190
step 650: train loss 5.7692, val loss 5.2925
step 700: train loss 5.8003, val loss 5.2605
step 750: train loss 5.7436, val loss 5.2619
step 800: train loss 5.7139, val loss 5.2769
step 850: train loss 5.6579, val loss 5.2022
step 900: train loss 5.6612, val loss 5.2253
step 950: train loss 5.6659, val loss 5.2594
step 1000: train loss 5.6437, val loss 5.2367
step 1050: train loss 5.6053, val loss 5.1830
step 1100: train loss 5.6543, val loss 5.1824
step 1150: train loss 5.6068, val loss 5.1869
step 1200: train loss 5.6151, val loss 5.1794
step 1250: train loss 5.5847, val loss 5.2259
step 1300: train loss 5.5835, val loss 5.2234
step 1350: train loss 5.5644, val loss 5.1748
step 1400: train loss 5.5977, val loss 5.1808
step 1450: train loss 5.5623, val loss 5.1863
step 1500: train loss 5.5510, val loss 5.1744
step 1550: train loss 5.5249, val loss 5.1796
step 1600: train loss 5.5298, val loss 5.1436
step 1650: train loss 5.5045, val loss 5.1450
step 1700: train loss 5.5185, val loss 5.1488
step 1750: train loss 5.5168, val loss 5.1154
step 1800: train loss 5.4770, val loss 5.1791
step 1850: train loss 5.5181, val loss 5.1240
step 1900: train loss 5.4775, val loss 5.1478
step 1950: train loss 5.4770, val loss 5.1090
step 2000: train loss 5.4998, val loss 5.0990
"""

# 实验2：实现的因果自注意力机制
causal_attention_experiment = """
step 0: train loss 10.8662, val loss 10.8542
step 50: train loss 7.5865, val loss 7.1151
step 100: train loss 6.6412, val loss 5.8539
step 150: train loss 6.3711, val loss 5.5990
step 200: train loss 6.1624, val loss 5.4743
step 250: train loss 5.9518, val loss 5.3351
step 300: train loss 5.9141, val loss 5.2737
step 350: train loss 5.7503, val loss 5.2125
step 400: train loss 5.6557, val loss 5.1563
step 450: train loss 5.6166, val loss 5.0761
step 500: train loss 5.5589, val loss 4.9830
step 550: train loss 5.5321, val loss 4.9942
step 600: train loss 5.4571, val loss 4.9575
step 650: train loss 5.3777, val loss 4.9157
step 700: train loss 5.3466, val loss 4.9017
step 750: train loss 5.3151, val loss 4.8989
step 800: train loss 5.2337, val loss 4.8804
step 850: train loss 5.2252, val loss 4.8674
step 900: train loss 5.1698, val loss 4.7963
step 950: train loss 5.1331, val loss 4.7871
step 1000: train loss 5.0937, val loss 4.7383
step 1050: train loss 5.0598, val loss 4.7787
step 1100: train loss 5.0431, val loss 4.7035
step 1150: train loss 5.0246, val loss 4.7914
step 1200: train loss 5.0566, val loss 4.7793
step 1250: train loss 4.9954, val loss 4.7369
step 1300: train loss 4.9862, val loss 4.7073
step 1350: train loss 4.9738, val loss 4.7048
step 1400: train loss 4.9081, val loss 4.7076
step 1450: train loss 4.9727, val loss 4.6432
step 1500: train loss 4.9203, val loss 4.5844
step 1550: train loss 4.9268, val loss 4.6486
step 1600: train loss 4.8944, val loss 4.6228
step 1650: train loss 4.9005, val loss 4.6196
step 1700: train loss 4.8782, val loss 4.6330
step 1750: train loss 4.8147, val loss 4.6198
step 1800: train loss 4.8108, val loss 4.5844
step 1850: train loss 4.8373, val loss 4.6096
step 1900: train loss 4.8561, val loss 4.5650
step 1950: train loss 4.8106, val loss 4.5878
step 2000: train loss 4.8221, val loss 4.5844
"""

# 解析数据
naive_steps, naive_train_losses, naive_val_losses = parse_experiment_data(naive_attention_experiment)
causal_steps, causal_train_losses, causal_val_losses = parse_experiment_data(causal_attention_experiment)

# 方法2：使用分段坐标轴
def plot_attention_comparison_with_broken_axis():
    # 创建带有分段坐标轴的图表
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    
    # 创建两个子图，一个显示4-6区间，一个显示6-11区间
    ax1 = plt.subplot(2, 1, 1)  # 上面的子图 (6-11)
    ax2 = plt.subplot(2, 1, 2)  # 下面的子图 (4-6)
    
    # 计算性能指标
    final_train_improvement = (naive_train_losses[-1] - causal_train_losses[-1])
    final_val_improvement = (naive_val_losses[-1] - causal_val_losses[-1])
    train_improvement_pct = (final_train_improvement / naive_train_losses[-1]) * 100
    val_improvement_pct = (final_val_improvement / naive_val_losses[-1]) * 100
    
    # 在上半部分绘制曲线 (6-11区间)
    # 原始注意力-训练损失
    ax1.plot(naive_steps, naive_train_losses,
             color='#D86613',  # 橙色
             label='原始注意力-训练损失',
             linewidth=2,
             marker='o',
             markersize=4,
             markerfacecolor='#D86613',
             markeredgecolor='#D86613',
             markevery=5)
    
    # 原始注意力-验证损失
    ax1.plot(naive_steps, naive_val_losses,
             color='#F2A058',  # 浅橙色
             label='原始注意力-验证损失',
             linewidth=2,
             marker='s',
             markersize=4,
             markerfacecolor='#F2A058',
             markeredgecolor='#F2A058',
             markevery=5)
    
    # 因果自注意力-训练损失
    ax1.plot(causal_steps, causal_train_losses,
             color='#1E88E5',  # 蓝色
             label='因果自注意力-训练损失',
             linewidth=2,
             marker='^',
             markersize=4,
             markerfacecolor='#1E88E5',
             markeredgecolor='#1E88E5',
             markevery=5)
    
    # 因果自注意力-验证损失
    ax1.plot(causal_steps, causal_val_losses,
             color='#90CAF9',  # 浅蓝色
             label='因果自注意力-验证损失',
             linewidth=2,
             marker='d',
             markersize=4,
             markerfacecolor='#90CAF9',
             markeredgecolor='#90CAF9',
             markevery=5)
    
    # 在下半部分绘制相同的曲线 (4-6区间)
    # 原始注意力-训练损失
    ax2.plot(naive_steps, naive_train_losses,
             color='#D86613',  # 橙色
             label='原始注意力-训练损失',
             linewidth=2,
             marker='o',
             markersize=4,
             markerfacecolor='#D86613',
             markeredgecolor='#D86613',
             markevery=5)
    
    # 原始注意力-验证损失
    ax2.plot(naive_steps, naive_val_losses,
             color='#F2A058',  # 浅橙色
             label='原始注意力-验证损失',
             linewidth=2,
             marker='s',
             markersize=4,
             markerfacecolor='#F2A058',
             markeredgecolor='#F2A058',
             markevery=5)
    
    # 因果自注意力-训练损失
    ax2.plot(causal_steps, causal_train_losses,
             color='#1E88E5',  # 蓝色
             label='因果自注意力-训练损失',
             linewidth=2,
             marker='^',
             markersize=4,
             markerfacecolor='#1E88E5',
             markeredgecolor='#1E88E5',
             markevery=5)
    
    # 因果自注意力-验证损失
    ax2.plot(causal_steps, causal_val_losses,
             color='#90CAF9',  # 浅蓝色
             label='因果自注意力-验证损失',
             linewidth=2,
             marker='d',
             markersize=4,
             markerfacecolor='#90CAF9',
             markeredgecolor='#90CAF9',
             markevery=5)
    
    # 设置坐标轴范围
    ax1.set_ylim(6, 11)  # 上半部分显示6-11区间
    ax2.set_ylim(4, 6)   # 下半部分显示4-6区间（更大的比例尺）
    
    # 两个子图都设置相同的x轴范围
    ax1.set_xlim(-50, 2050)
    ax2.set_xlim(-50, 2050)
    
    # 添加初始损失参考线
    ax1.axhline(y=naive_val_losses[0], color='gray', linestyle='--', alpha=0.7)
    ax1.text(1300, naive_val_losses[0]-0.1, f'初始损失: {naive_val_losses[0]:.2f}', 
             color='gray', fontsize=10, ha='left', va='top')
    
    # 在下图添加最终损失参考线
    ax2.axhline(y=naive_val_losses[-1], color='#F2A058', linestyle='--', alpha=0.7)
    ax2.text(1700, naive_val_losses[-1]+0.05, f'原始注意力最终验证损失: {naive_val_losses[-1]:.2f}', 
             color='#F2A058', fontsize=10, ha='left', va='bottom')
    
    ax2.axhline(y=causal_val_losses[-1], color='#90CAF9', linestyle='--', alpha=0.7)
    ax2.text(1700, causal_val_losses[-1]+0.05, f'因果自注意力最终验证损失: {causal_val_losses[-1]:.2f}', 
             color='#90CAF9', fontsize=10, ha='left', va='bottom')
    
    # 标记500步时的差异
    step_500_idx = naive_steps.index(500)
    diff_500 = naive_val_losses[step_500_idx] - causal_val_losses[step_500_idx]
    mid_point = (naive_val_losses[step_500_idx] + causal_val_losses[step_500_idx]) / 2
    
    # 根据mid_point值确定在哪个子图中标注
    if mid_point >= 6:
        # 在上图添加比较箭头
        ax1.annotate('', 
                    xy=(500, causal_val_losses[step_500_idx]),
                    xytext=(500, naive_val_losses[step_500_idx]),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax1.text(520, mid_point, f'差异: {diff_500:.4f}',
                 fontsize=10, ha='left', va='center')
    else:
        # 在下图添加比较箭头
        ax2.annotate('', 
                    xy=(500, causal_val_losses[step_500_idx]),
                    xytext=(500, naive_val_losses[step_500_idx]),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax2.text(520, mid_point, f'差异: {diff_500:.4f}',
                 fontsize=10, ha='left', va='center')
    
    # 添加标题和轴标签
    fig.suptitle('因果自注意力机制对GPT语言模型训练的影响', fontsize=16, y=0.98)
    ax2.set_xlabel('训练步数', fontsize=12)
    ax1.set_ylabel('损失值', fontsize=12)
    ax2.set_ylabel('损失值', fontsize=12)
    
    # 在上图显示图例
    ax1.legend(loc='upper right',bbox_to_anchor=(0.999, 0.96), fontsize=10)
    
    # 设置x轴刻度
    ax1.set_xticks(range(0, 2001, 200))
    ax2.set_xticks(range(0, 2001, 200))
    
    # 只在下图显示x轴标签
    ax1.set_xticklabels([])
    
    # 使用不同的刻度密度
    ax1.yaxis.set_major_locator(MultipleLocator(1.0))
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    
    # 设置网格线
    ax1.grid(True, linestyle='--', alpha=0.2)
    ax2.grid(True, linestyle='--', alpha=0.2)
    
    # 添加断轴标记
    d = .005  # 断轴线的大小
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # 左下
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右下
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # 左上
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 右上
    
    # 添加性能参数信息
    param_text = ('实验参数设置:\n'
                 'n_layer=4, n_head=4, n_embd=256\n'
                 'batch_size=8, block_size=128\n'
                 'gradient_accumulation_steps=8\n\n'
                 '模型性能对比:\n'
                 f'原始注意力最终训练损失: {naive_train_losses[-1]:.4f}\n'
                 f'因果自注意力最终训练损失: {causal_train_losses[-1]:.4f}\n'
                 f'训练损失提升: {final_train_improvement:.4f} ({train_improvement_pct:.2f}%)\n\n'
                 f'原始注意力最终验证损失: {naive_val_losses[-1]:.4f}\n'
                 f'因果自注意力最终验证损失: {causal_val_losses[-1]:.4f}\n'
                 f'验证损失提升: {final_val_improvement:.4f} ({val_improvement_pct:.2f}%)'
                 )
    
    # 在下图的左侧添加性能信息
    ax1.text(0.81, 1.25, param_text,
             transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=10,
             color="#666666",
             )
    
    # 优化图表布局
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.04)  # 减少两个子图之间的间距
    
    # 保存图片
    plt.savefig('causal_attention_comparison_broken_axis.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    
    plt.show()

# 调用绘图函数
plot_attention_comparison_with_broken_axis()