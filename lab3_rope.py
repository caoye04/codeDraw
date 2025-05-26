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

# 实验3：标准位置编码（Absolute Position Embedding）
naive_position_experiment = """
step 0: train loss 10.9455, val loss 10.9393
step 50: train loss 6.9422, val loss 6.1181
step 100: train loss 6.3389, val loss 5.6308
step 150: train loss 6.0167, val loss 5.3877
step 200: train loss 5.8298, val loss 5.2515
step 250: train loss 5.7321, val loss 5.1689
step 300: train loss 5.5972, val loss 5.0427
step 350: train loss 5.4711, val loss 4.9975
step 400: train loss 5.3834, val loss 4.9059
step 450: train loss 5.2072, val loss 4.8584
step 500: train loss 5.1690, val loss 4.8475
step 550: train loss 5.0770, val loss 4.6871
step 600: train loss 4.9966, val loss 4.7322
step 650: train loss 4.9436, val loss 4.6504
step 700: train loss 4.8544, val loss 4.6145
step 750: train loss 4.7970, val loss 4.5346
step 800: train loss 4.7999, val loss 4.5473
step 850: train loss 4.7277, val loss 4.4899
step 900: train loss 4.6679, val loss 4.4068
step 950: train loss 4.6435, val loss 4.4515
step 1000: train loss 4.5691, val loss 4.4060
step 1050: train loss 4.5461, val loss 4.3444
step 1100: train loss 4.5618, val loss 4.4044
step 1150: train loss 4.5230, val loss 4.3773
step 1200: train loss 4.4589, val loss 4.3113
step 1250: train loss 4.4076, val loss 4.3543
step 1300: train loss 4.3561, val loss 4.3246
step 1350: train loss 4.3843, val loss 4.3034
step 1400: train loss 4.3101, val loss 4.3189
step 1450: train loss 4.3308, val loss 4.2536
step 1500: train loss 4.2461, val loss 4.2929
step 1550: train loss 4.2189, val loss 4.2083
step 1600: train loss 4.2372, val loss 4.2740
step 1650: train loss 4.2541, val loss 4.3002
step 1700: train loss 4.1463, val loss 4.2354
step 1750: train loss 4.2132, val loss 4.2163
step 1800: train loss 4.1521, val loss 4.2311
step 1850: train loss 4.1513, val loss 4.1745
step 1900: train loss 4.0683, val loss 4.1788
step 1950: train loss 4.0894, val loss 4.1860
step 2000: train loss 4.0983, val loss 4.1668
step 2050: train loss 4.0741, val loss 4.1577
step 2100: train loss 4.0147, val loss 4.1232
step 2150: train loss 4.0283, val loss 4.0747
step 2200: train loss 3.9467, val loss 4.0794
step 2250: train loss 3.9866, val loss 4.1750
step 2300: train loss 3.8887, val loss 4.1293
step 2350: train loss 3.9563, val loss 4.0918
step 2400: train loss 3.9160, val loss 4.0921
step 2450: train loss 3.8699, val loss 4.1425
step 2500: train loss 3.9941, val loss 4.0736
step 2550: train loss 3.8926, val loss 4.0992
step 2600: train loss 3.8815, val loss 4.0890
step 2650: train loss 3.8377, val loss 4.0955
step 2700: train loss 3.8966, val loss 4.0574
step 2750: train loss 3.8170, val loss 4.0415
step 2800: train loss 3.8542, val loss 4.0491
step 2850: train loss 3.8319, val loss 4.0531
step 2900: train loss 3.8269, val loss 4.0670
step 2950: train loss 3.7905, val loss 4.0091
step 3000: train loss 3.7817, val loss 4.0031
step 3050: train loss 3.7596, val loss 4.1076
step 3100: train loss 3.7587, val loss 4.0176
step 3150: train loss 3.7896, val loss 3.9710
step 3200: train loss 3.7083, val loss 3.9989
step 3250: train loss 3.7531, val loss 4.0281
step 3300: train loss 3.6688, val loss 3.9744
step 3350: train loss 3.7595, val loss 3.9656
step 3400: train loss 3.7041, val loss 4.0241
step 3450: train loss 3.7197, val loss 3.9633
step 3500: train loss 3.7104, val loss 3.9638
step 3550: train loss 3.6615, val loss 3.9822
step 3600: train loss 3.6893, val loss 4.0166
step 3650: train loss 3.6431, val loss 3.9692
step 3700: train loss 3.6705, val loss 3.9320
step 3750: train loss 3.6507, val loss 3.9832
step 3800: train loss 3.6378, val loss 3.9518
step 3850: train loss 3.6349, val loss 3.9343
step 3900: train loss 3.6565, val loss 3.9158
step 3950: train loss 3.5764, val loss 3.9777
step 4000: train loss 3.6038, val loss 3.9689
step 4050: train loss 3.6143, val loss 3.9454
step 4100: train loss 3.6278, val loss 3.9259
step 4150: train loss 3.6197, val loss 3.9279
step 4200: train loss 3.5717, val loss 3.9459
step 4250: train loss 3.5220, val loss 3.9271
step 4300: train loss 3.6055, val loss 3.9949
step 4350: train loss 3.5935, val loss 3.9140
step 4400: train loss 3.5356, val loss 3.9206
step 4450: train loss 3.5806, val loss 3.9726
step 4500: train loss 3.5661, val loss 3.9143
step 4550: train loss 3.5447, val loss 3.9244
step 4600: train loss 3.5922, val loss 3.8771
step 4650: train loss 3.5541, val loss 3.8494
step 4700: train loss 3.5456, val loss 3.9636
step 4750: train loss 3.5464, val loss 3.9547
step 4800: train loss 3.5855, val loss 3.9049
step 4850: train loss 3.5383, val loss 3.9214
step 4900: train loss 3.5396, val loss 3.9005
step 4950: train loss 3.5125, val loss 3.8836
step 5000: train loss 3.5618, val loss 3.8967
"""

# 实验3：RoPE位置编码
rope_position_experiment = """
step 0: train loss 10.9134, val loss 10.8852
step 50: train loss 6.8501, val loss 6.0032
step 100: train loss 6.1983, val loss 5.5790
step 150: train loss 5.8455, val loss 5.2754
step 200: train loss 5.6569, val loss 5.1338
step 250: train loss 5.4524, val loss 5.0028
step 300: train loss 5.3144, val loss 4.9076
step 350: train loss 5.2020, val loss 4.8459
step 400: train loss 5.1265, val loss 4.7520
step 450: train loss 4.9618, val loss 4.6859
step 500: train loss 4.9484, val loss 4.6452
step 550: train loss 4.8678, val loss 4.6019
step 600: train loss 4.7936, val loss 4.5169
step 650: train loss 4.7583, val loss 4.5237
step 700: train loss 4.7325, val loss 4.4976
step 750: train loss 4.6471, val loss 4.4500
step 800: train loss 4.6244, val loss 4.4284
step 850: train loss 4.5508, val loss 4.4235
step 900: train loss 4.5627, val loss 4.4031
step 950: train loss 4.4933, val loss 4.3828
step 1000: train loss 4.4569, val loss 4.3675
step 1050: train loss 4.4472, val loss 4.3388
step 1100: train loss 4.3431, val loss 4.2489
step 1150: train loss 4.3457, val loss 4.3582
step 1200: train loss 4.2828, val loss 4.2172
step 1250: train loss 4.2678, val loss 4.2813
step 1300: train loss 4.2434, val loss 4.2074
step 1350: train loss 4.2331, val loss 4.2139
step 1400: train loss 4.2215, val loss 4.3104
step 1450: train loss 4.1614, val loss 4.2397
step 1500: train loss 4.2170, val loss 4.1798
step 1550: train loss 4.1910, val loss 4.2154
step 1600: train loss 4.1248, val loss 4.1915
step 1650: train loss 4.0961, val loss 4.1622
step 1700: train loss 4.0992, val loss 4.1380
step 1750: train loss 4.0479, val loss 4.2179
step 1800: train loss 4.0153, val loss 4.1364
step 1850: train loss 3.9903, val loss 4.1494
step 1900: train loss 3.9807, val loss 4.1418
step 1950: train loss 3.9307, val loss 4.1074
step 2000: train loss 3.9817, val loss 4.1442
step 2050: train loss 3.9301, val loss 4.0428
step 2100: train loss 3.8849, val loss 4.1044
step 2150: train loss 3.9086, val loss 4.0404
step 2200: train loss 3.8788, val loss 4.0825
step 2250: train loss 3.8975, val loss 4.0911
step 2300: train loss 3.8657, val loss 4.0686
step 2350: train loss 3.8465, val loss 4.0094
step 2400: train loss 3.8695, val loss 4.0312
step 2450: train loss 3.8104, val loss 4.0029
step 2500: train loss 3.8167, val loss 4.0262
step 2550: train loss 3.7770, val loss 4.0910
step 2600: train loss 3.7879, val loss 4.0022
step 2650: train loss 3.7501, val loss 4.0000
step 2700: train loss 3.7758, val loss 3.9675
step 2750: train loss 3.7816, val loss 4.0192
step 2800: train loss 3.7053, val loss 4.0351
step 2850: train loss 3.7316, val loss 4.0450
step 2900: train loss 3.7107, val loss 3.9708
step 2950: train loss 3.6584, val loss 3.9544
step 3000: train loss 3.6833, val loss 3.9919
step 3050: train loss 3.6640, val loss 3.9803
step 3100: train loss 3.6613, val loss 4.0053
step 3150: train loss 3.6478, val loss 3.9594
step 3200: train loss 3.6396, val loss 3.9496
step 3250: train loss 3.6617, val loss 3.9530
step 3300: train loss 3.6457, val loss 3.9573
step 3350: train loss 3.5653, val loss 3.9896
step 3400: train loss 3.6229, val loss 3.9104
step 3450: train loss 3.6030, val loss 3.9324
step 3500: train loss 3.6062, val loss 3.9169
step 3550: train loss 3.5364, val loss 3.9296
step 3600: train loss 3.5964, val loss 3.8600
step 3650: train loss 3.5540, val loss 3.9112
step 3700: train loss 3.5714, val loss 3.9278
step 3750: train loss 3.5658, val loss 3.9385
step 3800: train loss 3.5548, val loss 3.8757
step 3850: train loss 3.5622, val loss 3.8629
step 3900: train loss 3.5308, val loss 3.8786
step 3950: train loss 3.5629, val loss 3.9325
step 4000: train loss 3.5001, val loss 3.9054
step 4050: train loss 3.5362, val loss 3.8887
step 4100: train loss 3.5174, val loss 3.8705
step 4150: train loss 3.4808, val loss 3.8663
step 4200: train loss 3.5641, val loss 3.8966
step 4250: train loss 3.4897, val loss 3.8764
step 4300: train loss 3.4565, val loss 3.8998
step 4350: train loss 3.4854, val loss 3.8928
step 4400: train loss 3.4614, val loss 3.8238
step 4450: train loss 3.4492, val loss 3.8877
step 4500: train loss 3.4556, val loss 3.8548
step 4550: train loss 3.4939, val loss 3.8246
step 4600: train loss 3.4199, val loss 3.8818
step 4650: train loss 3.4584, val loss 3.8324
step 4700: train loss 3.4490, val loss 3.8885
step 4750: train loss 3.4572, val loss 3.9002
step 4800: train loss 3.4471, val loss 3.8529
step 4850: train loss 3.4132, val loss 3.8817
step 4900: train loss 3.4462, val loss 3.8659
step 4950: train loss 3.4515, val loss 3.9219
step 5000: train loss 3.4169, val loss 3.8175
"""

# 解析数据
naive_steps, naive_train_losses, naive_val_losses = parse_experiment_data(naive_position_experiment)
rope_steps, rope_train_losses, rope_val_losses = parse_experiment_data(rope_position_experiment)

# 方法：使用分段坐标轴展示位置编码对比
def plot_position_encoding_comparison_with_broken_axis():
    # 创建带有分段坐标轴的图表
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    
    # 创建两个子图，保持原有的显示区间
    ax1 = plt.subplot(2, 1, 1)  # 上面的子图 (4.5-11)
    ax2 = plt.subplot(2, 1, 2)  # 下面的子图 (3.4-4.5)
    
    # 计算性能指标
    final_train_improvement = (naive_train_losses[-1] - rope_train_losses[-1])
    final_val_improvement = (naive_val_losses[-1] - rope_val_losses[-1])
    train_improvement_pct = (final_train_improvement / naive_train_losses[-1]) * 100
    val_improvement_pct = (final_val_improvement / naive_val_losses[-1]) * 100
    
    # 优化配色方案
    colors = {
        'naive_train': '#E74C3C',  # 红色
        'naive_val': '#C0392B',    # 深红色
        'rope_train': '#3498DB',   # 蓝色
        'rope_val': '#2980B9',     # 深蓝色
        'grid': '#CCCCCC',         # 网格线颜色
        'annotation': '#666666'    # 注释文字颜色
    }
    
    # 在上半部分绘制曲线 (4.5-11区间)
    # 标准位置编码-训练损失
    l1 = ax1.plot(naive_steps, naive_train_losses,
             color=colors['naive_train'],
             label='标准位置编码-训练损失',
             linewidth=2.5,
             marker='o',
             markersize=4,
             markerfacecolor=colors['naive_train'],
             markeredgecolor=colors['naive_train'],
             markevery=10)
    
    
    # 标准位置编码-验证损失
    l2 = ax1.plot(naive_steps, naive_val_losses,
             color=colors['naive_val'],
             label='标准位置编码-验证损失',
             linewidth=2.5,
             marker='s',
             markersize=4,
             markerfacecolor=colors['naive_val'],
             markeredgecolor=colors['naive_val'],
             markevery=10)
    
    
    # RoPE位置编码-训练损失
    l3 = ax1.plot(rope_steps, rope_train_losses,
             color=colors['rope_train'],
             label='RoPE位置编码-训练损失',
             linewidth=2.5,
             marker='^',
             markersize=4,
             markerfacecolor=colors['rope_train'],
             markeredgecolor=colors['rope_train'],
             markevery=10)
    
    
    # RoPE位置编码-验证损失
    l4 = ax1.plot(rope_steps, rope_val_losses,
             color=colors['rope_val'],
             label='RoPE位置编码-验证损失',
             linewidth=2.5,
             marker='d',
             markersize=4,
             markerfacecolor=colors['rope_val'],
             markeredgecolor=colors['rope_val'],
             markevery=10)
    
    
    # 在下半部分绘制相同的曲线 (3.4-4.5区间)
    for curve in [l1, l2, l3, l4]:
        ax2.plot(curve[0].get_xdata(), 
                curve[0].get_ydata(),
                color=curve[0].get_color(),
                linewidth=2.5,
                marker=curve[0].get_marker(),
                markersize=4,
                markerfacecolor=curve[0].get_markerfacecolor(),
                markeredgecolor=curve[0].get_markeredgecolor(),
                markevery=10)
        

    
    # 设置坐标轴范围
    ax1.set_ylim(4.5, 11)
    ax2.set_ylim(3.4, 4.5)
    ax1.set_xlim(-100, 5100)
    ax2.set_xlim(-100, 5100)
    
    # 添加网格线
    ax1.grid(True, linestyle='--', alpha=0.2, color=colors['grid'])
    ax2.grid(True, linestyle='--', alpha=0.2, color=colors['grid'])
    
    # 添加初始损失参考线
    ax1.axhline(y=naive_val_losses[0], color='gray', linestyle='--', alpha=0.7)
    ax1.text(3500, naive_val_losses[0]-0.1, 
             f'初始损失: {naive_val_losses[0]:.2f}', 
             color='gray', 
             fontsize=10,
             ha='left',
             va='top')
    
    # 在下图添加最终损失参考线
    ax2.axhline(y=naive_val_losses[-1], color=colors['naive_val'], linestyle='--', alpha=0.7)
    ax2.text(4200, naive_val_losses[-1]+0.1, 
             f'标准位置编码最终验证损失: {naive_val_losses[-1]:.2f}', 
             color=colors['naive_val'],
             fontsize=12,
             ha='left',
             va='bottom')
    
    ax2.axhline(y=rope_val_losses[-1], color=colors['rope_val'], linestyle='--', alpha=0.7)
    ax2.text(4200, rope_val_losses[-1]-0.08, 
             f'RoPE位置编码最终验证损失: {rope_val_losses[-1]:.2f}', 
             color=colors['rope_val'],
             fontsize=12,
             ha='left',
             va='bottom')
    
    # 添加标题和轴标签
    fig.suptitle('RoPE位置编码与标准位置编码在GPT训练中的对比', 
                 fontsize=16, 
                 y=0.98)
    ax2.set_xlabel('训练步数', fontsize=12)
    ax1.set_ylabel('损失值', fontsize=12)
    ax2.set_ylabel('损失值', fontsize=12)
    
    # 在上图显示图例
    ax1.legend(loc='upper right', 
              bbox_to_anchor=(0.999, 0.96), 
              fontsize=10,
              frameon=True,
              fancybox=True,
              shadow=True)
    
    # 设置x轴刻度
    ax1.set_xticks(range(0, 5001, 500))
    ax2.set_xticks(range(0, 5001, 500))
    ax1.set_xticklabels([])
    
    # 使用不同的刻度密度
    ax1.yaxis.set_major_locator(MultipleLocator(1.0))
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    
    # 添加断轴标记
    d = .005
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    # 添加RoPE架构示意
    rope_text = ('RoPE位置编码原理:\n'
                '对每个token的特征向量\n'
                '按照位置进行旋转变换，\n'
                '使相对位置信息被编码进向量中\n'
                '有利于处理长序列')
    
    ax1.text(0.1, 0.4, rope_text,
             transform=ax1.transAxes,
             bbox=dict(facecolor='#E3F2FD', 
                      alpha=0.8, 
                      edgecolor=colors['rope_train'], 
                      boxstyle='round,pad=0.5'),
             fontsize=12,
             color=colors['annotation'])
    
    # 找出每个模型的最低验证损失
    min_naive_val = min(naive_val_losses)
    min_rope_val = min(rope_val_losses)
    min_naive_idx = naive_val_losses.index(min_naive_val)
    min_rope_idx = rope_val_losses.index(min_rope_val)
    
    # 计算收敛速度
    threshold_val = naive_val_losses[-1]
    for idx, val_loss in enumerate(rope_val_losses):
        if val_loss <= threshold_val:
            steps_to_threshold = rope_steps[idx]
            speedup = ((5000 - steps_to_threshold) / 5000) * 100
            break
            
    # 添加性能参数信息
    param_text = (
        '实验参数设置:\n'
        'n_layer=8, n_head=8, n_embd=512\n'
        'batch_size=8, block_size=256\n'
        'gradient_accumulation_steps=8\n'
        'max_iters=5000\n\n'
        '位置编码对比:\n'
        '标准位置编码使用可学习的嵌入层\n'
        'RoPE使用旋转位置编码\n\n'
        f'收敛速度分析:\n'
        f'RoPE在第{steps_to_threshold}步达到naive的最终性能\n'
        f'收敛速度提升约{speedup:.1f}%'
    )
    
    # 在图的右侧添加性能信息
    ax1.text(0.83, 1.35, param_text,
             transform=ax2.transAxes,
             bbox=dict(facecolor='white', 
                      alpha=0.8, 
                      edgecolor='gray'),
             fontsize=10,
             color=colors['annotation'])
    
    # 优化图表布局
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.04)
    
    # 保存图片
    plt.savefig('position_encoding_comparison_broken_axis.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    
    plt.show()
    
    # 打印关键分析结果
    print("=== 位置编码对比实验结果 ===")
    print("\n标准位置编码:")
    print(f"初始训练损失: {naive_train_losses[0]:.4f}, 初始验证损失: {naive_val_losses[0]:.4f}")
    print(f"最终训练损失: {naive_train_losses[-1]:.4f}, 最终验证损失: {naive_val_losses[-1]:.4f}")
    print(f"最低训练损失: {min(naive_train_losses):.4f}, 最低验证损失: {min_naive_val:.4f} (步数: {naive_steps[min_naive_idx]})")

    print("\nRoPE位置编码:")
    print(f"初始训练损失: {rope_train_losses[0]:.4f}, 初始验证损失: {rope_val_losses[0]:.4f}")
    print(f"最终训练损失: {rope_train_losses[-1]:.4f}, 最终验证损失: {rope_val_losses[-1]:.4f}")
    print(f"最低训练损失: {min(rope_train_losses):.4f}, 最低验证损失: {min_rope_val:.4f} (步数: {rope_steps[min_rope_idx]})")

    print("\n性能提升:")
    print(f"训练损失绝对提升: {final_train_improvement:.4f} ({train_improvement_pct:.2f}%)")
    print(f"验证损失绝对提升: {final_val_improvement:.4f} ({val_improvement_pct:.2f}%)")
    print(f"\n收敛速度分析:")
    print(f"RoPE在第{steps_to_threshold}步达到标准位置编码的最终性能")
    print(f"收敛速度提升约{speedup:.1f}%")


# 调用绘图函数
plot_position_encoding_comparison_with_broken_axis()