import matplotlib.pyplot as plt
import numpy as np

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

# 实验1：没有梯度累积 (gradient_accumulation_steps=1)
naive_experiment = """
step 0: train loss 10.8687, val loss 10.8612
step 50: train loss 8.6614, val loss 8.9377
step 100: train loss 9.5972, val loss 10.0618
step 150: train loss 10.5328, val loss 10.9315
step 200: train loss 10.6483, val loss 11.0833
step 250: train loss 10.8252, val loss 11.2894
step 300: train loss 10.8880, val loss 11.3613
step 350: train loss 10.9829, val loss 11.3378
step 400: train loss 11.0517, val loss 11.4744
step 450: train loss 11.2346, val loss 11.6310
step 500: train loss 11.1710, val loss 11.4906
step 550: train loss 11.3033, val loss 11.7154
step 600: train loss 11.5041, val loss 11.8897
step 650: train loss 11.7044, val loss 12.1023
step 700: train loss 11.6033, val loss 11.9441
step 750: train loss 11.5800, val loss 11.9206
step 800: train loss 11.7217, val loss 12.0807
step 850: train loss 11.8720, val loss 12.2190
step 900: train loss 11.7532, val loss 12.0825
step 950: train loss 11.7374, val loss 12.0701
step 1000: train loss 11.8302, val loss 12.1709
step 1050: train loss 11.8813, val loss 12.1523
step 1100: train loss 11.8738, val loss 12.1829
step 1150: train loss 12.0026, val loss 12.4131
step 1200: train loss 12.0670, val loss 12.3772
step 1250: train loss 12.1074, val loss 12.4054
step 1300: train loss 12.1954, val loss 12.5282
step 1350: train loss 12.1704, val loss 12.3990
step 1400: train loss 12.1784, val loss 12.4475
step 1450: train loss 12.4197, val loss 12.6230
step 1500: train loss 12.3797, val loss 12.6665
step 1550: train loss 12.4507, val loss 12.7866
step 1600: train loss 12.4796, val loss 12.7887
step 1650: train loss 12.4841, val loss 12.8198
step 1700: train loss 12.5191, val loss 12.8368
step 1750: train loss 12.5671, val loss 12.9021
step 1800: train loss 12.6172, val loss 12.9194
step 1850: train loss 12.7098, val loss 12.9565
step 1900: train loss 12.7207, val loss 12.9662
step 1950: train loss 12.7523, val loss 13.0501
step 2000: train loss 12.7623, val loss 13.0877
"""

# 实验2：使用梯度累积 (gradient_accumulation_steps=8)
accum_experiment = """
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

# 解析数据
naive_steps, naive_train_losses, naive_val_losses = parse_experiment_data(naive_experiment)
accum_steps, accum_train_losses, accum_val_losses = parse_experiment_data(accum_experiment)

# 创建绘图函数
def plot_gradient_accumulation_comparison(naive_steps, naive_train_losses, naive_val_losses,
                                         accum_steps, accum_train_losses, accum_val_losses):
    # 创建图表
    plt.figure(figsize=(14, 7), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # 计算损失下降曲线的稳定性（标准差）
    naive_train_std = np.std(naive_train_losses[5:]) 
    naive_val_std = np.std(naive_val_losses[5:])
    accum_train_std = np.std(accum_train_losses[5:])
    accum_val_std = np.std(accum_val_losses[5:])
    
    if naive_train_std > accum_train_std:
        stability_improvement = (1 - accum_train_std/naive_train_std) * 100
        stability_text = f"稳定性提升: {stability_improvement:.2f}%"
    else:
        stability_change = (accum_train_std/naive_train_std - 1) * 100
        stability_text = f"稳定性变化: {stability_change:.2f}%"
    
    # 绘制四条曲线
    # 无梯度累积-训练损失
    plt.plot(naive_steps, naive_train_losses,
             color='#0000FF',  # 蓝色
             label='无梯度累积-训练损失',
             linewidth=2,
             marker='o',
             markersize=4,
             markerfacecolor='#0000FF',
             markeredgecolor='#0000FF',
             markevery=5)  # 减少标记点，每5个点标记一次
    
    # 无梯度累积-验证损失
    plt.plot(naive_steps, naive_val_losses,
             color='#4169E1',  # 皇家蓝
             label='无梯度累积-验证损失',
             linewidth=2,
             marker='s',
             markersize=4,
             markerfacecolor='#4169E1',
             markeredgecolor='#4169E1',
             markevery=5)
    
    # 梯度累积-训练损失
    plt.plot(accum_steps, accum_train_losses,
             color='#228B22',  # 森林绿
             label='梯度累积-训练损失',
             linewidth=2,
             marker='^',
             markersize=4,
             markerfacecolor='#228B22',
             markeredgecolor='#228B22',
             markevery=5)
    
    # 梯度累积-验证损失
    plt.plot(accum_steps, accum_val_losses,
             color='#2E8B57',  # 海洋绿
             label='梯度累积-验证损失',
             linewidth=2,
             marker='d',
             markersize=4,
             markerfacecolor='#2E8B57',
             markeredgecolor='#2E8B57',
             markevery=5)
    
    # 添加初始和最终损失的水平虚线
    # 初始损失（两者相同）
    plt.axhline(y=naive_val_losses[0], color='gray', linestyle='--', alpha=0.7)
    plt.text(1800, naive_val_losses[0]+0.1, f'初始损失: {naive_val_losses[0]:.2f}', 
             color='gray', fontsize=10, ha='left', va='bottom')
    
    # 最终无梯度累积验证损失
    plt.axhline(y=naive_val_losses[-1], color='#4169E1', linestyle='--', alpha=0.7)
    plt.text(1700, naive_val_losses[-1]+0.1, f'无梯度累积最终验证损失: {naive_val_losses[-1]:.2f}', 
             color='#4169E1', fontsize=10, ha='left', va='bottom')
    
    # 最终梯度累积验证损失
    plt.axhline(y=accum_val_losses[-1], color='#2E8B57', linestyle='--', alpha=0.7)
    plt.text(1700, accum_val_losses[-1]-0.4, f'梯度累积最终验证损失: {accum_val_losses[-1]:.2f}', 
             color='#2E8B57', fontsize=10, ha='left', va='bottom')
    
    # 设置标题和标签
    plt.title('梯度累积对GPT语言模型训练的影响',
              fontsize=16, 
              pad=20)
    
    plt.xlabel('训练步数', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    
    # 添加图例在左上角
    plt.legend(loc='upper left', fontsize=10)
    
    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.2)

    # 在右侧中央添加参数信息
    param_text = ('实验参数设置:\n'
                 'n_layer=4, n_head=4, n_embd=256\n'
                 'batch_size=8, block_size=128\n'
                 'learning_rate=6e-4\n'
                 'eval_interval=50, eval_iters=50\n\n'
                 '无梯度累积: gradient_accumulation_steps=1\n'
                 '梯度累积: gradient_accumulation_steps=8\n\n'
                 '训练曲线稳定性(第5个点之后):\n'
                 f'无梯度累积训练标准差: {naive_train_std:.4f}\n'
                 f'梯度累积训练标准差: {accum_train_std:.4f}\n'
                 f'{stability_text}'
                 )
    
    plt.text(0.75, 0.25, param_text,
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=10,
             color="#666666",
             )
    
    # 设置坐标轴范围，扩展x轴以容纳文本
    plt.xlim(-50, 2050)
    plt.ylim(4, 14)  
    
    # 设置x轴刻度
    plt.xticks(range(0, 2001, 200))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('gradient_accumulation_comparison.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    
    plt.show()
    
    # 打印关键分析结果
    print("=== 实验结果对比 ===")
    print("\n无梯度累积:")
    print(f"初始训练损失: {naive_train_losses[0]:.4f}, 初始验证损失: {naive_val_losses[0]:.4f}")
    print(f"最终训练损失: {naive_train_losses[-1]:.4f}, 最终验证损失: {naive_val_losses[-1]:.4f}")
    print(f"最低训练损失: {min(naive_train_losses):.4f}, 最低验证损失: {min(naive_val_losses):.4f}")

    print("\n梯度累积:")
    print(f"初始训练损失: {accum_train_losses[0]:.4f}, 初始验证损失: {accum_val_losses[0]:.4f}")
    print(f"最终训练损失: {accum_train_losses[-1]:.4f}, 最终验证损失: {accum_val_losses[-1]:.4f}")
    print(f"最低训练损失: {min(accum_train_losses):.4f}, 最低验证损失: {min(accum_val_losses):.4f}")

    print("\n收敛性能提升:")
    print(f"训练损失差异: {naive_train_losses[-1] - accum_train_losses[-1]:.4f}")
    print(f"验证损失差异: {naive_val_losses[-1] - accum_val_losses[-1]:.4f}")

    print(f"\n训练曲线稳定性 (标准差):")
    print(f"无梯度累积训练标准差: {naive_train_std:.4f}")
    print(f"梯度累积训练标准差: {accum_train_std:.4f}")
    print(f"无梯度累积验证标准差: {naive_val_std:.4f}")
    print(f"梯度累积验证标准差: {accum_val_std:.4f}")
    
    if naive_train_std > accum_train_std:
        print(f"训练稳定性提升: {(1 - accum_train_std/naive_train_std)*100:.2f}%")
    else:
        print(f"训练稳定性变化: {(accum_train_std/naive_train_std - 1)*100:.2f}%")

# 调用绘图函数
plot_gradient_accumulation_comparison(naive_steps, naive_train_losses, naive_val_losses,
                                    accum_steps, accum_train_losses, accum_val_losses)