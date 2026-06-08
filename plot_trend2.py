import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

adhd = pd.read_excel('test.xlsx', sheet_name='ADHD').head(16)
non = pd.read_excel('test.xlsx', sheet_name='nonADHD').head(16)

conditions = ['A', 'B', 'C']
metrics = ['complete', 'quiz', 'quit', 'satisfaction']
metric_labels = ['완료율 (%)', '퀴즈 정답률 (%)', '이탈 횟수', '만족도']

c_adhd = '#1A9AF5'
c_non = '#C8C8C8'

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.patch.set_facecolor('#FAFAFA')

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx // 2][idx % 2]
    ax.set_facecolor('#FAFAFA')

    adhd_vals = [adhd[f'{c}_{metric}'].mean() for c in conditions]
    non_vals = [non[f'{c}_{metric}'].mean() for c in conditions]
    adhd_errs = [adhd[f'{c}_{metric}'].sem() for c in conditions]
    non_errs = [non[f'{c}_{metric}'].sem() for c in conditions]

    all_vals = adhd_vals + non_vals
    val_min = min(all_vals)
    val_max = max(all_vals)
    margin = (val_max - val_min) * 0.35
    if margin < 0.3:
        margin = max(val_max * 0.05, 0.3)
    y_lo = max(0, val_min - margin)
    y_hi = val_max + margin

    # nonADHD (뒤에, 연한 회색)
    ax.fill_between(range(3),
                    [v - e for v, e in zip(non_vals, non_errs)],
                    [v + e for v, e in zip(non_vals, non_errs)],
                    color=c_non, alpha=0.25, linewidth=0)
    ax.plot(range(3), non_vals, color=c_non, linewidth=3, marker='o', markersize=14,
            markeredgecolor='white', markeredgewidth=2, zorder=3, label='nonADHD')

    # ADHD (앞에, 파란색)
    ax.fill_between(range(3),
                    [v - e for v, e in zip(adhd_vals, adhd_errs)],
                    [v + e for v, e in zip(adhd_vals, adhd_errs)],
                    color=c_adhd, alpha=0.15, linewidth=0)
    ax.plot(range(3), adhd_vals, color=c_adhd, linewidth=3, marker='o', markersize=14,
            markeredgecolor='white', markeredgewidth=2.5, zorder=4, label='ADHD')

    # 값 표시
    for i, (av, nv) in enumerate(zip(adhd_vals, non_vals)):
        fmt = '.1f' if metric == 'satisfaction' else '.1f'
        ax.annotate(f'{av:{fmt}}', (i, av), textcoords='offset points', xytext=(0, 14),
                    ha='center', fontsize=9.5, fontweight='bold', color=c_adhd)
        offset_y = -18 if abs(av - nv) > (y_hi - y_lo) * 0.06 else 14
        if metric == 'quit' and i == 2:
            offset_y = -18
        ax.annotate(f'{nv:{fmt}}', (i, nv), textcoords='offset points', xytext=(0, offset_y),
                    ha='center', fontsize=9.5, fontweight='bold', color='#999999')

    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['조건 A', '조건 B', '조건 C'], fontsize=11)
    ax.set_title(label, fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=10, loc='best', framealpha=0.8, edgecolor='none',
              fancybox=True, borderpad=0.8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(colors='#888888', labelsize=10)
    ax.yaxis.grid(True, color='#EEEEEE', linewidth=0.8)
    ax.set_axisbelow(True)

plt.tight_layout(pad=2.0)
plt.savefig('adhd_trend2.png', dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
print('Saved')
