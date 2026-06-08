import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

adhd = pd.read_excel('test.xlsx', sheet_name='ADHD').head(16)
non = pd.read_excel('test.xlsx', sheet_name='nonADHD').head(16)

conditions = ['A', 'B', 'C']
metrics = ['complete', 'quiz', 'quit', 'satisfaction']
metric_labels = ['완료율 (%)', '퀴즈 정답률 (%)', '이탈 횟수', '만족도 (1-5)']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ADHD vs nonADHD 그룹 비교', fontsize=16, fontweight='bold', y=0.98)

colors_adhd = '#E74C3C'
colors_non = '#3498DB'
x = np.arange(len(conditions))
width = 0.3

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx // 2][idx % 2]

    adhd_means = [adhd[f'{c}_{metric}'].mean() for c in conditions]
    non_means = [non[f'{c}_{metric}'].mean() for c in conditions]
    adhd_sems = [adhd[f'{c}_{metric}'].sem() for c in conditions]
    non_sems = [non[f'{c}_{metric}'].sem() for c in conditions]

    bars1 = ax.bar(x - width/2, adhd_means, width, label='ADHD', color=colors_adhd, alpha=0.85,
                   yerr=adhd_sems, capsize=4, error_kw={'linewidth': 1.2})
    bars2 = ax.bar(x + width/2, non_means, width, label='nonADHD', color=colors_non, alpha=0.85,
                   yerr=non_sems, capsize=4, error_kw={'linewidth': 1.2})

    for i, c in enumerate(conditions):
        t, p = stats.ttest_ind(adhd[f'{c}_{metric}'].dropna(), non[f'{c}_{metric}'].dropna())
        y_max = max(adhd_means[i] + adhd_sems[i], non_means[i] + non_sems[i])
        sig = ''
        if p < 0.001: sig = '***'
        elif p < 0.01: sig = '**'
        elif p < 0.05: sig = '*'
        if sig:
            ax.text(i, y_max * 1.02 + ax.get_ylim()[1]*0.02, sig, ha='center', fontsize=13, fontweight='bold')

    ax.set_xlabel('조건', fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'조건 {c}' for c in conditions], fontsize=10)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if metric in ['complete', 'quiz']:
        ax.set_ylim(0, 115)
    elif metric == 'satisfaction':
        ax.set_ylim(0, 6)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('adhd_comparison.png', dpi=200, bbox_inches='tight')
print('Saved: adhd_comparison.png')

# --- 조건별 개선도 (A→B→C) 비교 그래프 ---
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('조건 A → B → C 변화 추이 (ADHD vs nonADHD)', fontsize=14, fontweight='bold')

for idx, (metric, label) in enumerate(zip(['complete', 'quiz'], ['완료율 (%)', '퀴즈 정답률 (%)'])):
    ax = axes2[idx]
    adhd_vals = [adhd[f'{c}_{metric}'].mean() for c in conditions]
    non_vals = [non[f'{c}_{metric}'].mean() for c in conditions]
    adhd_errs = [adhd[f'{c}_{metric}'].sem() for c in conditions]
    non_errs = [non[f'{c}_{metric}'].sem() for c in conditions]

    ax.errorbar(conditions, adhd_vals, yerr=adhd_errs, marker='o', markersize=8,
                linewidth=2, color=colors_adhd, label='ADHD', capsize=5)
    ax.errorbar(conditions, non_vals, yerr=non_errs, marker='s', markersize=8,
                linewidth=2, color=colors_non, label='nonADHD', capsize=5)

    ax.set_xlabel('조건', fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig('adhd_trend.png', dpi=200, bbox_inches='tight')
print('Saved: adhd_trend.png')
