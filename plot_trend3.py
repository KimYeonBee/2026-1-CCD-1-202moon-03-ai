import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams['font.family'] = 'Apple SD Gothic Neo'
plt.rcParams['axes.unicode_minus'] = False

adhd = pd.read_excel('test.xlsx', sheet_name='ADHD').head(16)
non = pd.read_excel('test.xlsx', sheet_name='nonADHD').head(16)

conditions = ['A', 'B', 'C']
metrics = ['complete', 'quiz', 'quit', 'satisfaction']
metric_labels = ['완료율 (%)', '퀴즈 정답률 (%)', '이탈 횟수', '만족도']

c_adhd = '#1A9AF5'
c_non = '#BBBBBB'

def draw_panel(ax, metric, label):
    ax.set_facecolor('white')
    adhd_vals = [adhd[f'{c}_{metric}'].mean() for c in conditions]
    non_vals = [non[f'{c}_{metric}'].mean() for c in conditions]
    adhd_errs = [adhd[f'{c}_{metric}'].sem() for c in conditions]
    non_errs = [non[f'{c}_{metric}'].sem() for c in conditions]

    all_vals = adhd_vals + non_vals
    val_min, val_max = min(all_vals), max(all_vals)
    span = val_max - val_min
    base = span if span > 0 else max(val_max * 0.12, 0.6)
    # 바닥 여백을 더 줘서 최저점이 축 라벨과 겹치지 않게
    y_lo = val_min - base * 0.55
    y_hi = val_max + base * 0.42
    yrange = y_hi - y_lo

    # 밴드
    ax.fill_between(range(3), [v-e for v,e in zip(non_vals,non_errs)],
                    [v+e for v,e in zip(non_vals,non_errs)], color=c_non, alpha=0.22, lw=0)
    ax.fill_between(range(3), [v-e for v,e in zip(adhd_vals,adhd_errs)],
                    [v+e for v,e in zip(adhd_vals,adhd_errs)], color=c_adhd, alpha=0.13, lw=0)
    # 선
    ax.plot(range(3), non_vals, color=c_non, lw=3.5, marker='o', ms=15,
            markeredgecolor='white', markeredgewidth=2.5, zorder=3, solid_capstyle='round')
    ax.plot(range(3), adhd_vals, color=c_adhd, lw=3.5, marker='o', ms=15,
            markeredgecolor='white', markeredgewidth=2.5, zorder=4, solid_capstyle='round')

    # 값 라벨 — 두 선이 가까우면 강제로 위/아래 분리
    for i in range(3):
        av, nv = adhd_vals[i], non_vals[i]
        a_up = av >= nv
        a_off = (0, 14) if a_up else (0, -21)
        n_off = (0, -21) if a_up else (0, 14)
        ax.annotate(f'{av:.1f}', (i, av), xytext=a_off, textcoords='offset points',
                    ha='center', fontsize=13.5, fontweight='bold', color=c_adhd, zorder=6)
        ax.annotate(f'{nv:.1f}', (i, nv), xytext=n_off, textcoords='offset points',
                    ha='center', fontsize=13.5, fontweight='bold', color='#888888', zorder=6)

    # 끝점 둥근 라벨 박스 — 가까우면 위아래로 벌림
    end_a, end_n = adhd_vals[2], non_vals[2]
    min_gap = yrange * 0.14
    if abs(end_a - end_n) < min_gap:
        mid = (end_a + end_n) / 2
        ya, yn = mid + min_gap/2, mid - min_gap/2
    else:
        ya, yn = end_a, end_n
    # 핀이 바닥 축 라벨이나 천장과 겹치지 않게 제한
    lo_clamp, hi_clamp = y_lo + yrange*0.14, y_hi - yrange*0.05
    if abs(ya - yn) < min_gap*0.95:
        pass
    ya = min(max(ya, lo_clamp), hi_clamp)
    yn = min(max(yn, lo_clamp), hi_clamp)
    if abs(ya - yn) < min_gap*0.9:  # 클램프로 다시 붙었으면 벌림
        if ya >= yn:
            ya, yn = yn + min_gap, yn
        else:
            yn, ya = ya + min_gap, ya
        ya = min(ya, hi_clamp); yn = min(yn, hi_clamp)
    def pill(y, text, color, tcolor='white'):
        ax.annotate(text, (2, y), xytext=(24, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=12.5, fontweight='bold', color=tcolor,
                    zorder=7, annotation_clip=False,
                    bbox=dict(boxstyle='round,pad=0.5', fc=color, ec='none'))
    pill(ya, 'ADHD', c_adhd)
    pill(yn, 'nonADHD', c_non, tcolor='#555555')

    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-0.35, 2.55)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['조건 A', '조건 B', '조건 C'], fontsize=14, fontweight='bold', color='#444444')
    ax.set_title(label, fontsize=18, fontweight='bold', pad=16, loc='left', color='#222222')
    for s in ['top','right','left']:
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(length=0, colors='#888888')
    ax.set_yticks([])
    ax.yaxis.grid(True, color='#F0F0F0', lw=1.2)
    ax.set_axisbelow(True)

# 각 항목 = 분리된 블록(카드)
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.patch.set_facecolor('#ECECEC')
fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.06, wspace=0.20, hspace=0.40)

for idx, (m, l) in enumerate(zip(metrics, metric_labels)):
    draw_panel(axes[idx//2][idx%2], m, l)

# 카드 배경 (각 패널을 둥근 흰 블록으로)
fig.canvas.draw()
for ax in axes.flat:
    bb = ax.get_position()
    pad_x, pad_y = 0.025, 0.045
    card = FancyBboxPatch((bb.x0 - pad_x, bb.y0 - pad_y),
                          bb.width + pad_x*1.5, bb.height + pad_y*1.6,
                          boxstyle='round,pad=0.01,rounding_size=0.02',
                          mutation_aspect=1.3,
                          fc='white', ec='none', zorder=-1,
                          transform=fig.transFigure)
    fig.patches.append(card)

fig.suptitle('ADHD vs nonADHD  조건별 변화 추이', fontsize=24, fontweight='bold',
             color='#1A1A1A', x=0.05, ha='left', y=0.965)

# 범례 (제목 오른쪽 상단)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color=c_adhd, lw=4, marker='o', markersize=13,
           markeredgecolor='white', markeredgewidth=2.5, label='ADHD'),
    Line2D([0], [0], color=c_non, lw=4, marker='o', markersize=13,
           markeredgecolor='white', markeredgewidth=2.5, label='nonADHD'),
]
leg = fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.95, 0.975),
                 ncol=2, fontsize=15, frameon=True, handletextpad=0.5, columnspacing=1.5,
                 borderpad=0.8, edgecolor='#DDDDDD', facecolor='white')
leg.get_frame().set_boxstyle('round,pad=0.4,rounding_size=0.5')
for t in leg.get_texts():
    t.set_fontweight('bold')
    t.set_color('#444444')

plt.savefig('adhd_trend3.png', dpi=200, facecolor='#ECECEC', bbox_inches='tight')
print('Saved')
