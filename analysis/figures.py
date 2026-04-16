"""
Figure generator for the Metacognitive Monitoring Battery paper.
Produces the six publication figures from outputs/per_track_data.json,
outputs/t6_data.json, and outputs/probe_adjusted_leaderboard.csv.
"""
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

# Style
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

# Data holders — populated by _load_data() at run time
PER_TRACK: dict = {}
T6: dict = {}
LEADERBOARD: dict = {}


def _load_data():
    """Load outputs/*.json and outputs/*.csv into module globals."""
    global PER_TRACK, T6, LEADERBOARD
    with open('outputs/per_track_data.json') as f:
        PER_TRACK = json.load(f)
    with open('outputs/t6_data.json') as f:
        T6 = json.load(f)
    LEADERBOARD = {}
    with open('outputs/probe_adjusted_leaderboard.csv') as f:
        for r in csv.DictReader(f):
            try:
                meta_rank = int(r['metacognition_rank'])
            except ValueError:
                meta_rank = -1  # R1 excluded
            LEADERBOARD[r['model']] = {
                'acc_rank': int(r['accuracy_rank']),
                'mean_wd': float(r['mean_withdraw_delta']),
                'meta_rank': meta_rank,
            }

# Profile assignments
PROFILE_A = ['Gemini 3 Flash', 'Gemini 2.5 Flash', 'Gemini 2.5 Pro', 'Gemini 3.1 Pro', 'Qwen 80B Think']
PROFILE_B = ['DeepSeek R1']
PROFILE_C = ['Claude Sonnet 4.6', 'Claude Haiku 4.5', 'Qwen Coder 480B', 'Qwen 80B Inst',
             'GPT-5.4', 'Qwen 235B', 'GPT-5.4 mini', 'Gemma 3 12B']

GPT_FAMILY = ['GPT-5.4 nano', 'GPT-5.4 mini', 'GPT-5.4']
QWEN_FAMILY = ['Qwen 80B Inst', 'Qwen 235B', 'Qwen Coder 480B']
GEMMA_FAMILY = ['Gemma 3 1B', 'Gemma 3 12B', 'Gemma 3 27B']

# Colours
COL_A = '#2E8B57'  # sea green
COL_B = '#4169E1'  # royal blue
COL_C = '#E67E22'  # orange
COL_GPT = '#8E44AD'  # purple
COL_QWEN = '#E74C3C'  # red
COL_GEMMA = '#16A085'  # teal


# ===================================================================
# FIGURE 1: Phenotypes — KEEP rates correct vs incorrect, 3 panels
# Show full Profile C set (8 models)
# ===================================================================
def fig1_phenotypes():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    def get_kc_ki(model):
        """Mean KEEP rate on correct, incorrect across T1-T5"""
        kcs, kis = [], []
        for t in ['T1','T2','T3','T4','T5']:
            if t in PER_TRACK.get(model, {}):
                kcs.append(PER_TRACK[model][t]['kc_rate'])
                kis.append(PER_TRACK[model][t]['ki_rate'])
        return (np.mean(kcs) if kcs else 0, np.mean(kis) if kis else 0)

    # Panel A: Profile A - Blanket Confidence
    ax = axes[0]
    a_models = PROFILE_A
    x = np.arange(len(a_models))
    w = 0.35
    correct_rates = [get_kc_ki(m)[0] for m in a_models]
    incorrect_rates = [get_kc_ki(m)[1] for m in a_models]
    ax.bar(x - w/2, correct_rates, w, label='KEEP | correct', color=COL_A, edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, incorrect_rates, w, label='KEEP | incorrect', color=COL_A, alpha=0.45, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('Gemini ', 'Gem ').replace('Qwen 80B Think', 'Qwen 80B\nThink') for m in a_models], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('KEEP rate (%)')
    ax.set_ylim(0, 105)
    ax.set_title('A: Blanket Confidence (n=5)\nMonitoring without control', fontsize=11)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.95)
    ax.axhline(95, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    ax.grid(axis='y', alpha=0.2)

    # Panel B: Profile B - Blanket Withdrawal
    ax = axes[1]
    b_models = PROFILE_B
    x = np.arange(len(b_models))
    correct_rates = [get_kc_ki(m)[0] for m in b_models]
    incorrect_rates = [get_kc_ki(m)[1] for m in b_models]
    ax.bar(x - w/2, correct_rates, w, label='KEEP | correct', color=COL_B, edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, incorrect_rates, w, label='KEEP | incorrect', color=COL_B, alpha=0.45, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['DeepSeek R1'], fontsize=9)
    ax.set_ylabel('KEEP rate (%)')
    ax.set_ylim(0, 105)
    ax.set_title('B: Blanket Withdrawal (n=1)\nControl without monitoring', fontsize=11)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax.axhline(10, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    ax.grid(axis='y', alpha=0.2)

    # Panel C: Profile C - Selective Sensitivity (all 8)
    ax = axes[2]
    c_models = sorted(PROFILE_C, key=lambda m: -LEADERBOARD[m]['mean_wd'])
    x = np.arange(len(c_models))
    correct_rates = [get_kc_ki(m)[0] for m in c_models]
    incorrect_rates = [get_kc_ki(m)[1] for m in c_models]
    ax.bar(x - w/2, correct_rates, w, label='KEEP | correct', color=COL_C, edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, incorrect_rates, w, label='KEEP | incorrect', color=COL_C, alpha=0.45, edgecolor='black', linewidth=0.5)
    short_labels = []
    for m in c_models:
        if m == 'Claude Sonnet 4.6': s = 'Sonnet'
        elif m == 'Claude Haiku 4.5': s = 'Haiku'
        elif m == 'Qwen Coder 480B': s = 'Qwen\nCoder'
        elif m == 'Qwen 80B Inst': s = 'Qwen\n80B Inst'
        elif m == 'Qwen 235B': s = 'Qwen\n235B'
        elif m == 'GPT-5.4': s = 'GPT-5.4'
        elif m == 'GPT-5.4 mini': s = 'GPT-5.4\nmini'
        elif m == 'Gemma 3 12B': s = 'Gemma\n12B'
        else: s = m
        short_labels.append(s)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel('KEEP rate (%)')
    ax.set_ylim(0, 105)
    ax.set_title('C: Selective Sensitivity (n=8)\nCoupled monitoring-control', fontsize=11)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.95)
    ax.grid(axis='y', alpha=0.2)

    fig.suptitle('Three metacognitive profiles: KEEP rates on correct vs incorrect items (T1–T5 mean)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / 'fig1_phenotypes.png')
    plt.close()
    print('Fig 1 saved')


# ===================================================================
# FIGURE 2: Inverted leaderboard slope chart
# ===================================================================
def fig2_slope():
    fig, ax = plt.subplots(figsize=(11, 11))

    # Sort
    by_acc = sorted(LEADERBOARD.items(), key=lambda x: x[1]['acc_rank'])
    # For meta rank, exclude R1
    meta_ranked = [(m, d) for m, d in LEADERBOARD.items() if m != 'DeepSeek R1']
    by_meta = sorted(meta_ranked, key=lambda x: -x[1]['mean_wd'])

    n = len(by_acc)
    n_meta = len(by_meta)

    # Position dict
    acc_pos = {m: i for i, (m, _) in enumerate(by_acc)}
    meta_pos = {m: i for i, (m, _) in enumerate(by_meta)}

    for m in LEADERBOARD:
        ay = -acc_pos[m]
        if m == 'DeepSeek R1':
            # Show but dashed at bottom
            my = -(n_meta + 0.5)
            color = '#A8C8E8'
            ls = '--'
            lw = 1.2
            alpha = 0.6
        else:
            my = -meta_pos[m]
            # Determine colour by rank shift
            shift = acc_pos[m] - meta_pos[m]
            if m in GPT_FAMILY:
                color = COL_GPT
                lw = 2.2
                alpha = 0.85
            elif shift >= 3:
                color = COL_A  # rises
                lw = 1.8
                alpha = 0.75
            elif shift <= -3:
                color = COL_C  # drops
                lw = 1.8
                alpha = 0.75
            else:
                color = '#888888'
                lw = 0.9
                alpha = 0.4
            ls = '-'

        ax.plot([0, 1], [ay, my], color=color, lw=lw, alpha=alpha, linestyle=ls, zorder=2)

    # Labels
    for m, d in by_acc:
        y = -acc_pos[m]
        wt = 'bold' if m in GPT_FAMILY else 'normal'
        ax.text(-0.04, y, f"{d['acc_rank']}. {m}", ha='right', va='center', fontsize=9, fontweight=wt)
        ax.text(0.03, y, f"{d['acc_rank']:.0f}", ha='left', va='center', fontsize=8, color='gray')

    for i, (m, d) in enumerate(by_meta):
        y = -i
        wt = 'bold' if m in GPT_FAMILY else 'normal'
        ax.text(1.04, y, f"{m} ({d['mean_wd']:+.1f}%)", ha='left', va='center', fontsize=9, fontweight=wt)

    # R1 row
    ax.text(1.04, -(n_meta + 0.5), 'DeepSeek R1 (excl.†)',
            ha='left', va='center', fontsize=9, style='italic', color='#7AAEDC')

    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(-(n + 1), 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.text(0, 0.7, 'Accuracy Rank', ha='center', fontsize=12, fontweight='bold')
    ax.text(1, 0.7, 'Metacognition Rank (WΔ)', ha='center', fontsize=12, fontweight='bold')

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color=COL_A, lw=2.5, label='Rises ≥3 ranks'),
        Line2D([0],[0], color=COL_C, lw=2.5, label='Drops ≥3 ranks'),
        Line2D([0],[0], color=COL_GPT, lw=2.5, label='GPT-5.4 family'),
    ]
    ax.legend(handles=legend_elems, loc='lower center', ncol=3, frameon=False,
              bbox_to_anchor=(0.5, -0.06), fontsize=10)

    ax.set_title('Accuracy rank vs metacognitive sensitivity rank across 20 models\n†R1 excluded from metacognition ranking (blanket withdrawal, 1–9% KEEP)',
                 fontsize=11, pad=20)

    plt.tight_layout()
    plt.savefig(OUT / 'fig2_slope.png')
    plt.close()
    print('Fig 2 saved')


# ===================================================================
# FIGURE 3: Retrospective vs prospective dissociation scatter
# ===================================================================
def fig3_dissociation():
    fig, ax = plt.subplots(figsize=(11, 7.5))

    for m in LEADERBOARD:
        if m not in T6: continue
        wd = LEADERBOARD[m]['mean_wd']
        direct = T6[m]['direct_rate']

        if m in PROFILE_A:
            color = COL_A
            marker = 'o'
            label_pheno = 'Profile A'
        elif m in PROFILE_B:
            color = COL_B
            marker = 's'
            label_pheno = 'Profile B'
        elif m in PROFILE_C:
            color = COL_C
            marker = 'o'
            label_pheno = 'Profile C'
        else:
            color = '#999'
            marker = 'o'
            label_pheno = 'Other'

        # GPT family overlay
        is_gpt = m in GPT_FAMILY
        edgecolor = COL_GPT if is_gpt else 'black'
        size = 180 if is_gpt else 130
        lw = 2.5 if is_gpt else 0.8
        if is_gpt:
            marker = 'D'

        ax.scatter(wd, direct, s=size, c=color, marker=marker, edgecolors=edgecolor,
                   linewidths=lw, zorder=3, alpha=0.85)

        # Selective labels with smarter placement
        if m in ['Claude Sonnet 4.6', 'Claude Haiku 4.5', 'GPT-5.4', 'GPT-5.4 mini',
                 'GPT-5.4 nano', 'Gemma 3 27B', 'Gemma 3 1B', 'DeepSeek R1', 'GLM-5',
                 'Gemini 3.1 Pro', 'Qwen Coder 480B', 'Gemma 3 12B', 'Qwen 80B Inst']:
            short_map = {
                'Claude Sonnet 4.6': 'Sonnet', 'Claude Haiku 4.5': 'Haiku',
                'GPT-5.4': 'GPT-5.4', 'GPT-5.4 mini': 'GPT-5.4 mini',
                'GPT-5.4 nano': 'GPT-5.4 nano', 'Gemma 3 27B': 'Gem 27B',
                'Gemma 3 1B': 'Gem 1B', 'Gemma 3 12B': 'Gem 12B',
                'DeepSeek R1': 'R1', 'GLM-5': 'GLM-5',
                'Gemini 3.1 Pro': '3.1 Pro', 'Qwen Coder 480B': 'Qwen Coder',
                'Qwen 80B Inst': 'Qwen 80B Inst',
            }
            short = short_map.get(m, m)
            # Custom offsets per model
            offsets = {
                'Claude Sonnet 4.6': (1.0, 0),
                'Claude Haiku 4.5': (1.0, -1),
                'GPT-5.4': (-7, 1.5),
                'GPT-5.4 mini': (-7.5, 2.0),
                'GPT-5.4 nano': (-1.0, -4),
                'Gemma 3 27B': (1.0, 0),
                'Gemma 3 1B': (1.0, -2),
                'Gemma 3 12B': (1.0, 0),
                'DeepSeek R1': (1.0, 1),
                'GLM-5': (-3.5, -3.5),
                'Gemini 3.1 Pro': (-3.5, 2.5),
                'Qwen Coder 480B': (1.0, -1),
                'Qwen 80B Inst': (-2, -4),
            }
            ox, oy = offsets.get(m, (1.0, 1.5))
            ax.annotate(short, (wd, direct), xytext=(wd + ox, direct + oy),
                        fontsize=8.5, ha='left', alpha=0.85)

    ax.axhline(50, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(15, color='gray', linestyle=':', alpha=0.3)
    ax.text(15.3, 5, '+15% threshold', fontsize=8, color='gray', alpha=0.7, rotation=90, va='bottom')

    ax.set_xlabel('Retrospective WΔ (%) — monitors after answering', fontsize=11)
    ax.set_ylabel('T6 ANSWER_DIRECTLY rate (%) — does not regulate before', fontsize=11)
    ax.set_title('Retrospective monitoring vs prospective regulation\n'
                 'Two dissociable capacities (r = .16, ρ = −.14, n.s.)', fontsize=12, pad=12)

    # Quadrant labels
    ax.text(38, 88, 'Monitors but does not\nregulate', ha='center', fontsize=9.5,
            style='italic', color='#444', alpha=0.85)
    ax.text(38, 25, 'Monitors AND\nregulates', ha='center', fontsize=9.5,
            style='italic', color='#444', alpha=0.85)
    ax.text(0, 25, 'Regulates without\nmonitoring well', ha='center', fontsize=9.5,
            style='italic', color='#444', alpha=0.85)
    ax.text(0, 65, 'Neither', ha='center', fontsize=9.5,
            style='italic', color='#444', alpha=0.85)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=COL_A, markersize=10, label='Profile A'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=COL_C, markersize=10, label='Profile C'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=COL_B, markersize=10, label='Profile B'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='gray', markeredgecolor=COL_GPT,
               markeredgewidth=2, markersize=10, label='GPT-5.4 family'),
    ]
    ax.legend(handles=legend_elems, loc='center right', fontsize=10, framealpha=0.95)

    ax.set_xlim(-5, 45)
    ax.set_ylim(-5, 110)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    plt.savefig(OUT / 'fig3_dissociation.png')
    plt.close()
    print('Fig 3 saved')


# ===================================================================
# FIGURE 4: Domain fragmentation — Sonnet, GLM-5, Haiku per-track
# ===================================================================
def fig4_fragmentation():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    tracks = ['T1','T2','T3','T4','T5','T6']

    examples = [
        ('Claude Sonnet 4.6', 'Sonnet: ranges from +8% (T1) to +93% (T5)'),
        ('GLM-5', 'GLM-5: +59% on T3, near zero elsewhere'),
        ('Claude Haiku 4.5', 'Haiku: selective on T1–T3 + T5, fails on T4 + T6'),
    ]

    for ax, (model, title) in zip(axes, examples):
        wds = []
        for t in tracks:
            if t in PER_TRACK.get(model, {}):
                wds.append(PER_TRACK[model][t]['wd'])
            else:
                wds.append(0)

        colors = [COL_A if w >= 15 else (COL_C if w < 0 else '#FFD27F') for w in wds]
        # Clearer: green if >=15, red if negative, amber for low positive
        colors = []
        for w in wds:
            if w >= 15: colors.append('#27AE60')
            elif w < 0: colors.append('#E74C3C')
            else: colors.append('#F39C12')

        bars = ax.bar(tracks, wds, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.axhline(15, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)

        for bar, w in zip(bars, wds):
            y = bar.get_height()
            offset = 2 if y >= 0 else -4
            ax.text(bar.get_x() + bar.get_width()/2, y + offset, f'{w:+.0f}%',
                    ha='center', va='bottom' if y >= 0 else 'top', fontsize=9)

        ax.set_ylabel('Withdraw delta (%)')
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-15, 105)
        ax.grid(axis='y', alpha=0.2)

    fig.suptitle('Per-track withdraw delta for three models — domain-dependent metacognitive profiles',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / 'fig4_fragmentation.png')
    plt.close()
    print('Fig 4 saved')


# ===================================================================
# FIGURE 5: Architecture-dependent scaling (THREE families, T2 WD)
# ===================================================================
def fig5_scaling():
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # Use T2 WD for all families — principled single-metric comparison
    def family_t2_data(models):
        accs, wds = [], []
        for m in models:
            if 'T2' in PER_TRACK.get(m, {}):
                accs.append(PER_TRACK[m]['T2']['acc'] * 100)
                wds.append(PER_TRACK[m]['T2']['wd'])
            else:
                accs.append(0)
                wds.append(0)
        return accs, wds

    w = 0.35

    # Panel A: Qwen
    ax = axes[0]
    accs, wds = family_t2_data(QWEN_FAMILY)
    x = np.arange(len(QWEN_FAMILY))
    ax.bar(x - w/2, accs, w, label='T2 Accuracy (%)', color='#5DADE2', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, wds, w, label='T2 WΔ (%)', color='#E74C3C', edgecolor='black', linewidth=0.5)
    for i, a in enumerate(accs):
        ax.text(i - w/2, a + 1, f'{a:.1f}', ha='center', fontsize=8)
    for i, wd_val in enumerate(wds):
        ax.text(i + w/2, wd_val + 1, f'+{wd_val:.1f}', ha='center', fontsize=8, color='#A52A2A')
    ax.set_xticks(x)
    ax.set_xticklabels(['80B\nInst', '235B\nInst', 'Coder\n480B'], fontsize=9)
    ax.set_ylabel('Percentage')
    ax.set_ylim(0, 110)
    ax.set_title('(a) Qwen: T2 WΔ declines\nmonotonically', fontsize=11)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(axis='y', alpha=0.2)

    # Panel B: GPT
    ax = axes[1]
    accs, wds = family_t2_data(GPT_FAMILY)
    x = np.arange(len(GPT_FAMILY))
    ax.bar(x - w/2, accs, w, label='T2 Accuracy (%)', color='#5DADE2', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, wds, w, label='T2 WΔ (%)', color='#27AE60', edgecolor='black', linewidth=0.5)
    for i, a in enumerate(accs):
        ax.text(i - w/2, a + 1, f'{a:.1f}', ha='center', fontsize=8)
    for i, wd_val in enumerate(wds):
        ax.text(i + w/2, wd_val + 1, f'+{wd_val:.1f}', ha='center', fontsize=8, color='#1E7A4D')
    ax.set_xticks(x)
    ax.set_xticklabels(['nano', 'mini', '5.4'], fontsize=9)
    ax.set_ylabel('Percentage')
    ax.set_ylim(0, 110)
    ax.set_title('(b) GPT-5.4: T2 WΔ rises\nmonotonically', fontsize=11)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(axis='y', alpha=0.2)

    # Panel C: Gemma
    ax = axes[2]
    accs, wds = family_t2_data(GEMMA_FAMILY)
    x = np.arange(len(GEMMA_FAMILY))
    ax.bar(x - w/2, accs, w, label='T2 Accuracy (%)', color='#5DADE2', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, wds, w, label='T2 WΔ (%)', color=COL_GEMMA, edgecolor='black', linewidth=0.5)
    for i, a in enumerate(accs):
        ax.text(i - w/2, a + 1, f'{a:.1f}', ha='center', fontsize=8)
    for i, wd_val in enumerate(wds):
        ax.text(i + w/2, wd_val + 1, f'+{wd_val:.1f}', ha='center', fontsize=8, color='#0E6655')
    ax.set_xticks(x)
    ax.set_xticklabels(['1B', '12B', '27B'], fontsize=9)
    ax.set_ylabel('Percentage')
    ax.set_ylim(0, 110)
    ax.axhline(15, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    ax.set_title('(c) Gemma: T2 WΔ flat\nacross scale', fontsize=11)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(axis='y', alpha=0.2)

    # Panel D: All three trajectories on one plot (T2 WD)
    ax = axes[3]
    _, qwen_wd = family_t2_data(QWEN_FAMILY)
    _, gpt_wd = family_t2_data(GPT_FAMILY)
    _, gemma_wd = family_t2_data(GEMMA_FAMILY)
    pos = ['Small', 'Medium', 'Large']
    ax.plot(pos, qwen_wd, marker='o', color=COL_QWEN, linewidth=2.5, markersize=12, label='Qwen (↓)')
    ax.plot(pos, gpt_wd, marker='D', color=COL_GPT, linewidth=2.5, markersize=12, label='GPT-5.4 (↑)')
    ax.plot(pos, gemma_wd, marker='s', color=COL_GEMMA, linewidth=2.5, markersize=12, label='Gemma (—)')
    for i, w_val in enumerate(qwen_wd):
        ax.text(i, w_val + 1.2, f'+{w_val:.1f}', color=COL_QWEN, ha='center', fontsize=9)
    for i, w_val in enumerate(gpt_wd):
        ax.text(i, w_val - 2.5, f'+{w_val:.1f}', color=COL_GPT, ha='center', fontsize=9)
    for i, w_val in enumerate(gemma_wd):
        offset = 1.2 if i != 1 else -2.5  # avoid Gemma 12B overlapping with GPT labels
        ax.text(i, w_val + offset, f'+{w_val:.1f}', color=COL_GEMMA, ha='center', fontsize=9)
    ax.axhline(15, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    ax.text(2, 14, '+15% threshold', fontsize=8, color='gray', ha='right')
    ax.set_xlabel('Model scale (within-family)', fontsize=10)
    ax.set_ylabel('T2 WΔ (%)')
    ax.set_ylim(0, 40)
    ax.set_title('(d) Three architectures,\nthree trajectories (T2)', fontsize=11)
    ax.legend(fontsize=10, framealpha=0.95, loc='upper left')
    ax.grid(alpha=0.2)

    fig.suptitle('Architecture-dependent scaling on T2 (metacognitive calibration): no universal scaling law',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / 'fig5_scaling.png')
    plt.close()
    print('Fig 5 saved')


# ===================================================================
# FIGURE 6: Psychometric robustness (corrected Cohen's d = 4.57)
# ===================================================================
def fig6_robustness():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Inter-track correlation matrix
    ax = axes[0]
    tracks = ['T1','T2','T3','T4','T5','T6']
    n_tracks = len(tracks)
    # Build per-track WD matrix
    models = list(LEADERBOARD.keys())
    wd_matrix = []
    for m in models:
        row = []
        for t in tracks:
            if t in PER_TRACK.get(m, {}):
                w = PER_TRACK[m][t]['wd']
                row.append(float(w) if w is not None else np.nan)
            else:
                row.append(np.nan)
        wd_matrix.append(row)
    wd_matrix = np.array(wd_matrix, dtype=float)

    # Correlation matrix
    corr = np.zeros((n_tracks, n_tracks))
    for i in range(n_tracks):
        for j in range(n_tracks):
            mask = ~(np.isnan(wd_matrix[:,i]) | np.isnan(wd_matrix[:,j]))
            if mask.sum() >= 3:
                corr[i,j] = np.corrcoef(wd_matrix[mask,i], wd_matrix[mask,j])[0,1]
            else:
                corr[i,j] = np.nan

    im = ax.imshow(corr, cmap='RdYlGn', vmin=-0.5, vmax=1, aspect='auto')
    ax.set_xticks(range(n_tracks))
    ax.set_yticks(range(n_tracks))
    ax.set_xticklabels(tracks)
    ax.set_yticklabels(tracks)
    for i in range(n_tracks):
        for j in range(n_tracks):
            val = corr[i,j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='r')
    ax.set_title('(a) Inter-track WΔ correlations', fontsize=11)

    # Panel B: Accuracy vs WD scatter
    ax = axes[1]
    accs_lookup = {}
    with open('outputs/all_tracks_probe_results.csv') as f:
        for r in csv.DictReader(f):
            accs_lookup.setdefault(r['model'], []).append(float(r['accuracy']))
    accs_mean = {m: np.mean(v)*100 for m, v in accs_lookup.items()}

    for m in LEADERBOARD:
        wd = LEADERBOARD[m]['mean_wd']
        acc = accs_mean.get(m, 0)
        if m in PROFILE_A: c, marker = COL_A, 'o'
        elif m in PROFILE_B: c, marker = COL_B, 's'
        elif m in PROFILE_C: c, marker = COL_C, 'o'
        else: c, marker = '#999', 'o'
        size = 150 if m in GPT_FAMILY else 100
        edge = COL_GPT if m in GPT_FAMILY else 'black'
        elw = 2 if m in GPT_FAMILY else 0.5
        ax.scatter(acc, wd, s=size, c=c, marker=marker, edgecolors=edge, linewidths=elw, alpha=0.8)

    # Compute r
    accs_list = [accs_mean.get(m, 0) for m in LEADERBOARD]
    wds_list = [LEADERBOARD[m]['mean_wd'] for m in LEADERBOARD]
    r = np.corrcoef(accs_list, wds_list)[0,1]
    ax.text(0.97, 0.97, f'r = {r:.2f}', transform=ax.transAxes, ha='right', va='top',
            fontsize=12, color='gray', fontweight='bold')

    ax.set_xlabel('Overall accuracy (%)', fontsize=10)
    ax.set_ylabel('Mean WΔ (%)', fontsize=10)
    ax.set_title('(b) Accuracy and WΔ are independent', fontsize=11)
    ax.grid(alpha=0.2)

    # Panel C: Threshold stability
    ax = axes[2]
    thresholds = [10, 12, 14, 15, 17, 20, 25]
    pheno_a_counts, pheno_b_counts, pheno_c_counts = [], [], []
    for thr in thresholds:
        n_a = sum(1 for m in LEADERBOARD if LEADERBOARD[m]['mean_wd'] < thr and m != 'DeepSeek R1' and accs_mean.get(m,0) > 0)
        # Profile A: WD low; Profile C: WD high
        pheno_c_counts.append(sum(1 for m in LEADERBOARD if LEADERBOARD[m]['mean_wd'] >= thr))
        pheno_a_counts.append(sum(1 for m in LEADERBOARD if LEADERBOARD[m]['mean_wd'] < thr and m != 'DeepSeek R1'))
        pheno_b_counts.append(1)  # always R1

    ax.plot(thresholds, pheno_a_counts, marker='o', color=COL_A, linewidth=2, markersize=8, label='Profile A')
    ax.plot(thresholds, pheno_c_counts, marker='o', color=COL_C, linewidth=2, markersize=8, label='Profile C')
    ax.plot(thresholds, pheno_b_counts, marker='s', color=COL_B, linewidth=2, markersize=8, label='Profile B')
    ax.axvline(15, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='Used threshold')
    ax.set_xlabel('WΔ threshold (%)', fontsize=10)
    ax.set_ylabel('Model count', fontsize=10)
    ax.set_ylim(0, 16)
    ax.set_title('(c) Profile counts stable across thresholds', fontsize=11)
    ax.legend(fontsize=9, loc='center right')
    ax.grid(alpha=0.2)

    fig.suptitle("Psychometric robustness — α = .54, split-half r = .51 (SB = .68), Cohen's d (A vs C) = 4.57, 95% CI [3.65, 7.95]",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / 'fig6_robustness.png')
    plt.close()
    print('Fig 6 saved')


def run_figures():
    """Load data and generate all six figures."""
    _load_data()
    fig1_phenotypes()
    fig2_slope()
    fig3_dissociation()
    fig4_fragmentation()
    fig5_scaling()
    fig6_robustness()
    print('All figures generated in', OUT)


if __name__ == '__main__':
    run_figures()
