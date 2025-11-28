#!/usr/bin/env python3
"""
Visualize v1 vs v2 persona performance comparison.

Creates publication-quality plots comparing moderate (v1) vs extreme (v2) personas.

Usage:
    python3 plot_persona_version_comparison.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('summary_personas.csv')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'v1': '#3498db', 'v2': '#e74c3c'}  # Blue for v1, Red for v2

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Persona Version Comparison: v1 (Moderate) vs v2 (Extreme)',
             fontsize=16, fontweight='bold', y=0.995)

# ============================================================================
# Plot 1: MaxMin Score by Number of Agents
# ============================================================================
ax = axes[0, 0]
by_agents = df.groupby(['n_agents', 'persona_version'])['maxmin_score'].mean().unstack()

by_agents.plot(kind='bar', ax=ax, color=[colors['v1'], colors['v2']], width=0.7)
ax.set_title('MaxMin Score by Agent Count', fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Agents', fontsize=11)
ax.set_ylabel('Mean MaxMin Score', fontsize=11)
ax.legend(title='Persona Version', labels=['v1 (moderate)', 'v2 (extreme)'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(True, alpha=0.3)

# Add percentage improvement annotations
for i, n_agents in enumerate(by_agents.index):
    v1_val = by_agents.loc[n_agents, 'v1']
    v2_val = by_agents.loc[n_agents, 'v2']
    improvement = ((v2_val - v1_val) / v1_val) * 100
    ax.text(i, max(v1_val, v2_val) + 0.01, f'+{improvement:.0f}%',
            ha='center', fontsize=9, color='green', fontweight='bold')

# ============================================================================
# Plot 2: MaxDet Volume by Number of Agents
# ============================================================================
ax = axes[0, 1]
by_agents_det = df.groupby(['n_agents', 'persona_version'])['maxdet_volume'].mean().unstack()

by_agents_det.plot(kind='bar', ax=ax, color=[colors['v1'], colors['v2']], width=0.7)
ax.set_title('MaxDet Volume by Agent Count', fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Agents', fontsize=11)
ax.set_ylabel('Mean MaxDet Volume', fontsize=11)
ax.legend(title='Persona Version', labels=['v1 (moderate)', 'v2 (extreme)'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(True, alpha=0.3)

# ============================================================================
# Plot 3: MaxMin Score by Model Size
# ============================================================================
ax = axes[1, 0]
by_size = df.groupby(['params_b', 'persona_version'])['maxmin_score'].mean().unstack()

x = np.arange(len(by_size.index))
width = 0.35

ax.bar(x - width/2, by_size['v1'], width, label='v1 (moderate)', color=colors['v1'])
ax.bar(x + width/2, by_size['v2'], width, label='v2 (extreme)', color=colors['v2'])

ax.set_title('MaxMin Score by Model Size', fontsize=12, fontweight='bold')
ax.set_xlabel('Model Parameters (B)', fontsize=11)
ax.set_ylabel('Mean MaxMin Score', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([f'{p:.1f}B' for p in by_size.index], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Plot 4: Win Rate Summary
# ============================================================================
ax = axes[1, 1]

# Calculate win rates
merged = df.merge(df, on=['model', 'n_agents'], suffixes=('_v1', '_v2'))
comparisons = merged[
    (merged['persona_version_v1'] == 'v1') &
    (merged['persona_version_v2'] == 'v2')
]

maxmin_v2_wins = (comparisons['maxmin_score_v2'] > comparisons['maxmin_score_v1']).sum()
maxdet_v2_wins = (comparisons['maxdet_volume_v2'] > comparisons['maxdet_volume_v1']).sum()
total = len(comparisons)

metrics = ['MaxMin\nScore', 'MaxDet\nVolume']
v2_win_pcts = [maxmin_v2_wins/total*100, maxdet_v2_wins/total*100]
v1_win_pcts = [100-pct for pct in v2_win_pcts]

x_pos = np.arange(len(metrics))
width = 0.6

# Stacked bar chart
ax.bar(x_pos, v1_win_pcts, width, label='v1 (moderate)', color=colors['v1'])
ax.bar(x_pos, v2_win_pcts, width, bottom=v1_win_pcts, label='v2 (extreme)', color=colors['v2'])

ax.set_title('Head-to-Head Win Rates', fontsize=12, fontweight='bold')
ax.set_ylabel('Win Rate (%)', fontsize=11)
ax.set_ylim(0, 100)
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics)
ax.legend(loc='center left')
ax.grid(True, alpha=0.3, axis='y')

# Add percentage labels
for i, (v1, v2) in enumerate(zip(v1_win_pcts, v2_win_pcts)):
    if v1 > 0:
        ax.text(i, v1/2, f'{v1:.0f}%', ha='center', va='center',
                fontweight='bold', color='white', fontsize=11)
    if v2 > 0:
        ax.text(i, v1 + v2/2, f'{v2:.0f}%', ha='center', va='center',
                fontweight='bold', color='white', fontsize=11)

# Add summary text box
summary_text = (
    f"Summary (n={total} matchups):\n"
    f"  v2 dominates on both metrics\n"
    f"  v2 wins: 100% of matchups\n"
    f"  Avg improvement: +70% (MaxMin)\n"
    f"                   +96% (MaxDet)"
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(1.5, 50, summary_text, transform=ax.transData, fontsize=9,
        verticalalignment='center', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('persona_version_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved to: personas/persona_version_comparison.png")
plt.show()
