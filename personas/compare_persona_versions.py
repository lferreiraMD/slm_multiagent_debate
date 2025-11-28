#!/usr/bin/env python3
"""
Compare v1 (moderate) vs v2 (extreme) persona performance.

Analyzes which persona version achieves better MaxMin and MaxDet scores
across all models and agent counts.

Usage:
    python3 compare_persona_versions.py
"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('summary_personas.csv')

print("=" * 80)
print("PERSONA VERSION COMPARISON: v1 (moderate) vs v2 (extreme)")
print("=" * 80)
print()

# ============================================================================
# 1. OVERALL COMPARISON (across all models and agent counts)
# ============================================================================
print("1. OVERALL PERFORMANCE")
print("-" * 80)

overall_stats = df.groupby('persona_version').agg({
    'maxmin_score': ['mean', 'std', 'median', 'min', 'max'],
    'maxdet_volume': ['mean', 'std', 'median', 'min', 'max'],
    'intersection_count': ['mean', 'std', 'median']
}).round(6)

print("\nMaxMin Score (higher is better - maximum minimum distance):")
print(overall_stats['maxmin_score'])
print()

print("MaxDet Volume (higher is better - determinant of similarity matrix):")
print(overall_stats['maxdet_volume'])
print()

print("Intersection Count (agreement between MaxMin and MaxDet):")
print(overall_stats['intersection_count'])
print()

# Statistical comparison
v1_maxmin = df[df['persona_version'] == 'v1']['maxmin_score']
v2_maxmin = df[df['persona_version'] == 'v2']['maxmin_score']
v1_maxdet = df[df['persona_version'] == 'v1']['maxdet_volume']
v2_maxdet = df[df['persona_version'] == 'v2']['maxdet_volume']

print("WINNER SUMMARY:")
print(f"  MaxMin Score:  {'v2' if v2_maxmin.mean() > v1_maxmin.mean() else 'v1'} "
      f"(v1: {v1_maxmin.mean():.6f}, v2: {v2_maxmin.mean():.6f}, "
      f"Δ: {abs(v2_maxmin.mean() - v1_maxmin.mean()):.6f})")
print(f"  MaxDet Volume: {'v2' if v2_maxdet.mean() > v1_maxdet.mean() else 'v1'} "
      f"(v1: {v1_maxdet.mean():.6f}, v2: {v2_maxdet.mean():.6f}, "
      f"Δ: {abs(v2_maxdet.mean() - v1_maxdet.mean()):.6f})")
print()

# ============================================================================
# 2. WIN RATE (how often v1 beats v2, and vice versa)
# ============================================================================
print("2. HEAD-TO-HEAD WIN RATES")
print("-" * 80)

# Compare matched pairs (same model, same n_agents)
merged = df.merge(
    df,
    on=['model', 'n_agents', 'params_b', 'framework'],
    suffixes=('_v1', '_v2')
)
# Filter to only v1 vs v2 comparisons
comparisons = merged[
    (merged['persona_version_v1'] == 'v1') &
    (merged['persona_version_v2'] == 'v2')
]

maxmin_v1_wins = (comparisons['maxmin_score_v1'] > comparisons['maxmin_score_v2']).sum()
maxmin_v2_wins = (comparisons['maxmin_score_v2'] > comparisons['maxmin_score_v1']).sum()
maxmin_ties = (comparisons['maxmin_score_v1'] == comparisons['maxmin_score_v2']).sum()

maxdet_v1_wins = (comparisons['maxdet_volume_v1'] > comparisons['maxdet_volume_v2']).sum()
maxdet_v2_wins = (comparisons['maxdet_volume_v2'] > comparisons['maxdet_volume_v1']).sum()
maxdet_ties = (comparisons['maxdet_volume_v1'] == comparisons['maxdet_volume_v2']).sum()

total_matchups = len(comparisons)

print(f"Total head-to-head matchups: {total_matchups}")
print()

print("MaxMin Score Wins:")
print(f"  v1 wins: {maxmin_v1_wins}/{total_matchups} ({maxmin_v1_wins/total_matchups*100:.1f}%)")
print(f"  v2 wins: {maxmin_v2_wins}/{total_matchups} ({maxmin_v2_wins/total_matchups*100:.1f}%)")
print(f"  Ties:    {maxmin_ties}/{total_matchups} ({maxmin_ties/total_matchups*100:.1f}%)")
print()

print("MaxDet Volume Wins:")
print(f"  v1 wins: {maxdet_v1_wins}/{total_matchups} ({maxdet_v1_wins/total_matchups*100:.1f}%)")
print(f"  v2 wins: {maxdet_v2_wins}/{total_matchups} ({maxdet_v2_wins/total_matchups*100:.1f}%)")
print(f"  Ties:    {maxdet_ties}/{total_matchups} ({maxdet_ties/total_matchups*100:.1f}%)")
print()

# ============================================================================
# 3. BREAKDOWN BY NUMBER OF AGENTS
# ============================================================================
print("3. PERFORMANCE BY NUMBER OF AGENTS")
print("-" * 80)

by_agents = df.groupby(['n_agents', 'persona_version']).agg({
    'maxmin_score': 'mean',
    'maxdet_volume': 'mean',
    'intersection_count': 'mean'
}).round(6)

print("\nMaxMin Score (mean):")
print(by_agents['maxmin_score'].unstack())
print()

print("MaxDet Volume (mean):")
print(by_agents['maxdet_volume'].unstack())
print()

print("Which version wins for each agent count?")
maxmin_by_agents = by_agents['maxmin_score'].unstack()
maxdet_by_agents = by_agents['maxdet_volume'].unstack()

print("\nMaxMin Winners:")
for n in maxmin_by_agents.index:
    winner = 'v1' if maxmin_by_agents.loc[n, 'v1'] > maxmin_by_agents.loc[n, 'v2'] else 'v2'
    print(f"  {n} agents: {winner} (v1: {maxmin_by_agents.loc[n, 'v1']:.6f}, v2: {maxmin_by_agents.loc[n, 'v2']:.6f})")

print("\nMaxDet Winners:")
for n in maxdet_by_agents.index:
    winner = 'v1' if maxdet_by_agents.loc[n, 'v1'] > maxdet_by_agents.loc[n, 'v2'] else 'v2'
    print(f"  {n} agents: {winner} (v1: {maxdet_by_agents.loc[n, 'v1']:.6f}, v2: {maxdet_by_agents.loc[n, 'v2']:.6f})")
print()

# ============================================================================
# 4. BREAKDOWN BY MODEL SIZE
# ============================================================================
print("4. PERFORMANCE BY MODEL SIZE")
print("-" * 80)

by_size = df.groupby(['params_b', 'persona_version']).agg({
    'maxmin_score': 'mean',
    'maxdet_volume': 'mean',
    'intersection_count': 'mean'
}).round(6)

print("\nMaxMin Score (mean):")
print(by_size['maxmin_score'].unstack())
print()

print("MaxDet Volume (mean):")
print(by_size['maxdet_volume'].unstack())
print()

print("Which version wins for each model size?")
maxmin_by_size = by_size['maxmin_score'].unstack()
maxdet_by_size = by_size['maxdet_volume'].unstack()

print("\nMaxMin Winners:")
for params in sorted(maxmin_by_size.index):
    winner = 'v1' if maxmin_by_size.loc[params, 'v1'] > maxmin_by_size.loc[params, 'v2'] else 'v2'
    print(f"  {params}B: {winner} (v1: {maxmin_by_size.loc[params, 'v1']:.6f}, v2: {maxmin_by_size.loc[params, 'v2']:.6f})")

print("\nMaxDet Winners:")
for params in sorted(maxdet_by_size.index):
    winner = 'v1' if maxdet_by_size.loc[params, 'v1'] > maxdet_by_size.loc[params, 'v2'] else 'v2'
    print(f"  {params}B: {winner} (v1: {maxdet_by_size.loc[params, 'v1']:.6f}, v2: {maxdet_by_size.loc[params, 'v2']:.6f})")
print()

# ============================================================================
# 5. PER-MODEL BREAKDOWN
# ============================================================================
print("5. PERFORMANCE BY INDIVIDUAL MODEL")
print("-" * 80)

by_model = df.groupby(['model', 'persona_version']).agg({
    'maxmin_score': 'mean',
    'maxdet_volume': 'mean'
}).round(6)

print("\nMaxMin Score (mean):")
maxmin_by_model = by_model['maxmin_score'].unstack()
maxmin_by_model['winner'] = maxmin_by_model.apply(
    lambda row: 'v1' if row['v1'] > row['v2'] else 'v2', axis=1
)
print(maxmin_by_model)
print()

print("MaxDet Volume (mean):")
maxdet_by_model = by_model['maxdet_volume'].unstack()
maxdet_by_model['winner'] = maxdet_by_model.apply(
    lambda row: 'v1' if row['v1'] > row['v2'] else 'v2', axis=1
)
print(maxdet_by_model)
print()

# Count wins per model
maxmin_model_v1_wins = (maxmin_by_model['winner'] == 'v1').sum()
maxmin_model_v2_wins = (maxmin_by_model['winner'] == 'v2').sum()
maxdet_model_v1_wins = (maxdet_by_model['winner'] == 'v1').sum()
maxdet_model_v2_wins = (maxdet_by_model['winner'] == 'v2').sum()

print(f"MaxMin: v1 wins {maxmin_model_v1_wins}/10 models, v2 wins {maxmin_model_v2_wins}/10 models")
print(f"MaxDet: v1 wins {maxdet_model_v1_wins}/10 models, v2 wins {maxdet_model_v2_wins}/10 models")
print()

# ============================================================================
# 6. FINAL VERDICT
# ============================================================================
print("=" * 80)
print("FINAL VERDICT")
print("=" * 80)

maxmin_verdict = "v2 (extreme)" if v2_maxmin.mean() > v1_maxmin.mean() else "v1 (moderate)"
maxdet_verdict = "v2 (extreme)" if v2_maxdet.mean() > v1_maxdet.mean() else "v1 (moderate)"

print(f"\nMaxMin Score:  {maxmin_verdict} performs better overall")
print(f"  - Mean score: {max(v1_maxmin.mean(), v2_maxmin.mean()):.6f}")
print(f"  - Win rate: {max(maxmin_v1_wins, maxmin_v2_wins)}/{total_matchups} head-to-head matchups")
print(f"  - Consistent across {max(maxmin_model_v1_wins, maxmin_model_v2_wins)}/10 models")

print(f"\nMaxDet Volume: {maxdet_verdict} performs better overall")
print(f"  - Mean volume: {max(v1_maxdet.mean(), v2_maxdet.mean()):.6f}")
print(f"  - Win rate: {max(maxdet_v1_wins, maxdet_v2_wins)}/{total_matchups} head-to-head matchups")
print(f"  - Consistent across {max(maxdet_model_v1_wins, maxdet_model_v2_wins)}/10 models")

print()
print("RECOMMENDATION:")
if maxmin_verdict == maxdet_verdict:
    print(f"  Use {maxmin_verdict} personas - they consistently outperform on both metrics")
else:
    print(f"  Mixed results: {maxmin_verdict} for MaxMin, {maxdet_verdict} for MaxDet")
    print(f"  Consider your priority metric when choosing persona version")
print()
print("=" * 80)
