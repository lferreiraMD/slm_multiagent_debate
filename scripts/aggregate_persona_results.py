#!/usr/bin/env python3
"""
Aggregate persona diversity experiment results and compare with baselines.

Scans task directories for persona diversity results, computes accuracy
deltas vs baseline, and generates analysis DataFrames.

Usage:
    python scripts/aggregate_persona_results.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_summary():
    """Load existing summary.p file."""
    project_root = Path(__file__).parent.parent
    summary_path = project_root / 'results' / 'summary.p'

    if not summary_path.exists():
        print(f"Error: Summary file not found: {summary_path}")
        print("Run 'python scripts/aggregate_results.py' first to generate summary.p")
        sys.exit(1)

    df = pd.read_pickle(summary_path)
    print(f"Loaded {len(df)} results from {summary_path}")

    return df


def analyze_persona_diversity(df):
    """
    Analyze persona diversity experiments vs baselines.

    For each model/task/agent count combination:
    - Find baseline (diversity_type='baseline')
    - Find persona result (diversity_type='persona')
    - Compute delta and percent improvement

    Returns:
        DataFrame with columns: model, task, agents, baseline_acc, persona_acc,
        delta_acc, pct_improvement
    """
    # Ensure diversity_type column exists
    if 'diversity_type' not in df.columns:
        print("Warning: No diversity_type column found in summary")
        print("Results may have been generated before diversity tracking was added")
        df['diversity_type'] = 'baseline'  # Assume all are baseline

    # Filter for baselines and persona experiments
    baselines = df[df['diversity_type'] == 'baseline'].copy()
    personas = df[df['diversity_type'] == 'persona'].copy()

    print(f"\nFound {len(baselines)} baseline results")
    print(f"Found {len(personas)} persona diversity results")

    if len(personas) == 0:
        print("\nNo persona diversity results found yet.")
        print("Run persona experiments first:")
        print("  cd experiments/slurm")
        print("  bash submit_all_persona_experiments.sh")
        return pd.DataFrame()

    # Create comparison rows
    comparison_rows = []

    for _, persona_row in personas.iterrows():
        # Find matching baseline
        baseline_row = baselines[
            (baselines['task'] == persona_row['task']) &
            (baselines['model_name'] == persona_row['model_name']) &
            (baselines['num_agents'] == persona_row['num_agents']) &
            (baselines['num_rounds'] == persona_row['num_rounds'])
        ]

        if len(baseline_row) == 0:
            print(f"Warning: No baseline found for {persona_row['task']}/{persona_row['model_name']}/agents{persona_row['num_agents']}")
            continue

        baseline_row = baseline_row.iloc[0]

        baseline_acc = baseline_row['avg_accuracy']
        persona_acc = persona_row['avg_accuracy']
        delta = persona_acc - baseline_acc
        pct_improvement = (delta / baseline_acc * 100) if baseline_acc > 0 else 0

        comparison_rows.append({
            'task': persona_row['task'],
            'model_name': persona_row['model_name'],
            'num_agents': persona_row['num_agents'],
            'num_rounds': persona_row['num_rounds'],
            'baseline_acc': baseline_acc,
            'baseline_std': baseline_row['std_accuracy'],
            'persona_acc': persona_acc,
            'persona_std': persona_row['std_accuracy'],
            'delta_acc': delta,
            'pct_improvement': pct_improvement
        })

    comparison_df = pd.DataFrame(comparison_rows)

    if len(comparison_df) == 0:
        print("\nNo matching baseline-persona pairs found")
        return comparison_df

    # Sort by improvement
    comparison_df = comparison_df.sort_values('pct_improvement', ascending=False)

    return comparison_df


def print_summary_statistics(comparison_df):
    """Print summary statistics about persona diversity benefits."""
    if len(comparison_df) == 0:
        return

    print("\n" + "=" * 80)
    print("PERSONA DIVERSITY IMPACT SUMMARY")
    print("=" * 80)

    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Total comparisons: {len(comparison_df)}")
    print(f"  Improvements: {len(comparison_df[comparison_df['delta_acc'] > 0])}")
    print(f"  Regressions: {len(comparison_df[comparison_df['delta_acc'] < 0])}")
    print(f"  No change: {len(comparison_df[comparison_df['delta_acc'] == 0])}")

    print(f"\n  Average accuracy delta: {comparison_df['delta_acc'].mean():.4f}")
    print(f"  Average percent improvement: {comparison_df['pct_improvement'].mean():.2f}%")
    print(f"  Median percent improvement: {comparison_df['pct_improvement'].median():.2f}%")

    # Best improvements
    print("\nTop 10 Improvements:")
    top_improvements = comparison_df.nlargest(10, 'pct_improvement')
    for _, row in top_improvements.iterrows():
        print(f"  {row['task']:10s} {row['model_name']:25s} agents={row['num_agents']} "
              f"baseline={row['baseline_acc']:.3f} → persona={row['persona_acc']:.3f} "
              f"(+{row['pct_improvement']:6.2f}%)")

    # Worst regressions
    if len(comparison_df[comparison_df['delta_acc'] < 0]) > 0:
        print("\nTop 10 Regressions:")
        worst_regressions = comparison_df.nsmallest(10, 'pct_improvement')
        for _, row in worst_regressions.iterrows():
            print(f"  {row['task']:10s} {row['model_name']:25s} agents={row['num_agents']} "
                  f"baseline={row['baseline_acc']:.3f} → persona={row['persona_acc']:.3f} "
                  f"({row['pct_improvement']:6.2f}%)")

    # Per-task analysis
    print("\nBy Task:")
    for task in sorted(comparison_df['task'].unique()):
        task_df = comparison_df[comparison_df['task'] == task]
        print(f"  {task:10s}: {len(task_df)} experiments, "
              f"avg improvement: {task_df['pct_improvement'].mean():6.2f}%, "
              f"median: {task_df['pct_improvement'].median():6.2f}%")

    # Per-agent count analysis
    print("\nBy Agent Count:")
    for agents in sorted(comparison_df['num_agents'].unique()):
        agents_df = comparison_df[comparison_df['num_agents'] == agents]
        print(f"  {agents} agents: {len(agents_df)} experiments, "
              f"avg improvement: {agents_df['pct_improvement'].mean():6.2f}%, "
              f"median: {agents_df['pct_improvement'].median():6.2f}%")


def main():
    """Main entry point."""
    print("=" * 80)
    print("Persona Diversity Results Aggregation & Analysis")
    print("=" * 80)

    # Load summary
    df = load_summary()

    # Analyze persona diversity
    comparison_df = analyze_persona_diversity(df)

    if len(comparison_df) == 0:
        print("\nNo persona diversity comparisons available yet.")
        return

    # Print statistics
    print_summary_statistics(comparison_df)

    # Save results
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)

    # Save comparison table
    comparison_path = results_dir / 'persona_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n\nComparison table saved to: {comparison_path}")

    # Also save as pickle for easy loading
    comparison_df.to_pickle(results_dir / 'persona_comparison.p')
    print(f"Pickle saved to: {results_dir / 'persona_comparison.p'}")

    # Full results table
    print("\n" + "=" * 80)
    print("FULL COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total experiments analyzed: {len(comparison_df)}")
    print(f"Results saved to: {results_dir}")
    print("\nNext steps:")
    print("  - Review persona_comparison.csv for detailed results")
    print("  - Generate plots with plot_persona_diversity.py")
    print("  - Compare with model/temperature diversity results")
    print("=" * 80)


if __name__ == "__main__":
    main()
