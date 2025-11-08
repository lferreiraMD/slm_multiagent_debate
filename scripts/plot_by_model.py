#!/usr/bin/env python3
"""
Generate plots per (model, task) combination.

For each unique (model, task) pair, creates a line plot with:
- X-axis: number of agents
- Y-axis: average accuracy
- One line per number of rounds

Output: plots/{model}_{task}.png

Usage:
    python scripts/plot_by_model.py
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_model_task(df, model_name, task, output_dir):
    """
    Create a plot for a specific (model, task) combination.

    Args:
        df: DataFrame with results for this model/task
        model_name: Model name
        task: Task name
        output_dir: Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique rounds, sorted
    rounds_list = sorted(df['num_rounds'].unique())

    # Plot one line per round
    for rounds in rounds_list:
        round_data = df[df['num_rounds'] == rounds].sort_values('num_agents')

        if len(round_data) > 0:
            ax.plot(round_data['num_agents'],
                   round_data['avg_accuracy'],
                   marker='o',
                   label=f'{rounds} rounds',
                   linewidth=2,
                   markersize=8)

    ax.set_xlabel('Number of Agents', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title(f'{model_name} - {task.upper()}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set integer ticks for x-axis
    ax.set_xticks(sorted(df['num_agents'].unique()))

    plt.tight_layout()

    # Save plot
    filename = f"{model_name}_{task}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Generate Plots by Model and Task")
    print("=" * 60)

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Load summary data
    summary_path = project_root / 'results' / 'summary.p'

    if not summary_path.exists():
        print(f"Error: Summary file not found at {summary_path}")
        print("Run 'python scripts/aggregate_results.py' first")
        return

    df = pd.read_pickle(summary_path)
    print(f"Loaded {len(df)} results from {summary_path}\n")

    # Create output directory
    output_dir = project_root / 'plots'
    output_dir.mkdir(exist_ok=True)

    # Group by (model, task)
    grouped = df.groupby(['model_name', 'task'])

    print(f"Generating {len(grouped)} plots...\n")

    for (model_name, task), group_df in grouped:
        print(f"Processing: {model_name} - {task}")
        plot_model_task(group_df, model_name, task, output_dir)

    print("\n" + "=" * 60)
    print(f"All plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
