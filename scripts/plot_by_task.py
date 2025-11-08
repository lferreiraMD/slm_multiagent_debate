#!/usr/bin/env python3
"""
Generate comparison plots per task showing all models.

For each task, creates a plot with:
- X-axis: number of agents
- Y-axis: average accuracy
- One subplot per number of rounds
- Different line for each model

Output: plots/{task}_comparison.png

Usage:
    python scripts/plot_by_task.py
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_task_comparison(df, task, output_dir):
    """
    Create a comparison plot for a specific task showing all models.

    Args:
        df: DataFrame with results for this task
        task: Task name
        output_dir: Directory to save plot
    """
    # Get unique rounds, sorted
    rounds_list = sorted(df['num_rounds'].unique())
    num_subplots = len(rounds_list)

    # Create subplots (one per round)
    if num_subplots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]  # Make it iterable
    else:
        # Arrange subplots in a grid
        ncols = min(2, num_subplots)
        nrows = (num_subplots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 6 * nrows))
        axes = axes.flatten() if num_subplots > 1 else [axes]

    # Get unique models
    models = sorted(df['model_name'].unique())

    # Color map for models
    colors = plt.cm.tab10(range(len(models)))
    model_colors = dict(zip(models, colors))

    # Plot each round in a separate subplot
    for idx, rounds in enumerate(rounds_list):
        ax = axes[idx]
        round_data = df[df['num_rounds'] == rounds]

        # Plot one line per model
        for model in models:
            model_data = round_data[round_data['model_name'] == model].sort_values('num_agents')

            if len(model_data) > 0:
                ax.plot(model_data['num_agents'],
                       model_data['avg_accuracy'],
                       marker='o',
                       label=model,
                       linewidth=2,
                       markersize=8,
                       color=model_colors[model])

        ax.set_xlabel('Number of Agents', fontsize=11)
        ax.set_ylabel('Average Accuracy', fontsize=11)
        ax.set_title(f'{rounds} Rounds', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Set integer ticks for x-axis
        if len(round_data) > 0:
            ax.set_xticks(sorted(round_data['num_agents'].unique()))

    # Hide unused subplots
    for idx in range(num_subplots, len(axes)):
        axes[idx].set_visible(False)

    # Overall title
    fig.suptitle(f'{task.upper()} - Model Comparison', fontsize=16, fontweight='bold', y=1.0)

    plt.tight_layout()

    # Save plot
    filename = f"{task}_comparison.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Generate Task Comparison Plots")
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

    # Group by task
    tasks = sorted(df['task'].unique())

    print(f"Generating {len(tasks)} comparison plots...\n")

    for task in tasks:
        print(f"Processing task: {task}")
        task_df = df[df['task'] == task]
        plot_task_comparison(task_df, task, output_dir)

    print("\n" + "=" * 60)
    print(f"All plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
