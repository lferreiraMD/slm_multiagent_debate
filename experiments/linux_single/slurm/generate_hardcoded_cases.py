#!/usr/bin/env python3
"""
Generate hardcoded case statement for SLURM script from persona CSV files.
This eliminates runtime CSV parsing - everything is hardcoded at generation time.
"""

import csv
import ast
from pathlib import Path

# Paths
CONFIG_DIR = Path(__file__).parent.parent / "configs"
TASKS = [
    ("math", "gen_math.py", "--num-problems", 15),
    ("gsm", "gen_gsm.py", "--num-problems", 15),
    ("biography", "gen_conversation.py", "--num-people", 15),
    ("mmlu", "gen_mmlu.py", "--num-questions", 15)
]

def escape_bash(s):
    """Escape string for bash single quotes."""
    return s.replace("'", "'\"'\"'")

def generate_case_statement():
    """Generate complete case statement for all 60 jobs."""

    print("# Generated case statement - DO NOT EDIT MANUALLY")
    print("# Regenerate with: python3 generate_hardcoded_cases.py")
    print("")
    print("case $TASK_ID in")

    task_id = 1

    for task_name, script_name, num_param, count in TASKS:
        config_file = CONFIG_DIR / f"persona_{task_name}_jobs.txt"

        with open(config_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                job_id = row['job_id']
                model = row['model_alias']
                n_agents = row['n_agents']
                rounds = row['rounds']
                num_value = row['num_value']
                personas_tuple = row['personas_tuple']

                # Parse personas tuple
                try:
                    personas = ast.literal_eval(personas_tuple)
                except:
                    personas = []

                # Build persona arguments
                persona_args = []
                for p in personas:
                    escaped = escape_bash(p)
                    persona_args.append(f"'{escaped}'")

                # Generate case entry
                print(f"    {task_id})")
                print(f"        echo \"[Task {task_id}] {task_name.upper()} - job {job_id}: {model}, {n_agents} agents\"")
                print(f"        cd \"$PROJECT_ROOT/tasks/{task_name}\"")
                print(f"        srun -p bch-gpu-pe -c 1 --mem 16G --gpus 1 \\")
                print(f"            python3 {script_name} \\")
                print(f"                --model {model} \\")
                print(f"                --agents {n_agents} \\")
                print(f"                --rounds {rounds} \\")
                print(f"                {num_param} {num_value} \\")

                if persona_args:
                    print(f"                --agent-personas \\")
                    for i, pa in enumerate(persona_args):
                        if i < len(persona_args) - 1:
                            print(f"                    {pa} \\")
                        else:
                            print(f"                    {pa} \\")

                print(f"                --output-directory \"$RESULTS_DIR/{task_name}\" \\")
                print(f"            > \"$LOG_DIR/{task_name}/job_{job_id}_task{task_id}.out\" 2>&1")
                print(f"        EXIT_CODE=$?")
                print(f"        ;;")
                print("")

                task_id += 1

    print("    *)")
    print("        echo \"ERROR: Invalid task ID: $TASK_ID\"")
    print("        exit 1")
    print("        ;;")
    print("esac")

if __name__ == "__main__":
    generate_case_statement()
