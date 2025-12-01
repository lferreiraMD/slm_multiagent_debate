#!/bin/bash

# Quick setup script for Linux baseline and persona diversity experiments
# Generates job configs for both baseline and persona experiments
# Makes scripts executable

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo "Linux Baseline & Persona Diversity Experiments - Setup"
echo "=================================================="
echo ""

# 1. Make all scripts executable
echo "[1/4] Making scripts executable..."
chmod +x "$SCRIPT_DIR"/*.sh
echo "  ✓ All .sh files are now executable"
echo ""

# 2. Generate baseline job configurations
echo "[2/4] Generating baseline job configurations..."
if python3 "$SCRIPT_DIR/generate_baseline_configs.py"; then
    echo "  ✓ Baseline job configs generated successfully"
else
    echo "  ✗ Failed to generate baseline job configs"
    exit 1
fi
echo ""

# 3. Generate persona job configurations
echo "[3/4] Generating persona job configurations..."
if python3 "$SCRIPT_DIR/generate_job_configs.py"; then
    echo "  ✓ Persona job configs generated successfully"
else
    echo "  ✗ Failed to generate persona job configs"
    exit 1
fi
echo ""

# 4. Create log directories
echo "[4/4] Creating log directories..."
mkdir -p "$SCRIPT_DIR/logs"/{math,gsm,biography,mmlu,test}
echo "  ✓ Log directories created"
echo ""

echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Test system:"
echo "     bash $SCRIPT_DIR/test_system.sh"
echo ""
echo "  2. Run experiments:"
echo "     bash $SCRIPT_DIR/run_all_experiments.sh"
echo ""
echo "  3. Monitor progress (in separate terminal):"
echo "     bash $SCRIPT_DIR/monitor_experiments.sh"
echo "=================================================="
