#!/bin/bash

# Quick setup script for Linux persona diversity experiments
# Prepares job configs and makes scripts executable

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo "Linux Persona Diversity Experiments - Setup"
echo "=================================================="
echo ""

# 1. Make all scripts executable
echo "[1/3] Making scripts executable..."
chmod +x "$SCRIPT_DIR"/*.sh
echo "  ✓ All .sh files are now executable"
echo ""

# 2. Generate job configurations
echo "[2/3] Generating job configurations..."
if python3 "$SCRIPT_DIR/generate_job_configs.py"; then
    echo "  ✓ Job configs generated successfully"
else
    echo "  ✗ Failed to generate job configs"
    exit 1
fi
echo ""

# 3. Create log directories
echo "[3/3] Creating log directories..."
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
