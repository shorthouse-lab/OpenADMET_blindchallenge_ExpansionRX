#!/bin/bash
# quick_start.sh
# Quick start script for OpenADMET experiments

set -e  # Exit on error

echo "=========================================="
echo "OpenADMET with ChemGridML - Quick Start"
echo "=========================================="
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "chemgridml" ]]; then
    echo "⚠️  ChemGridML conda environment not activated"
    echo "Please run: conda activate chemgridml"
    echo ""
    read -p "Would you like me to try activating it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        eval "$(conda shell.bash hook)"
        conda activate chemgridml
    else
        exit 1
    fi
fi

echo "✓ Environment: $CONDA_DEFAULT_ENV"
echo ""

# Check if training data exists
if [ ! -f "ExpansionRX_train.csv" ]; then
    echo "✗ Error: ExpansionRX_train.csv not found"
    echo "Please ensure the training data is in the current directory"
    exit 1
fi

echo "✓ Found training data: ExpansionRX_train.csv"
echo ""

# Check data availability
echo "Checking data availability..."
echo ""
python check_data.py --csv ExpansionRX_train.csv

echo ""
echo "=========================================="
echo "Ready to run experiments!"
echo "=========================================="
echo ""
echo "Choose an option:"
echo "  1. Quick test (fast, minimal config)"
echo "  2. Full benchmark (comprehensive, slow)"
echo "  3. Custom config"
echo "  4. Exit"
echo ""

read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "Running quick test with minimal configuration..."
        python run_openadmet.py --config openadmet_config_quicktest.yaml
        echo ""
        echo "Analyzing results..."
        python analyze_results.py --results-dir openadmet_results_test --experiment-name OpenADMET_QuickTest
        ;;
    2)
        echo ""
        echo "⚠️  Warning: Full benchmark may take several hours!"
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python run_openadmet.py --config openadmet_config.yaml
            echo ""
            echo "Analyzing results..."
            python analyze_results.py --results-dir openadmet_results
        fi
        ;;
    3)
        read -p "Enter config file path: " config_path
        if [ -f "$config_path" ]; then
            python run_openadmet.py --config "$config_path"
        else
            echo "✗ Config file not found: $config_path"
            exit 1
        fi
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✓ Complete!"
echo "=========================================="
