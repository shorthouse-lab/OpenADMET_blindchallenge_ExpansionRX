#!/usr/bin/env python
# check_data.py
"""
Utility script to check available targets and data coverage in ExpansionRX dataset.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add code directory to path
sys.path.append(str(Path(__file__).parent / "code"))

from openadmet_dataset import OpenADMETDataset


def check_data_availability(csv_path: str, smiles_column: str = 'SMILES'):
    """
    Check and display data availability for all targets.
    
    Args:
        csv_path: Path to CSV file
        smiles_column: Name of SMILES column
    """
    print("="*80)
    print(f"Checking data in: {csv_path}")
    print("="*80)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"SMILES column: {smiles_column}")
    
    # Get available targets
    print("\nAnalyzing target properties...")
    availability = OpenADMETDataset.get_available_targets(csv_path, smiles_column)
    
    print("\n" + "="*80)
    print("Target Property Data Availability")
    print("="*80)
    print(f"{'Target':<40} {'Available':<12} {'Coverage':<12}")
    print("-"*80)
    
    total_samples = len(df)
    
    # Sort by availability (descending)
    sorted_targets = sorted(availability.items(), key=lambda x: x[1], reverse=True)
    
    for target, count in sorted_targets:
        coverage = (count / total_samples) * 100
        print(f"{target:<40} {count:<12} {coverage:>6.1f}%")
    
    print("="*80)
    print(f"\nTotal targets found: {len(availability)}")
    
    # Recommendations
    print("\nRecommendations:")
    print("-"*80)
    
    high_coverage = [(t, c) for t, c in sorted_targets if (c / total_samples) > 0.5]
    medium_coverage = [(t, c) for t, c in sorted_targets if 0.2 < (c / total_samples) <= 0.5]
    low_coverage = [(t, c) for t, c in sorted_targets if (c / total_samples) <= 0.2]
    
    if high_coverage:
        print(f"\n✓ High coverage targets (>50%): {len(high_coverage)}")
        for target, count in high_coverage[:5]:
            print(f"  - {target}")
    
    if medium_coverage:
        print(f"\n⚠ Medium coverage targets (20-50%): {len(medium_coverage)}")
        for target, count in medium_coverage[:5]:
            print(f"  - {target}")
    
    if low_coverage:
        print(f"\n✗ Low coverage targets (<20%): {len(low_coverage)}")
        print("  Consider excluding these unless specifically needed.")
    
    print("\n" + "="*80)
    
    return availability


def generate_config_from_data(csv_path: str, output_path: str = "openadmet_config_auto.yaml",
                              min_coverage: float = 0.3):
    """
    Generate a config file with targets that have sufficient data coverage.
    
    Args:
        csv_path: Path to CSV file
        output_path: Path to save generated config
        min_coverage: Minimum coverage threshold (0.0 to 1.0)
    """
    import yaml
    
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    
    availability = OpenADMETDataset.get_available_targets(csv_path)
    
    # Filter targets by coverage
    good_targets = [
        target for target, count in availability.items()
        if (count / total_samples) >= min_coverage
    ]
    
    # Create config
    config = {
        'data': {
            'train_csv': csv_path,
            'test_csv': csv_path.replace('train', 'test'),
            'smiles_column': 'SMILES'
        },
        'targets': good_targets,
        'features': ['ECFP', 'MACCS', 'RDKitFP', 'MOL2VEC'],
        'models': ['RF', 'XGBoost', 'FNN'],
        'experiment': {
            'name': 'OpenADMET_Benchmark',
            'n_tests': 10,
            'n_folds': 5,
            'n_trials': 15,
            'test_size': 0.2
        },
        'output': {
            'results_dir': 'openadmet_results',
            'save_predictions': True,
            'save_models': False
        },
        'visualization': {
            'create_plots': True,
            'plot_types': ['performance_comparison', 'target_distribution', 
                          'prediction_scatter', 'feature_importance'],
            'figure_format': 'png',
            'dpi': 300
        }
    }
    
    # Save config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Generated config file: {output_path}")
    print(f"  Included {len(good_targets)} targets with ≥{min_coverage*100:.0f}% coverage")
    print(f"\nTo use this config:")
    print(f"  python run_openadmet.py --config {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Check data availability in OpenADMET CSV files"
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='ExpansionRX_train.csv',
        help='Path to CSV file'
    )
    parser.add_argument(
        '--smiles-column',
        type=str,
        default='SMILES',
        help='Name of SMILES column'
    )
    parser.add_argument(
        '--generate-config',
        action='store_true',
        help='Generate config file with available targets'
    )
    parser.add_argument(
        '--min-coverage',
        type=float,
        default=0.3,
        help='Minimum data coverage for auto-generated config (0.0-1.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='openadmet_config_auto.yaml',
        help='Output path for generated config'
    )
    
    args = parser.parse_args()
    
    # Check data availability
    availability = check_data_availability(args.csv, args.smiles_column)
    
    # Generate config if requested
    if args.generate_config:
        generate_config_from_data(
            args.csv,
            output_path=args.output,
            min_coverage=args.min_coverage
        )


if __name__ == '__main__':
    main()
