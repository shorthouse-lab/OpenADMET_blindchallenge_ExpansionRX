#!/usr/bin/env python
# run_openadmet.py
"""
Main runner script for OpenADMET blind challenge experiments.
Reads configuration file and runs experiments for selected models, features, and targets.
"""

import os
import sys
import yaml
import time
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import sqlite3
from itertools import product

# Add code directory to path
sys.path.append(str(Path(__file__).parent / "code"))

import env
from experiments import Method
from openadmet_dataset import OpenADMETDataset
from study_manager import StudyManager
from database_manager import DatabaseManager


class OpenADMETRunner:
    """Runner for OpenADMET experiments"""
    
    def __init__(self, config_path: str):
        """
        Initialize runner with configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.results_dir = Path(self.config['output']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Override env.py settings with config
        self._setup_environment()
        
        # Determine which targets to run
        self.targets = self._get_targets()
        
        print(f"OpenADMET Runner initialized")
        print(f"Results directory: {self.results_dir}")
        print(f"Targets to predict: {self.targets}")
        
    def _load_config(self) -> dict:
        """Load YAML configuration file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_environment(self):
        """Update environment settings from config"""
        exp_config = self.config['experiment']
        env.N_TESTS = exp_config.get('n_tests', 10)
        env.N_FOLDS = exp_config.get('n_folds', 5)
        env.N_TRIALS = exp_config.get('n_trials', 15)
        env.TEST_SIZE = exp_config.get('test_size', 0.2)
        
        print(f"Environment configured:")
        print(f"  N_TESTS: {env.N_TESTS}")
        print(f"  N_FOLDS: {env.N_FOLDS}")
        print(f"  N_TRIALS: {env.N_TRIALS}")
        print(f"  TEST_SIZE: {env.TEST_SIZE}")
        print(f"  DEVICE: {env.DEVICE}")
    
    def _get_targets(self) -> list:
        """Get list of target properties to predict"""
        configured_targets = self.config.get('targets', [])
        
        if not configured_targets:
            # Auto-detect available targets
            csv_path = self.config['data']['train_csv']
            smiles_col = self.config['data'].get('smiles_column', 'SMILES')
            available = OpenADMETDataset.get_available_targets(csv_path, smiles_col)
            targets = list(available.keys())
            print(f"Auto-detected {len(targets)} targets with available data")
            for target, count in available.items():
                print(f"  {target}: {count} data points")
        else:
            targets = configured_targets
            
        return targets
    
    def _create_methods(self) -> list:
        """Create all Method combinations from config"""
        features = self.config['features']
        models = self.config['models']
        
        # Create dataset names in the format expected by openadmet_dataset.py
        datasets = [self._format_dataset_name(target) for target in self.targets]
        
        # Create all combinations
        methods = [
            Method(feature=f, model=m, dataset=d)
            for f, m, d in product(features, models, datasets)
        ]
        
        # Filter out invalid combinations (e.g., GNN models without GRAPH features)
        valid_methods = []
        for method in methods:
            if method.model in ['GCN', 'GAT'] and method.feature != 'GRAPH':
                print(f"Skipping {method}: {method.model} requires GRAPH features")
                continue
            valid_methods.append(method)
        
        return valid_methods
    
    def _format_dataset_name(self, target: str) -> str:
        """Format target property name to dataset name"""
        # Replace spaces with underscores for dataset naming
        return f"OpenADMET_{target.replace(' ', '_')}"
    
    def _unformat_dataset_name(self, dataset_name: str) -> str:
        """Extract target property name from dataset name"""
        return dataset_name.replace("OpenADMET_", "").replace("_", " ")
    
    def _is_method_complete(self, method: Method) -> bool:
        """
        Check if a method has already been completed.
        
        Args:
            method: Method to check
            
        Returns:
            True if method has complete predictions in database
        """
        experiment_name = self.config['experiment']['name']
        predictions_path = self.results_dir / f"predictions_{experiment_name}.db"
        
        # If database doesn't exist, nothing is complete
        if not predictions_path.exists():
            return False
        
        try:
            conn = sqlite3.connect(str(predictions_path))
            cursor = conn.cursor()
            
            # Check if predictions exist for this method for all expected seeds
            query = """
                SELECT COUNT(DISTINCT seed) 
                FROM predictions 
                WHERE dataset_name = ? 
                AND fingerprint = ? 
                AND model_name = ?
            """
            
            cursor.execute(query, (method.dataset, method.feature, method.model))
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] == env.N_TESTS:
                return True
            
        except Exception as e:
            # If there's any error checking, assume not complete
            pass
        
        return False
    
    def _get_completed_methods(self, methods: list) -> tuple:
        """
        Separate methods into completed and pending.
        
        Args:
            methods: List of Method objects
            
        Returns:
            Tuple of (completed_methods, pending_methods)
        """
        completed = []
        pending = []
        
        for method in methods:
            if self._is_method_complete(method):
                completed.append(method)
            else:
                pending.append(method)
        
        return completed, pending
    
    def run_single_method(self, method: Method, skip_if_complete: bool = True) -> dict:
        """
        Run experiment for a single method.
        
        Args:
            method: Method object specifying feature, model, and dataset
            skip_if_complete: If True, skip methods that are already complete
            
        Returns:
            Dictionary with execution results
        """
        # Check if already complete
        if skip_if_complete and self._is_method_complete(method):
            target_property = self._unformat_dataset_name(method.dataset)
            print(f"\n{'='*80}")
            print(f"Skipping (already complete): {method}")
            print(f"{'='*80}")
            
            return {
                'method': str(method),
                'feature': method.feature,
                'model': method.model,
                'target': target_property,
                'status': 'skipped',
                'n_samples': 0,
                'elapsed_time': 0
            }
        
        print(f"\n{'='*80}")
        print(f"Running: {method}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Create output paths
        experiment_name = self.config['experiment']['name']
        studies_path = str(self.results_dir / "studies" / str(method))
        predictions_path = str(self.results_dir / f"predictions_{experiment_name}.db")
        
        os.makedirs(studies_path, exist_ok=True)
        
        # Create custom dataset
        csv_path = self.config['data']['train_csv']
        target_property = self._unformat_dataset_name(method.dataset)
        
        try:
            # Load dataset
            dataset = OpenADMETDataset(
                csv_path=csv_path,
                target_property=target_property,
                feature_type=method.feature,
                smiles_column=self.config['data'].get('smiles_column', 'SMILES')
            )
            
            # Create modified study manager that uses our custom dataset
            manager = OpenADMETStudyManager(
                method=method,
                dataset=dataset,
                studies_path=studies_path,
                predictions_path=predictions_path
            )
            
            # Run nested cross-validation
            manager.run_nested_cv()
            
            elapsed_time = time.time() - start_time
            
            result = {
                'method': str(method),
                'feature': method.feature,
                'model': method.model,
                'target': target_property,
                'status': 'success',
                'n_samples': len(dataset.Y),
                'elapsed_time': elapsed_time
            }
            
            print(f"✓ Completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"✗ Failed after {elapsed_time:.2f} seconds: {str(e)}")
            
            result = {
                'method': str(method),
                'feature': method.feature,
                'model': method.model,
                'target': target_property,
                'status': 'failed',
                'error': str(e),
                'elapsed_time': elapsed_time
            }
        
        return result
    
    def run_all(self, parallel: bool = False, skip_completed: bool = True):
        """
        Run all configured experiments.
        
        Args:
            parallel: If True, run methods in parallel (not yet implemented)
            skip_completed: If True, skip already-completed methods
        """
        methods = self._create_methods()
        
        # Check for completed methods
        if skip_completed:
            completed_methods, pending_methods = self._get_completed_methods(methods)
            
            if completed_methods:
                print(f"\n{'='*80}")
                print(f"Found {len(completed_methods)} already-completed methods")
                print(f"{'='*80}")
                for method in completed_methods[:5]:  # Show first 5
                    print(f"  ✓ {method}")
                if len(completed_methods) > 5:
                    print(f"  ... and {len(completed_methods) - 5} more")
                print(f"{'='*80}\n")
            
            methods_to_run = pending_methods
        else:
            methods_to_run = methods
        
        print(f"\n{'='*80}")
        print(f"OpenADMET Benchmark")
        print(f"{'='*80}")
        print(f"Total methods: {len(methods)}")
        if skip_completed:
            print(f"Already completed: {len(methods) - len(methods_to_run)}")
            print(f"To run: {len(methods_to_run)}")
        else:
            print(f"To run: {len(methods_to_run)} (skip_completed=False)")
        print(f"Features: {self.config['features']}")
        print(f"Models: {self.config['models']}")
        print(f"Targets: {len(self.targets)}")
        print(f"{'='*80}\n")
        
        if not methods_to_run:
            print("✓ All methods already completed! Nothing to run.")
            return []
        
        results = []
        
        for i, method in enumerate(methods_to_run, 1):
            print(f"\nProgress: {i}/{len(methods_to_run)}")
            result = self.run_single_method(method, skip_if_complete=skip_completed)
            results.append(result)
            
            # Save intermediate results
            self._save_run_summary(results)
        
        print(f"\n{'='*80}")
        print(f"All experiments completed!")
        print(f"{'='*80}")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _save_run_summary(self, results: list):
        """Save summary of completed runs"""
        summary_path = self.results_dir / "run_summary.csv"
        df = pd.DataFrame(results)
        df.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")
    
    def _print_summary(self, results: list):
        """Print summary of results"""
        df = pd.DataFrame(results)
        
        n_success = (df['status'] == 'success').sum()
        n_failed = (df['status'] == 'failed').sum()
        n_skipped = (df['status'] == 'skipped').sum()
        total_time = df['elapsed_time'].sum()
        
        print(f"\nSummary:")
        print(f"  Successful: {n_success}/{len(results)}")
        if n_skipped > 0:
            print(f"  Skipped (already complete): {n_skipped}/{len(results)}")
        print(f"  Failed: {n_failed}/{len(results)}")
        print(f"  Total time: {total_time/60:.2f} minutes")
        
        if n_failed > 0:
            print(f"\nFailed experiments:")
            failed = df[df['status'] == 'failed']
            for _, row in failed.iterrows():
                print(f"  {row['method']}: {row.get('error', 'Unknown error')}")


class OpenADMETStudyManager(StudyManager):
    """Modified StudyManager that works with OpenADMETDataset"""
    
    def __init__(self, method: Method, dataset: OpenADMETDataset, 
                 studies_path: str, predictions_path: str):
        """
        Initialize with pre-loaded dataset.
        
        Args:
            method: Method specification
            dataset: Pre-loaded OpenADMETDataset
            studies_path: Path to store study files
            predictions_path: Path to predictions database
        """
        self.method = method
        self.studies_path = studies_path
        self.dataset = dataset  # Store the dataset
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        self.db = DatabaseManager(predictions_path)
        self.optuna_init = False
    
    def run_nested_cv(self):
        """
        Run nested cross-validation using pre-loaded dataset.
        Override parent method to use self.dataset instead of creating new one.
        """
        # Use the pre-loaded dataset
        data = self.dataset
        
        # Store dataset targets
        self.db.store_dataset_targets(self.method.dataset, data.Y)
        
        # Run experiments (rest is same as parent class)
        predictions = [None for _ in range(env.N_TESTS)]
        indices = [None for _ in range(env.N_TESTS)]
        
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from sklearn.model_selection import train_test_split
        
        # Prevent thread oversubscription inside child processes (MKL/BLAS/Numexpr etc.)
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
        os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

        # Determine seeds already completed for this method (so we can resume)
        existing_seeds = set()
        try:
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT DISTINCT seed FROM predictions
                    WHERE dataset_name = ? AND fingerprint = ? AND model_name = ?
                    """,
                    (self.method.dataset, self.method.feature, self.method.model),
                )
                rows = cur.fetchall()
                existing_seeds = {int(r[0]) for r in rows if r and r[0] is not None}
        except Exception:
            # If anything goes wrong, assume nothing completed
            existing_seeds = set()

        seeds_to_run = [s for s in range(env.N_TESTS) if s not in existing_seeds]
        if len(existing_seeds) > 0:
            print(f"Resume: found {len(existing_seeds)}/{env.N_TESTS} completed seeds for {self.method}; will run remaining {len(seeds_to_run)} seeds.")
        else:
            print(f"No completed seeds found for {self.method}; running all {env.N_TESTS} seeds.")

        if not seeds_to_run:
            print(f"All seeds already completed for {self.method}; nothing to do.")
            return

        # Compute safe level of seed-parallelism accounting for sklearn per-seed threads
        allocated_cores = int(os.environ.get('NSLOTS', multiprocessing.cpu_count()))
        per_seed_parallelism = env.N_JOBS_SKLEARN
        theoretical = allocated_cores // max(1, per_seed_parallelism)
        max_workers = max(1, min(theoretical, len(seeds_to_run)))
        if env.DEVICE != 'cpu':
            max_workers = 1
        print(f"Parallel seed workers: {max_workers} (allocated cores={allocated_cores}, per-seed threads={per_seed_parallelism})")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_seed = {
                executor.submit(self.run_single_experiment, seed, data): seed
                for seed in seeds_to_run
            }
            
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    seed_result, test_predictions, test_indices = future.result()
                    predictions[seed_result] = test_predictions
                    indices[seed_result] = test_indices
                    # Save this seed's predictions immediately for safe resume
                    self.db.store_predictions(
                        self.method.dataset, self.method.feature, self.method.model,
                        test_predictions, test_indices, seed_result, 'random'
                    )
                    print(f"[Store] {self.method} | saved predictions for seed {seed_result}", flush=True)
                except Exception as exc:
                    print(f"Seed {seed} failed: {exc}")
                    raise exc
        
        # Optional: final pass to ensure all seeds present (idempotent due to REPLACE)
        for seed in range(env.N_TESTS):
            if predictions[seed] is not None:
                self.db.store_predictions(
                    self.method.dataset, self.method.feature, self.method.model,
                    predictions[seed], indices[seed], seed, 'random'
                )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run OpenADMET blind challenge experiments with ChemGridML"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='openadmet_config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--method',
        type=str,
        help='Run single method in format: feature_model_dataset (e.g., ECFP_RF_OpenADMET_LogD)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run of all methods, ignoring checkpoints (default: skip completed)'
    )
    parser.add_argument(
        '--list-completed',
        action='store_true',
        help='List all completed methods and exit'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity (overrides CHEMGRID_VERBOSE)'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    # Override verbosity if --quiet passed
    if args.quiet:
        os.environ['CHEMGRID_VERBOSE'] = '0'
    runner = OpenADMETRunner(args.config)
    
    # List completed methods if requested
    if args.list_completed:
        methods = runner._create_methods()
        completed, pending = runner._get_completed_methods(methods)
        
        print(f"\n{'='*80}")
        print(f"Completed Methods: {len(completed)}/{len(methods)}")
        print(f"{'='*80}")
        
        if completed:
            for method in completed:
                print(f"  ✓ {method}")
        else:
            print("  No completed methods found")
        
        print(f"\n{'='*80}")
        print(f"Pending Methods: {len(pending)}/{len(methods)}")
        print(f"{'='*80}")
        
        if pending:
            for method in pending:
                print(f"  ○ {method}")
        else:
            print("  All methods completed!")
        
        return
    
    # Run experiments
    if args.method:
        # Run single method
        parts = args.method.split('_')
        if len(parts) < 3:
            print(f"Error: method must be in format feature_model_dataset")
            sys.exit(1)
        
        feature = parts[0]
        model = parts[1]
        dataset = '_'.join(parts[2:])
        
        method = Method(feature=feature, model=model, dataset=dataset)
        skip_if_complete = not args.force
        runner.run_single_method(method, skip_if_complete=skip_if_complete)
    else:
        # Run all configured experiments
        skip_completed = not args.force
        runner.run_all(skip_completed=skip_completed)
    
    print(f"\nResults saved to: {runner.results_dir}")


if __name__ == '__main__':
    main()
