#!/usr/bin/env python
# analyze_results.py
"""
Analysis and visualization module for OpenADMET experiments.
Generates plots, performance metrics, and human-readable reports.
"""

import sys
import argparse
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class ResultsAnalyzer:
    """Analyzer for OpenADMET experiment results"""
    
    def __init__(self, results_dir: str, experiment_name: str = "OpenADMET_Benchmark"):
        """
        Initialize analyzer.
        
        Args:
            results_dir: Directory containing results
            experiment_name: Name of the experiment
        """
        self.results_dir = Path(results_dir)
        self.experiment_name = experiment_name
        self.db_path = self.results_dir / f"predictions_{experiment_name}.db"
        
        # Create output directories
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.reports_dir = self.results_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        print(f"Analyzing results from: {self.db_path}")
    
    def load_all_predictions(self) -> pd.DataFrame:
        """Load all predictions from database"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT p.*, dt.target_value
            FROM predictions p
            JOIN dataset_targets dt ON p.dataset_name = dt.dataset_name 
                                    AND p.data_index = dt.data_index
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Add human-readable target name
        df['target'] = df['dataset_name'].str.replace('OpenADMET_', '').str.replace('_', ' ')
        
        return df
    
    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance metrics for each method.
        
        Args:
            df: DataFrame with predictions and target values
            
        Returns:
            DataFrame with metrics for each feature-model-target combination
        """
        metrics_list = []
        
        grouped = df.groupby(['fingerprint', 'model_name', 'target', 'seed'])
        
        for (fingerprint, model, target, seed), group in grouped:
            y_true = group['target_value'].values
            y_pred = group['prediction'].values
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            pearson_r, _ = pearsonr(y_true, y_pred)
            spearman_r, _ = spearmanr(y_true, y_pred)
            
            # Calculate relative errors (percentage-based)
            # MAPE: Mean Absolute Percentage Error
            # Handle division by zero by filtering out near-zero true values
            eps = 1e-10
            abs_errors = np.abs(y_true - y_pred)
            relative_errors = abs_errors / (np.abs(y_true) + eps)
            mape = np.mean(relative_errors)  # Keep as fraction (0-1 scale), not percentage
            
            # Relative RMSE (NRMSE) - normalized by mean of true values
            mean_y = np.mean(np.abs(y_true))
            nrmse = (rmse / (mean_y + eps)) if mean_y > eps else 0  # Keep as fraction (0-1 scale)
            
            metrics_list.append({
                'feature': fingerprint,
                'model': model,
                'target': target,
                'seed': seed,
                'n_samples': len(y_true),
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Pearson_R': pearson_r,
                'Spearman_R': spearman_r,
                'MAPE': mape,  # Mean Absolute Percentage Error (0-1 scale)
                'NRMSE': nrmse  # Normalized RMSE (0-1 scale)
            })
        
        return pd.DataFrame(metrics_list)
    
    def aggregate_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate metrics across seeds (calculate mean and std).
        
        Args:
            metrics_df: DataFrame with metrics for each seed
            
        Returns:
            DataFrame with aggregated metrics
        """
        agg_metrics = []
        
        grouped = metrics_df.groupby(['feature', 'model', 'target'])
        
        for (feature, model, target), group in grouped:
            agg = {
                'feature': feature,
                'model': model,
                'target': target,
                'n_samples': int(group['n_samples'].mean()),
                'n_seeds': len(group)
            }
            
            # Calculate mean and std for each metric
            for metric in ['RMSE', 'MAE', 'R2', 'Pearson_R', 'Spearman_R', 'MAPE', 'NRMSE']:
                agg[f'{metric}_mean'] = group[metric].mean()
                agg[f'{metric}_std'] = group[metric].std()
            
            agg_metrics.append(agg)
        
        return pd.DataFrame(agg_metrics)
    
    def create_performance_comparison_plot(self, metrics_df: pd.DataFrame, 
                                          metric: str = 'R2', 
                                          save_path: Path = None):
        """
        Create heatmap comparing model performance across features and targets.
        
        Args:
            metrics_df: Aggregated metrics DataFrame
            metric: Metric to plot (R2, RMSE, MAE, etc.)
            save_path: Path to save figure
        """
        # Get unique targets
        targets = metrics_df['target'].unique()
        
        # Create subplot for each target
        n_targets = len(targets)
        n_cols = min(3, n_targets)
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_targets == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, target in enumerate(targets):
            ax = axes[idx]
            
            # Filter data for this target
            target_data = metrics_df[metrics_df['target'] == target]
            
            # Pivot to create heatmap
            pivot = target_data.pivot_table(
                values=f'{metric}_mean',
                index='model',
                columns='feature',
                aggfunc='mean'
            )
            
            # Create heatmap
            sns.heatmap(
                pivot, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlGn' if metric in ['R2', 'Pearson_R', 'Spearman_R'] else 'RdYlGn_r',
                ax=ax,
                cbar_kws={'label': metric}
            )
            
            ax.set_title(f'{target} - {metric}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Feature', fontsize=10)
            ax.set_ylabel('Model', fontsize=10)
        
        # Hide unused subplots
        for idx in range(n_targets, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def create_prediction_scatter_plots(self, df: pd.DataFrame, save_dir: Path = None):
        """
        Create scatter plots of predicted vs actual values for best models.
        
        Args:
            df: DataFrame with predictions
            save_dir: Directory to save figures
        """
        if save_dir is None:
            save_dir = self.plots_dir / "scatter"
        save_dir.mkdir(exist_ok=True)
        
        # Calculate metrics to find best model for each target
        metrics_df = self.calculate_metrics(df)
        agg_metrics = self.aggregate_metrics(metrics_df)
        
        # For each target, find the best model (highest R2)
        for target in agg_metrics['target'].unique():
            target_metrics = agg_metrics[agg_metrics['target'] == target]
            best_idx = target_metrics['R2_mean'].idxmax()
            best = target_metrics.loc[best_idx]
            
            # Get predictions for best model
            mask = (
                (df['target'] == target) &
                (df['fingerprint'] == best['feature']) &
                (df['model_name'] == best['model'])
            )
            plot_data = df[mask]
            
            if len(plot_data) == 0:
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Get unique seeds and create a colormap
            seeds = sorted(plot_data['seed'].unique())
            colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
            
            # Plot each seed with different color
            for i, seed in enumerate(seeds):
                seed_data = plot_data[plot_data['seed'] == seed]
                ax.scatter(
                    seed_data['target_value'],
                    seed_data['prediction'],
                    alpha=0.6,
                    s=30,
                    color=colors[i],
                    label=f'Seed {seed}',
                    edgecolors='white',
                    linewidth=0.5
                )
            
            # Add diagonal line
            min_val = min(plot_data['target_value'].min(), plot_data['prediction'].min())
            max_val = max(plot_data['target_value'].max(), plot_data['prediction'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.5)
            
            # Labels and title
            ax.set_xlabel('Actual Value', fontsize=12)
            ax.set_ylabel('Predicted Value', fontsize=12)
            title = f"{target}\nBest: {best['feature']} + {best['model']}\n"
            title += f"R² = {best['R2_mean']:.3f} ± {best['R2_std']:.3f}"
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend(loc='upper left', framealpha=0.9, fontsize=8, ncol=2)
            
            plt.tight_layout()
            
            # Save
            safe_name = target.replace(' ', '_').replace('>', '').replace('<', '')
            save_path = save_dir / f"scatter_{safe_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            
            plt.close()
    
    def create_feature_comparison_plot(self, metrics_df: pd.DataFrame, save_path: Path = None):
        """
        Create box plots comparing features across all models and targets.
        
        Args:
            metrics_df: Metrics DataFrame (per seed, not aggregated)
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics_to_plot = ['R2', 'RMSE', 'MAE']
        
        for ax, metric in zip(axes, metrics_to_plot):
            sns.boxplot(
                data=metrics_df,
                x='feature',
                y=metric,
                ax=ax
            )
            ax.set_title(f'{metric} by Feature', fontsize=12, fontweight='bold')
            ax.set_xlabel('Feature Type', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def create_model_comparison_plot(self, metrics_df: pd.DataFrame, save_path: Path = None):
        """
        Create box plots comparing models across all features and targets.
        
        Args:
            metrics_df: Metrics DataFrame (per seed, not aggregated)
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics_to_plot = ['R2', 'RMSE', 'MAE']
        
        for ax, metric in zip(axes, metrics_to_plot):
            sns.boxplot(
                data=metrics_df,
                x='model',
                y=metric,
                ax=ax
            )
            ax.set_title(f'{metric} by Model', fontsize=12, fontweight='bold')
            ax.set_xlabel('Model Type', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def create_error_by_target_plots(self, metrics_df: pd.DataFrame, save_dir: Path = None):
        """
        Create plots showing error metrics across targets, split by fingerprint and model.
        
        Args:
            metrics_df: Metrics DataFrame (per seed, not aggregated)
            save_dir: Directory to save figures
        """
        if save_dir is None:
            save_dir = self.plots_dir / "error_by_target"
        save_dir.mkdir(exist_ok=True)
        
        # Get unique targets
        targets = sorted(metrics_df['target'].unique())
        
        # Error metrics to plot (now includes relative errors)
        error_metrics = ['RMSE', 'MAE', 'MAPE', 'NRMSE']
        
        for error_metric in error_metrics:
            # 1. Error split by fingerprint (feature)
            fig, ax = plt.subplots(figsize=(max(10, len(targets)*1.5), 6))
            
            # Prepare data for grouped bar plot
            fingerprints = sorted(metrics_df['feature'].unique())
            x = np.arange(len(targets))
            width = 0.8 / len(fingerprints)
            
            for i, fingerprint in enumerate(fingerprints):
                fp_data = metrics_df[metrics_df['feature'] == fingerprint]
                means = []
                stds = []
                
                for target in targets:
                    target_data = fp_data[fp_data['target'] == target]
                    if len(target_data) > 0:
                        means.append(target_data[error_metric].mean())
                        stds.append(target_data[error_metric].std())
                    else:
                        means.append(0)
                        stds.append(0)
                
                offset = (i - len(fingerprints)/2 + 0.5) * width
                # Convert to percentage for display if needed
                display_means = [m * 100 if error_metric in ['MAPE', 'NRMSE'] else m for m in means]
                display_stds = [s * 100 if error_metric in ['MAPE', 'NRMSE'] else s for s in stds]
                
                ax.bar(x + offset, display_means, width, label=fingerprint, 
                       yerr=display_stds, capsize=3, alpha=0.8)
            
            ax.set_xlabel('Target Property', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{error_metric}' + (' (%)' if error_metric in ['MAPE', 'NRMSE'] else ''), 
                         fontsize=12, fontweight='bold')
            ax.set_title(f'{error_metric} Across Targets by Fingerprint', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(targets, rotation=45, ha='right')
            ax.legend(title='Fingerprint', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            save_path = save_dir / f"{error_metric}_by_fingerprint.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
            
            # 2. Error split by model
            fig, ax = plt.subplots(figsize=(max(10, len(targets)*1.5), 6))
            
            models = sorted(metrics_df['model'].unique())
            width = 0.8 / len(models)
            
            for i, model in enumerate(models):
                model_data = metrics_df[metrics_df['model'] == model]
                means = []
                stds = []
                
                for target in targets:
                    target_data = model_data[model_data['target'] == target]
                    if len(target_data) > 0:
                        means.append(target_data[error_metric].mean())
                        stds.append(target_data[error_metric].std())
                    else:
                        means.append(0)
                        stds.append(0)
                
                offset = (i - len(models)/2 + 0.5) * width
                # Convert to percentage for display if needed
                display_means = [m * 100 if error_metric in ['MAPE', 'NRMSE'] else m for m in means]
                display_stds = [s * 100 if error_metric in ['MAPE', 'NRMSE'] else s for s in stds]
                
                ax.bar(x + offset, display_means, width, label=model, 
                       yerr=display_stds, capsize=3, alpha=0.8)
            
            ax.set_xlabel('Target Property', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{error_metric}' + (' (%)' if error_metric in ['MAPE', 'NRMSE'] else ''), 
                         fontsize=12, fontweight='bold')
            ax.set_title(f'{error_metric} Across Targets by Model', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(targets, rotation=45, ha='right')
            ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            save_path = save_dir / f"{error_metric}_by_model.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
        
        # 3. Create heatmaps for RMSE, MAE, MAPE, and NRMSE across fingerprint-target and model-target
        for error_metric in error_metrics:
            # Fingerprint x Target heatmap
            fig, ax = plt.subplots(figsize=(max(10, len(targets)*0.8), 
                                           max(6, len(fingerprints)*0.6)))
            
            # Aggregate across models and seeds
            pivot_fp = metrics_df.groupby(['feature', 'target'])[error_metric].mean().unstack()
            
            # Convert to percentage for display if needed
            if error_metric in ['MAPE', 'NRMSE']:
                pivot_fp = pivot_fp * 100
            
            ylabel = f'{error_metric}' + (' (%)' if error_metric in ['MAPE', 'NRMSE'] else '')
            sns.heatmap(pivot_fp, annot=True, fmt='.2f' if error_metric in ['MAPE', 'NRMSE'] else '.3f', 
                       cmap='YlOrRd', ax=ax, cbar_kws={'label': ylabel})
            ax.set_title(f'{error_metric} Heatmap: Fingerprint × Target', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Target Property', fontsize=12)
            ax.set_ylabel('Fingerprint', fontsize=12)
            
            plt.tight_layout()
            save_path = save_dir / f"{error_metric}_heatmap_fingerprint_target.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
            
            # Model x Target heatmap
            fig, ax = plt.subplots(figsize=(max(10, len(targets)*0.8), 
                                           max(6, len(models)*0.6)))
            
            # Aggregate across fingerprints and seeds
            pivot_model = metrics_df.groupby(['model', 'target'])[error_metric].mean().unstack()
            
            # Convert to percentage for display if needed
            if error_metric in ['MAPE', 'NRMSE']:
                pivot_model = pivot_model * 100
            
            ylabel = f'{error_metric}' + (' (%)' if error_metric in ['MAPE', 'NRMSE'] else '')
            sns.heatmap(pivot_model, annot=True, fmt='.2f' if error_metric in ['MAPE', 'NRMSE'] else '.3f', 
                       cmap='YlOrRd', ax=ax, cbar_kws={'label': ylabel})
            ax.set_title(f'{error_metric} Heatmap: Model × Target', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Target Property', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)
            
            plt.tight_layout()
            save_path = save_dir / f"{error_metric}_heatmap_model_target.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
    
    def generate_summary_report(self, agg_metrics: pd.DataFrame, save_path: Path = None):
        """
        Generate text summary report of results.
        
        Args:
            agg_metrics: Aggregated metrics DataFrame
            save_path: Path to save report
        """
        lines = []
        lines.append("="*80)
        lines.append("OpenADMET Benchmark - Results Summary")
        lines.append("="*80)
        lines.append("")
        
        # Overall statistics
        lines.append("Overall Statistics:")
        lines.append(f"  Total combinations tested: {len(agg_metrics)}")
        lines.append(f"  Features: {', '.join(agg_metrics['feature'].unique())}")
        lines.append(f"  Models: {', '.join(agg_metrics['model'].unique())}")
        lines.append(f"  Targets: {', '.join(agg_metrics['target'].unique())}")
        lines.append("")
        
        # Best performers for each target
        lines.append("Best Performing Methods by Target:")
        lines.append("-"*80)
        
        for target in agg_metrics['target'].unique():
            lines.append(f"\n{target}:")
            target_data = agg_metrics[agg_metrics['target'] == target]
            
            # Sort by R2
            best = target_data.nlargest(3, 'R2_mean')
            
            for idx, (i, row) in enumerate(best.iterrows(), 1):
                lines.append(f"  {idx}. {row['feature']:12} + {row['model']:12} | "
                           f"R² = {row['R2_mean']:.3f} ± {row['R2_std']:.3f} | "
                           f"RMSE = {row['RMSE_mean']:.3f} ± {row['RMSE_std']:.3f}")
        
        lines.append("")
        lines.append("-"*80)
        
        # Best features overall
        lines.append("\nBest Features (Average R² across all targets):")
        feature_avg = agg_metrics.groupby('feature')['R2_mean'].mean().sort_values(ascending=False)
        for feature, r2 in feature_avg.items():
            lines.append(f"  {feature:12} : {r2:.3f}")
        
        # Best models overall
        lines.append("\nBest Models (Average R² across all targets):")
        model_avg = agg_metrics.groupby('model')['R2_mean'].mean().sort_values(ascending=False)
        for model, r2 in model_avg.items():
            lines.append(f"  {model:12} : {r2:.3f}")
        
        lines.append("")
        lines.append("="*80)
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Saved: {save_path}")
        
        return report
    
    def save_detailed_metrics(self, agg_metrics: pd.DataFrame, save_path: Path = None):
        """
        Save detailed metrics table to CSV.
        
        Args:
            agg_metrics: Aggregated metrics DataFrame
            save_path: Path to save CSV
        """
        if save_path is None:
            save_path = self.reports_dir / "detailed_metrics.csv"
        
        # Sort by target and R2
        sorted_metrics = agg_metrics.sort_values(['target', 'R2_mean'], ascending=[True, False])
        
        # Save
        sorted_metrics.to_csv(save_path, index=False, float_format='%.4f')
        print(f"Saved: {save_path}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*80)
        print("Running Full Analysis")
        print("="*80 + "\n")
        
        # Load data
        print("Loading predictions...")
        df = self.load_all_predictions()
        print(f"  Loaded {len(df)} predictions")
        
        # Calculate metrics
        print("\nCalculating metrics...")
        metrics_df = self.calculate_metrics(df)
        agg_metrics = self.aggregate_metrics(metrics_df)
        print(f"  Calculated metrics for {len(agg_metrics)} method-target combinations")
        
        # Save detailed metrics
        print("\nSaving detailed metrics...")
        self.save_detailed_metrics(agg_metrics)
        
        # Generate summary report
        print("\nGenerating summary report...")
        report = self.generate_summary_report(
            agg_metrics,
            save_path=self.reports_dir / "summary_report.txt"
        )
        print("\n" + report)
        
        # Create visualizations
        print("\nCreating visualizations...")
        
        # Performance comparison heatmaps
        for metric in ['R2', 'RMSE', 'MAE']:
            self.create_performance_comparison_plot(
                agg_metrics,
                metric=metric,
                save_path=self.plots_dir / f"heatmap_{metric}.png"
            )
        
        # Feature and model comparisons
        self.create_feature_comparison_plot(
            metrics_df,
            save_path=self.plots_dir / "feature_comparison.png"
        )
        
        self.create_model_comparison_plot(
            metrics_df,
            save_path=self.plots_dir / "model_comparison.png"
        )
        
        # Scatter plots
        print("\nCreating prediction scatter plots...")
        self.create_prediction_scatter_plots(df)
        
        # Error by target plots
        print("\nCreating error analysis plots by target...")
        self.create_error_by_target_plots(metrics_df)
        
        print("\n" + "="*80)
        print("Analysis complete!")
        print(f"  Plots saved to: {self.plots_dir}")
        print(f"  Reports saved to: {self.reports_dir}")
        print("="*80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Analyze OpenADMET experiment results"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='openadmet_results',
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='OpenADMET_Benchmark',
        help='Name of the experiment'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ResultsAnalyzer(args.results_dir, args.experiment_name)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
