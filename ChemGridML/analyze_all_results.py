#!/usr/bin/env python
"""
Combine and analyze multiple OpenADMET experiment result databases.

Reads any number of prediction SQLite databases (each with `predictions` and
`dataset_targets` tables) from the results directory, merges them into a single
DataFrame with an `experiment` column, computes metrics, and produces
comparative plots across models, features, targets, and experiments.

Usage:
  python analyze_all_results.py \
      --results-dir openadmet_results \
      --experiments OpenADMET_Phase1_ElasticNet OpenADMET_Phase2_KNN

If --experiments is omitted, it will automatically include all files matching
`predictions_*.db` in the results directory.
"""

import argparse, sqlite3
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def load_db(db_path: Path, experiment_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = """
        SELECT p.*, dt.target_value
        FROM predictions p
        JOIN dataset_targets dt ON p.dataset_name = dt.dataset_name AND p.data_index = dt.data_index
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['target'] = df['dataset_name'].str.replace('OpenADMET_', '').str.replace('_', ' ')
    df['experiment'] = experiment_name
    return df


def compute_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouped = df.groupby(['experiment', 'fingerprint', 'model_name', 'target', 'seed'])
    for (exp, fp, model, target, seed), g in grouped:
        y_true = g['target_value'].values
        y_pred = g['prediction'].values
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        pearson_r, _ = pearsonr(y_true, y_pred)
        spearman_r, _ = spearmanr(y_true, y_pred)
        eps = 1e-10
        abs_err = np.abs(y_true - y_pred)
        
        # Use Symmetric MAPE (sMAPE) which handles near-zero values better
        # sMAPE = 100 * |y_true - y_pred| / (|y_true| + |y_pred|)
        # This ranges from 0% to 100% and doesn't blow up for small values
        denominator = (np.abs(y_true) + np.abs(y_pred))
        smape = 100.0 * np.mean(2.0 * abs_err / (denominator + eps))
        
        # Also compute traditional MAPE for reference, but only for values where |y_true| is significant
        # Use median absolute value as threshold to avoid division by near-zero
        threshold = max(np.median(np.abs(y_true)) * 0.1, 1.0)
        mask = np.abs(y_true) > threshold
        if mask.sum() > 0:
            rel_err = abs_err[mask] / np.abs(y_true[mask])
            mape = rel_err.mean() * 100.0  # percentage (0-100)
        else:
            mape = np.nan  # No valid samples for traditional MAPE
        
        mean_y = np.mean(np.abs(y_true))
        nrmse = rmse / (mean_y + eps) if mean_y > eps else 0.0
        records.append({
            'experiment': exp,
            'feature': fp,
            'model': model,
            'target': target,
            'seed': seed,
            'n_samples': len(y_true),
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Pearson_R': pearson_r,
            'Spearman_R': spearman_r,
            'MAPE': mape,
            'sMAPE': smape,
            'NRMSE': nrmse
        })
    return pd.DataFrame(records)


def aggregate_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    out = []
    grouped = seed_metrics.groupby(['experiment', 'feature', 'model', 'target'])
    for (exp, feat, model, target), g in grouped:
        row = {
            'experiment': exp,
            'feature': feat,
            'model': model,
            'target': target,
            'n_seeds': g['seed'].nunique(),
            'n_samples_mean': g['n_samples'].mean()
        }
        for metric in ['RMSE', 'MAE', 'R2', 'Pearson_R', 'Spearman_R', 'MAPE', 'sMAPE', 'NRMSE']:
            row[f'{metric}_mean'] = g[metric].mean()
            row[f'{metric}_std'] = g[metric].std()
        out.append(row)
    return pd.DataFrame(out)



def plot_experiment_comparison(agg: pd.DataFrame, results_dir: Path, metric: str = 'R2'):
    # Compare experiments side-by-side for each target (best mean metric by feature+model)
    targets = agg['target'].unique()
    for target in targets:
        sub = agg[agg['target'] == target]
        pivot = sub.pivot_table(values=f'{metric}_mean', index=['feature', 'model'], columns='experiment')
        plt.figure(figsize=(10, max(6, 0.4 * len(pivot))))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn' if metric in ['R2','Pearson_R','Spearman_R'] else 'RdYlGn_r')
        plt.title(f'Experiment Comparison ({metric}) — Target: {target}')
        plt.tight_layout()
        path = results_dir / f'plots/experiment_comparison_{metric}_{target.replace(" ","_")}.png'
        path.parent.mkdir(exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved: {path}')



def plot_metric_trends(agg: pd.DataFrame, results_dir: Path, metric: str = 'R2'):
    targets = agg['target'].unique()
    for target in targets:
        sub = agg[agg['target'] == target]
        plt.figure(figsize=(12,6))
        sns.boxplot(data=sub, x='experiment', y=f'{metric}_mean')
        plt.title(f'{metric} Distribution Across Experiments — Target: {target}')
        plt.tight_layout()
        path = results_dir / f'plots/{metric}_distribution_across_experiments_{target.replace(" ","_")}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved: {path}')


def save_combined_tables(seed_metrics: pd.DataFrame, agg: pd.DataFrame, results_dir: Path):
    combined_dir = results_dir / 'reports'
    combined_dir.mkdir(exist_ok=True)
    seed_metrics.to_csv(combined_dir / 'combined_seed_metrics.csv', index=False)
    agg.sort_values(['experiment','target','R2_mean'], ascending=[True, True, False]) \
        .to_csv(combined_dir / 'combined_aggregated_metrics.csv', index=False)
    print(f"Saved: {combined_dir / 'combined_seed_metrics.csv'}")
    print(f"Saved: {combined_dir / 'combined_aggregated_metrics.csv'}")



def plot_model_fingerprint_bars(agg: pd.DataFrame, results_dir: Path, metric: str = 'R2'):
    """
    For each target, plot a bar chart of model (x-axis), metric (y-axis), hue=fingerprint.
    """
    targets = agg['target'].unique()
    for target in targets:
        sub = agg[agg['target'] == target]
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            data=sub,
            x='model',
            y=f'{metric}_mean',
            hue='feature',
            ci=None
        )
        plt.title(f'{metric} by Model and Featurization — Target: {target}')
        ylabel = f'{metric} (%)' if metric in ['MAPE', 'sMAPE'] else f'{metric} (mean across seeds)'
        plt.ylabel(ylabel)
        plt.xlabel('Model')
        plt.legend(title='Fingerprint', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        path = results_dir / f'plots/model_fingerprint_bar_{metric}_{target.replace(" ","_")}.png'
        path.parent.mkdir(exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved: {path}')

def identify_winners(agg: pd.DataFrame, results_dir: Path):
    """
    For each target, identify the best model+feature combination by lowest sMAPE.
    Save results to CSV.
    """
    winners = []
    targets = agg['target'].unique()
    
    for target in targets:
        sub = agg[agg['target'] == target]
        # Find row with minimum sMAPE_mean (more robust than MAPE for near-zero values)
        best_idx = sub['sMAPE_mean'].idxmin()
        best = sub.loc[best_idx]
        
        winners.append({
            'target': target,
            'winning_model': best['model'],
            'winning_feature': best['feature'],
            'sMAPE_mean': best['sMAPE_mean'],
            'sMAPE_std': best['sMAPE_std'],
            'MAPE_mean': best['MAPE_mean'],
            'R2_mean': best['R2_mean'],
            'RMSE_mean': best['RMSE_mean'],
            'MAE_mean': best['MAE_mean'],
            'n_seeds': best['n_seeds']
        })
    
    winners_df = pd.DataFrame(winners)
    winners_df = winners_df.sort_values('sMAPE_mean')
    
    # Save to CSV
    winners_path = results_dir / 'reports' / 'winners_by_MAPE.csv'
    winners_path.parent.mkdir(exist_ok=True)
    winners_df.to_csv(winners_path, index=False)
    print(f'\nWinners by sMAPE (Symmetric MAPE - lowest is best):')
    print(f"{'Target':<15} {'Model':<12} {'Feature':<12} {'sMAPE%':>8} {'R²':>6} {'RMSE':>8} {'MAE':>8}")
    print('-' * 80)
    for _, row in winners_df.iterrows():
        print(f"{row['target']:<15} {row['winning_model']:<12} {row['winning_feature']:<12} "
              f"{row['sMAPE_mean']:>7.1f}% {row['R2_mean']:>6.3f} {row['RMSE_mean']:>8.2f} {row['MAE_mean']:>8.2f}")
    print(f'\nSaved: {winners_path}')
    
    return winners_df


def plot_best_model_scatter(predictions_df: pd.DataFrame, winners_df: pd.DataFrame, results_dir: Path):
    """
    Create scatter plots of actual vs predicted for the best model (by sMAPE) for each target.
    """
    scatter_dir = results_dir / 'plots' / 'scatter'
    scatter_dir.mkdir(parents=True, exist_ok=True)
    
    for _, winner in winners_df.iterrows():
        target = winner['target']
        model = winner['winning_model']
        feature = winner['winning_feature']
        
        # Filter predictions for this specific model+feature+target combination
        mask = (
            (predictions_df['target'] == target) &
            (predictions_df['model_name'] == model) &
            (predictions_df['fingerprint'] == feature)
        )
        data = predictions_df[mask].copy()
        
        if len(data) == 0:
            print(f'Warning: No data found for {target} - {model} - {feature}')
            continue
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot all predictions (across all seeds)
        ax.scatter(data['target_value'], data['prediction'], 
                  alpha=0.3, s=20, color='steelblue', edgecolors='none')
        
        # Add perfect prediction line
        min_val = min(data['target_value'].min(), data['prediction'].min())
        max_val = max(data['target_value'].max(), data['prediction'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label='Perfect prediction')
        
        # Add metrics to plot
        y_true = data['target_value'].values
        y_pred = data['prediction'].values
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Add text box with metrics
        textstr = f'Model: {model}\nFeature: {feature}\n'
        textstr += f'R² = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}\n'
        textstr += f'sMAPE = {winner["sMAPE_mean"]:.1f}%'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        ax.set_xlabel('Actual Value', fontsize=12)
        ax.set_ylabel('Predicted Value', fontsize=12)
        ax.set_title(f'Best Model for {target}\n({model} + {feature})', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        safe_target = target.replace(' ', '_').replace('>', '').replace('<', '')
        path = scatter_dir / f'best_model_scatter_{safe_target}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved: {path}')


def main():
    parser = argparse.ArgumentParser(description='Combine and analyze multiple OpenADMET experiment results')
    parser.add_argument('--results-dir', type=str, default='openadmet_results', help='Directory with prediction DBs')
    parser.add_argument('--experiments', nargs='*', help='Experiment names (omit for auto-detect)')
    parser.add_argument('--recursive', action='store_true', help='Recursively search for prediction DBs in all subfolders')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f'Results directory not found: {results_dir}')

    db_files = []
    if args.experiments:
        # Use explicit experiment names, only in top-level
        for exp in args.experiments:
            db_path = results_dir / f'predictions_{exp}.db'
            if db_path.exists():
                db_files.append((db_path, exp))
            else:
                print(f'Warning: missing DB for experiment {exp}: {db_path} (skipping)')
    else:
        # Auto-detect all prediction DBs, recursively if requested
        if args.recursive:
            found = list(results_dir.rglob('predictions_*.db'))
        else:
            found = list(results_dir.glob('predictions_*.db'))
        if not found:
            raise RuntimeError('No experiment databases found to combine.')
        for db_path in found:
            # Use relative path (without .db) as experiment label for uniqueness
            rel = db_path.relative_to(results_dir).with_suffix('')
            exp = str(rel)
            db_files.append((db_path, exp))

    print(f'Combining {len(db_files)} experiments/databases:')
    for db_path, exp in db_files:
        print(f'  {exp} -> {db_path}')

    all_predictions = []
    for db_path, exp in db_files:
        try:
            df = load_db(db_path, exp)
            all_predictions.append(df)
            print(f'    Loaded {len(df)} rows from {db_path}')
        except Exception as e:
            print(f'    Failed to load {db_path}: {e}')

    if not all_predictions:
        raise RuntimeError('No prediction data loaded; aborting.')

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    print(f'Total combined rows: {len(predictions_df)}')

    seed_metrics = compute_seed_metrics(predictions_df)
    agg_metrics = aggregate_metrics(seed_metrics)

    # Save tables
    save_combined_tables(seed_metrics, agg_metrics, results_dir)

    # Identify winners by MAPE
    winners_df = identify_winners(agg_metrics, results_dir)

    # Plot scatter plots for best models
    plot_best_model_scatter(predictions_df, winners_df, results_dir)

    # Plots

    for metric in ['R2','RMSE','MAE','sMAPE','MAPE','NRMSE']:
        plot_experiment_comparison(agg_metrics, results_dir, metric=metric)
        plot_metric_trends(agg_metrics, results_dir, metric=metric)
        plot_model_fingerprint_bars(agg_metrics, results_dir, metric=metric)

    print('Combined analysis complete.')


if __name__ == '__main__':
    main()
