# OpenADMET Blind Challenge with ChemGridML

This guide explains how to use the adapted ChemGridML pipeline for the OpenADMET blind challenge.

## Overview

The ChemGridML framework has been adapted to work with the ExpansionRX dataset, allowing you to:

- Test multiple molecular descriptors (ECFP, MACCS, RDKit, Mol2Vec, Graph-based)
- Evaluate various ML models (RF, XGBoost, FNN, SVM, ElasticNet, KNN, GCN, GAT)
- Predict multiple target properties (LogD, KSOL, HLM CLint, MLM CLint, Caco-2, etc.)
- Automatically perform nested cross-validation with hyperparameter optimization
- Generate comprehensive analysis reports and visualizations

## Quick Start

### 1. Setup Environment

First, ensure you have the ChemGridML conda environment set up:

```bash
cd ChemGridML
conda env create -f environment.yml
conda activate chemgridml
```

If you need to install additional dependencies:

```bash
pip install pyyaml seaborn
```

### 2. Configure Your Experiment

Edit `openadmet_config.yaml` to select which models, features, and targets to test:

```yaml
# Select target properties to predict
targets:
  - "LogD"
  - "KSOL"
  - "HLM CLint"

# Select molecular descriptors
features:
  - "ECFP"
  - "MACCS"
  - "RDKitFP"

# Select ML models
models:
  - "RF"
  - "XGBoost"
  - "FNN"
```

**Available Options:**

**Features:**
- `ECFP` - Extended Connectivity Fingerprints (Morgan)
- `AtomPair` - Atom pair fingerprints
- `MACCS` - MACCS keys (166 bits)
- `RDKitFP` - RDKit fingerprints
- `TOPOTOR` - Topological torsion fingerprints
- `MOL2VEC` - Mol2Vec embeddings
- `GRAPH` - Graph representations (for GNN models)

**Models:**
- `RF` - Random Forest
- `XGBoost` - Gradient Boosting
- `FNN` - Feedforward Neural Network
- `SVM` - Support Vector Machine
- `ElasticNet` - Elastic Net Regression
- `KNN` - K-Nearest Neighbors
- `GCN` - Graph Convolutional Network (requires GRAPH features)
- `GAT` - Graph Attention Network (requires GRAPH features)

**Targets:** (from ExpansionRX dataset)
- `LogD` - Distribution coefficient
- `KSOL` - Kinetic solubility
- `HLM CLint` - Human liver microsomal clearance
- `MLM CLint` - Mouse liver microsomal clearance
- `Caco-2 Permeability Papp A>B` - Permeability
- `Caco-2 Permeability Efflux` - Efflux ratio
- `MPPB` - Mouse plasma protein binding
- `MBPB` - Mouse brain protein binding
- `MGMB` - Mouse gut-blood barrier permeability

### 3. Run Experiments

Run all configured experiments:

```bash
python run_openadmet.py --config openadmet_config.yaml
```

Or run a single method:

```bash
python run_openadmet.py --config openadmet_config.yaml --method ECFP_RF_OpenADMET_LogD
```

The script will:
1. Load the ExpansionRX_train.csv data
2. Generate molecular features
3. For each combination of feature-model-target:
   - Perform 10 random train-test splits (80/20)
   - Run nested cross-validation with hyperparameter optimization
   - Store predictions in a SQLite database

**Progress Tracking:**
- Real-time progress updates for each method
- Intermediate results saved after each method completes
- Summary CSV generated in `openadmet_results/run_summary.csv`

### 4. Analyze Results

After experiments complete, run the analysis script:

```bash
python analyze_results.py --results-dir openadmet_results
```

This generates:

**Reports:**
- `reports/summary_report.txt` - Text summary of best methods
- `reports/detailed_metrics.csv` - Full metrics table

**Plots:**
- `plots/heatmap_R2.png` - Performance heatmap (R²)
- `plots/heatmap_RMSE.png` - Performance heatmap (RMSE)
- `plots/heatmap_MAE.png` - Performance heatmap (MAE)
- `plots/feature_comparison.png` - Feature comparison boxplots
- `plots/model_comparison.png` - Model comparison boxplots
- `plots/scatter/*.png` - Predicted vs actual scatter plots

## Workflow Details

### Data Processing

The pipeline automatically:
1. Loads SMILES from CSV
2. Converts to RDKit molecules
3. Generates selected molecular features
4. Filters out invalid molecules and missing target values
5. Reports data statistics

### Model Training

For each method:
1. **Outer loop** (10 iterations): Random 80/20 train-test split
2. **Hyperparameter optimization**: Optuna with 15 trials
   - **Inner loop** (5-fold CV): Evaluate hyperparameters
   - Select best hyperparameters
3. **Final training**: Train on full training set with best parameters
4. **Testing**: Evaluate on held-out test set

### Metrics Calculated

- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **R²** - Coefficient of determination
- **Pearson R** - Pearson correlation coefficient
- **Spearman R** - Spearman rank correlation

Results include both mean and standard deviation across the 10 test splits.

## File Structure

```
ChemGridML/
├── openadmet_config.yaml           # Configuration file
├── run_openadmet.py                # Main experiment runner
├── analyze_results.py              # Analysis and visualization
├── ExpansionRX_train.csv           # Training data
├── code/
│   ├── openadmet_dataset.py        # Custom dataset loader
│   ├── experiments.py              # Experiment definitions
│   ├── models.py                   # ML model implementations
│   ├── features.py                 # Molecular featurization
│   ├── study_manager.py            # Experiment orchestration
│   └── database_manager.py         # Results storage
└── openadmet_results/
    ├── predictions_OpenADMET_Benchmark.db  # SQLite database
    ├── run_summary.csv                      # Execution summary
    ├── studies/                             # Optuna studies
    ├── plots/                               # Visualizations
    └── reports/                             # Text reports
```

## Tips and Best Practices

### Starting Small

For initial testing, use a minimal configuration:

```yaml
targets:
  - "LogD"  # Single target

features:
  - "ECFP"  # Single feature

models:
  - "RF"    # Fast model

experiment:
  n_tests: 3   # Fewer splits
  n_trials: 5  # Fewer HP trials
```

### Scaling Up

For comprehensive benchmarking:

```yaml
targets:
  # List all available targets or use null for auto-detection

features:
  - "ECFP"
  - "MACCS"
  - "RDKitFP"
  - "MOL2VEC"

models:
  - "RF"
  - "XGBoost"
  - "FNN"

experiment:
  n_tests: 10
  n_trials: 15
```

### GPU Acceleration

For neural networks (FNN, GCN, GAT):
- CUDA GPU will be automatically detected and used
- Set `max_workers = 1` in the code to avoid GPU memory issues
- On CPU, neural networks will be significantly slower

### Memory Considerations

- Graph features (GRAPH) require more memory
- Graph neural networks (GCN, GAT) are memory-intensive
- If running out of memory, reduce batch size or run fewer methods in parallel

## Troubleshooting

### Missing Targets

If a target has too much missing data, you'll see:
```
No valid (non-missing) data points found for target 'TargetName'
```

Solution: Remove that target from the config or check the CSV file.

### Feature Generation Failures

Some molecules may fail feature generation. The pipeline will:
- Report: "Feature failed for X molecules"
- Continue with valid molecules
- Store valid indices for proper tracking

### Import Errors

If you get import errors when running scripts:

```bash
# Ensure you're in the ChemGridML directory
cd ChemGridML

# Ensure conda environment is activated
conda activate chemgridml

# Run from the correct location
python run_openadmet.py
```

## Interpreting Results

### Summary Report

The summary report shows:
1. **Best methods per target**: Top 3 feature-model combinations for each property
2. **Best features overall**: Average performance across all targets
3. **Best models overall**: Average performance across all targets

### Heatmaps

- **Green** = Better performance
- **Red** = Worse performance
- Look for consistent patterns across targets

### Scatter Plots

- Points near diagonal = good predictions
- Spread from diagonal = prediction error
- Multiple seeds shown with transparency

## Advanced Usage

### Custom Hyperparameter Spaces

Edit `code/models.py` to modify hyperparameter search spaces:

```python
@staticmethod
def get_hyperparameter_space(trial):
    return {
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300]),
        'max_depth': trial.suggest_categorical('max_depth', [5, 10, 15]),
        # Add more parameters...
    }
```

### Adding New Models

1. Create model class in `code/models.py`
2. Register with `@ModelRegistry.register('YourModel')`
3. Implement required methods: `fit()`, `predict()`, `get_hyperparameter_space()`
4. Add to config file

### Adding New Features

1. Add featurization function to `code/features.py`
2. Follow the pattern of existing functions
3. Return `(features_array, valid_indices)`
4. Add to config file

## Next Steps

After benchmarking:

1. **Review results** in `reports/summary_report.txt`
2. **Identify best methods** for each target
3. **Generate test predictions** using best models
4. **Submit to OpenADMET** challenge

## Support

For issues related to:
- **ChemGridML framework**: Check the main ChemGridML README and GitHub issues
- **OpenADMET data**: Refer to OpenADMET challenge documentation
- **Dependencies**: Check conda environment and package versions

---

**Happy benchmarking!** 🚀
