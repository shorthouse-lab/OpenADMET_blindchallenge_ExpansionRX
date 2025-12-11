# study_manager.py
import datasets, env, models, util.util as util
from database_manager import DatabaseManager
from sklearn.model_selection import KFold, train_test_split
import optuna, os, sqlite3, time
import numpy as np
from typing import Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from experiments import Method

from splitting_utils import ButinaKFold, butina_train_test_split

class StudyManager:
    def __init__(self, method: Method, studies_path: str = './studies/', predictions_path: str = 'studies/predictions.db'):
        self.method = method
        self.studies_path = studies_path
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        self.db = DatabaseManager(predictions_path)
        self.optuna_init = False

    def setup_optuna_storage(self):
        storage_path = f"{self.studies_path}/{str(self.method)}.db"
        os.makedirs(self.studies_path, exist_ok=True)

        temp_storage = optuna.storages.RDBStorage(f"sqlite:///{storage_path}")
        optuna.create_study(storage=temp_storage, study_name="__init__", direction="minimize")

        # Add connection pooling parameters
        conn = sqlite3.connect(storage_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.close()

        self.storage_url = f"sqlite:///{storage_path}?check_same_thread=false&pool_timeout=30"

    def kfold_cv(self, X, Y, hyperparams: Dict, cv_splitter=None):
        """
        Perform k-fold cross-validation.
        Accepts a pre-initialized cv_splitter object to avoid re-clustering.
        """
        # Fallback to standard KFold if no splitter is provided
        if cv_splitter is None:
            cv_splitter = KFold(env.N_FOLDS, shuffle=True, random_state=42)

        predictions = np.zeros_like(Y, dtype=np.float32)

        # Create model instance
        model_class = models.ModelRegistry.get_model(self.method.model)
        task_type = util.get_task_type(Y)
        model = model_class(task_type=task_type, **hyperparams)

        # Loop uses the passed splitter (which has clusters cached in memory)
        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, Y)):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            X_train, X_val, Y_train, Y_val = model.preprocess(X_train, X_val, Y_train, Y_val)

            model.fit(X_train, Y_train)

            fold_predictions = model.predict(X_val)
            predictions[val_idx] = fold_predictions

            if env.VERBOSE and env.PROGRESS:
                env.logger.info(f"[CV] Method={self.method} fold {fold+1}/{env.N_FOLDS} done")

        return predictions

    def train_and_predict(self, X_train, Y_train, X_test, hyperparams: Dict):
        """Train model and make predictions"""
        # Create model instance
        model_class = models.ModelRegistry.get_model(self.method.model)
        task_type = util.get_task_type(Y_train)
        model = model_class(task_type=task_type, **hyperparams)

        X_train, X_test, Y_train, _ = model.preprocess(X_train, X_test, Y_train, Y_train)

        # Train model
        model.fit(X_train, Y_train)

        return model.predict(X_test)

    def run_hyperparameter_optimization(self, X: np.ndarray, Y: np.ndarray, seed: int, smiles=None) -> Dict:
        """Run hyperparameter optimization with cached splitter"""
        model_class = models.ModelRegistry.get_model(self.method.model)
        task_type = util.get_task_type(Y)

        study_id = str(self.method)
        study = optuna.create_study(study_name=f"{study_id}_{seed}", direction="minimize")

        # --- OPTIMIZATION START: Initialize Splitter ONCE ---
        # This calculates the Tanimoto matrix one time and reuses it 15-50 times.
        if smiles is not None:
            cv_splitter = ButinaKFold(
                n_splits=env.N_FOLDS,
                smiles=smiles,
                random_state=seed,
                shuffle=True
            )
        else:
            # Fallback for datasets without SMILES
            cv_splitter = KFold(env.N_FOLDS, shuffle=True, random_state=seed)
        # --- OPTIMIZATION END ---

        splitter_name = type(cv_splitter).__name__
        if env.VERBOSE:
            cluster_msg = ""
            if hasattr(cv_splitter, "groups"):
                cluster_msg = f", groups={len(np.unique(cv_splitter.groups))}"
            env.logger.info(f"[HP] Seed={seed} using {splitter_name} for inner CV{cluster_msg}")

        def objective(trial):
            # ... (progress printing code remains the same) ...

            try:
                hyperparams = model_class.get_hyperparameter_space(trial)
                # Pass the PRE-CALCULATED splitter to kfold_cv
                cv_predictions = self.kfold_cv(X, Y, hyperparams, cv_splitter=cv_splitter)

                score = util.evaluate(Y, cv_predictions, task_type)
                if not np.isfinite(score):
                    return 1e12
                return float(score)
            except Exception as exc:
                print(f"[HP] Trial failed: {exc}")
                return 1e12

        opt_start = time.time()
        study.optimize(objective, n_trials=env.N_TRIALS)
        opt_elapsed = time.time() - opt_start

        if env.VERBOSE:
            env.logger.info(
                f"[HP] Seed={seed} {splitter_name} optimization finished in {opt_elapsed:.2f}s "
                f"(best_value={study.best_value:.4f})"
            )

        return study.best_params

    def run_single_experiment(self, seed: int, data) -> Tuple[int, np.ndarray, np.ndarray]:
        """Run a single experiment with structural splitting"""

        # Check if SMILES exist (OpenADMETDataset has them)
        if hasattr(data, 'smiles') and data.smiles is not None:
            # Use Butina Splitter for Outer Train/Test Split
            split_start = time.time()
            X_train, X_test, Y_train, Y_test, idx_train, idx_test = butina_train_test_split(
                data.X, data.Y, data.smiles,
                test_size=env.TEST_SIZE,
                random_state=seed
            )
            if env.VERBOSE:
                env.logger.info(
                    f"[Split] Seed={seed} outer Butina split finished in {time.time() - split_start:.2f}s "
                    f"(train={len(idx_train)}, test={len(idx_test)})"
                )
            # Subset SMILES to match the training set (needed for inner CV)
            smiles_train = data.smiles[idx_train]
        else:
            # Fallback to Random Split
            split_start = time.time()
            indices = np.arange(len(data.Y))
            X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
                data.X, data.Y, indices,
                test_size=env.TEST_SIZE, random_state=seed
            )
            if env.VERBOSE:
                env.logger.info(
                    f"[Split] Seed={seed} random outer split finished in {time.time() - split_start:.2f}s "
                    f"(train={len(idx_train)}, test={len(idx_test)})"
                )
            smiles_train = None

        # Pass the training-set SMILES to the optimization routine
        hpo_start = time.time()
        best_hyperparams = self.run_hyperparameter_optimization(
            X_train, Y_train, seed, smiles=smiles_train
        )
        if env.VERBOSE:
            env.logger.info(f"[HP] Seed={seed} hyperparameter search duration: {time.time() - hpo_start:.2f}s")

        test_predictions = self.train_and_predict(
            X_train, Y_train, X_test, best_hyperparams
        )

        return seed, test_predictions, idx_test

    def run_nested_cv(self):
        """Run nested cross-validation experiment"""
        data = datasets.Dataset(self.method)
        self.db.store_dataset_targets(self.method.dataset, data.Y)

        #self.setup_optuna_storage()

        predictions = [None for _ in range(env.N_TESTS)]
        indices = [None for _ in range(env.N_TESTS)]

        # Prevent thread oversubscription inside child processes (MKL/BLAS/Numexpr etc.)
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
        os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

        allocated_cores = int(os.environ.get('NSLOTS', multiprocessing.cpu_count()))
        # If sklearn models single-threaded (N_JOBS_SKLEARN=1), we can parallelize more seeds.
        per_seed_parallelism = env.N_JOBS_SKLEARN
        theoretical = allocated_cores // max(1, per_seed_parallelism)
        max_workers = max(1, min(theoretical, env.N_TESTS))
        # Keep GPU/MPS single-worker to avoid memory contention.
        if env.DEVICE != 'cpu':
            max_workers = 1
        if env.PROGRESS:
            print(f"Parallel seed workers: {max_workers} (allocated cores={allocated_cores}, per-seed threads={per_seed_parallelism})")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_seed = {
                executor.submit(self.run_single_experiment, seed, data): seed
                for seed in range(env.N_TESTS)
            }

            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    seed_result, test_predictions, test_indices = future.result()
                    predictions[seed_result] = test_predictions
                    indices[seed_result] = test_indices
                    if env.VERBOSE:
                        env.logger.info(f"[Seed] Completed seed {seed_result}")
                except Exception as exc:
                    print(f"Seed {seed} failed: {exc}")
                    raise exc

        for seed in range(env.N_TESTS):
            if predictions[seed] is not None:
                self.db.store_predictions(
                    self.method.dataset, self.method.feature, self.method.model,
                    predictions[seed], indices[seed], seed, 'random'
                )
        if env.VERBOSE:
            env.logger.info(f"[Method Done] {self.method} stored predictions for {sum(p is not None for p in predictions)} seeds")
