# ==============================================================================
# MODULE: optimization/hyperparameter_optimizer.py
# PURPOSE: Handles automated hyperparameter optimization using Optuna.
# VERSION: 1.0
# AUTHOR: Gemini Code Assist
# ==============================================================================
import copy
from typing import Dict, Any, List, Optional

import mlflow
import optuna
# --- DEFINITIVE FIX for MLflow Integration: Import the MLflow callback ---
# This is the standard, idiomatic way to connect Optuna and MLflow.
from optuna.integration import MLflowCallback
import numpy as np
from sklearn.model_selection import train_test_split

from configuration.config import Config
from source.experiments.ppi_1 import PPIPipeline
from source.utils.data.ground_truth_loader import GroundTruthLoader
from source.utils.post.embedding_processor import EmbeddingProcessor
from source.utils.post.embedding_loader import EmbeddingLoader
from source.utils.data.data_utils import DataUtils


class HyperparameterOptimizer:
    """
    Orchestrates hyperparameter optimization studies using Optuna.
    """

    def __init__(self, config: Config):
        self.base_config = config
        print("HyperparameterOptimizer initialized.")

    @staticmethod
    def _get_trial_param(trial: optuna.Trial, param_name: str, search_space: Dict[str, Any]):
        """Suggests a parameter value based on the configured search space."""
        space = search_space[param_name]
        param_type = space['type']

        if param_type == 'categorical':
            return trial.suggest_categorical(param_name, space['choices'])
        elif param_type == 'int':
            return trial.suggest_int(param_name, space['low'], space['high'], step=space.get('step', 1))
        elif param_type == 'uniform':
            return trial.suggest_float(param_name, space['low'], space['high'])
        elif param_type == 'loguniform':
            return trial.suggest_float(param_name, space['low'], space['high'], log=True)
        else:
            raise ValueError(f"Unsupported Optuna suggestion type: {param_type}")

    def _prepare_hpo_data(self, embedding_path: str) -> Optional[tuple]:
        """
        A helper method to load, filter, split, and create feature matrices for the HPO study.
        This centralizes the data preparation logic.
        """
        print("  Loading and splitting data for HPO study...")
        with EmbeddingLoader(embedding_path, config=self.base_config) as loader:
            available_ids = loader.get_keys()
            if not available_ids:
                print("  ERROR: No embeddings found. Cannot run HPO.")
                return None

            pos_pairs = GroundTruthLoader.load_interaction_pairs_filtered(
                self.base_config.POS_INTERACTIONS_PATH, 1, available_ids, random_state=self.base_config.RANDOM_STATE
            )
            neg_pairs = GroundTruthLoader.load_interaction_pairs_filtered(
                self.base_config.NEG_INTERACTIONS_PATH, 0, available_ids, sample_n=len(pos_pairs), random_state=self.base_config.RANDOM_STATE
            )
            all_pairs = pos_pairs + neg_pairs
            labels_array = np.array([p[2] for p in all_pairs])

            # Create a single, fixed train/validation split for the entire HPO study
            train_pairs, val_pairs = train_test_split(
                all_pairs, test_size=0.2, random_state=self.base_config.RANDOM_STATE, stratify=labels_array
            )
            print(f"  Data split: {len(train_pairs)} training pairs, {len(val_pairs)} validation pairs.")

            print("  Pre-computing feature matrices for HPO study...")
            X_train, y_train = EmbeddingProcessor.create_edge_features(
                train_pairs, loader, self.base_config.EVAL_EDGE_EMBEDDING_METHOD
            )
            X_val, y_val = EmbeddingProcessor.create_edge_features(
                val_pairs, loader, self.base_config.EVAL_EDGE_EMBEDDING_METHOD
            )
            print(f"  Feature matrices created. X_train shape: {X_train.shape}")
            return X_train, y_train, X_val, y_val

    def optimize_ppi_mlp(self, embedding_path: str, embedding_name: str) -> Optional[Dict[str, Any]]:
        """
        Runs an Optuna study to find the best hyperparameters for the PPI MLP model
        for a given set of embeddings.

        Returns:
            A dictionary of the best hyperparameters found, or None on failure.
        """
        DataUtils.print_header(f"Starting HPO for PPI MLP on '{embedding_name}'")

        # 1. Prepare data once to be shared across all trials
        data_tuple = self._prepare_hpo_data(embedding_path)
        if data_tuple is None:
            return None
        X_train, y_train, X_val, y_val = data_tuple

        # 2. Define the objective function for Optuna
        def objective(trial: optuna.Trial) -> float:
            trial_config = copy.deepcopy(self.base_config)
            search_space = self.base_config.HPO_PPI_MLP_SEARCH_SPACE

            trial_config.EVAL_MLP_LEARNING_RATE = self._get_trial_param(trial, 'MLP_LEARNING_RATE', search_space)
            trial_config.EVAL_MLP_DENSE1_UNITS = self._get_trial_param(trial, 'DENSE1_UNITS', search_space)
            trial_config.EVAL_MLP_DROPOUT1_RATE = self._get_trial_param(trial, 'DROPOUT1_RATE', search_space)
            trial_config.EVAL_MLP_DENSE2_UNITS = self._get_trial_param(trial, 'DENSE2_UNITS', search_space)
            trial_config.EVAL_MLP_DROPOUT2_RATE = self._get_trial_param(trial, 'DROPOUT2_RATE', search_space)
            trial_config.EVAL_MLP_L2_REG = self._get_trial_param(trial, 'L2_REG', search_space)
            trial_config.EVAL_BATCH_SIZE = self._get_trial_param(trial, 'BATCH_SIZE', search_space)

            # --- DEFINITIVE FIX for HPO Inefficiency: Disable slow analysis ---
            # SHAP analysis is computationally expensive and not needed for every HPO trial.
            # Disabling it here significantly speeds up the optimization process.
            trial_config.EVAL_GENERATE_SHAP_SUMMARY = False

            ppi_pipeline = PPIPipeline(trial_config)

            try:
                # --- DEFINITIVE FIX: Unpack the new return signature ---
                # The _train_and_evaluate_fold method now returns the model, which we ignore during HPO.
                metrics, _, _ = ppi_pipeline._train_and_evaluate_fold(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    embedding_name=embedding_name, fold_num=trial.number
                )
                return float(metrics.get('auc_sklearn', 0.0))
            except Exception as e:
                print(f"  Trial {trial.number} failed with error: {e}")
                return 0.0

        # 3. Set up MLflow tracking for the study
        mlflow.set_experiment(self.base_config.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"HPO_{embedding_name}") as parent_run:
            mlflow.set_tag("optuna.study_name", f"hpo_{embedding_name}")

            # --- DEFINITIVE FIX for MLflow Integration: Use the MLflow callback ---
            # This will automatically log each trial as a nested run.
            mlflow_callback = MLflowCallback(
                tracking_uri=mlflow.get_tracking_uri(),
                metric_name="validation_auc",
                create_experiment=False,
                mlflow_kwargs={"run_name": f"hpo_trial_{embedding_name}"} # Base name for trials
            )

            # 4. Run the study
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.base_config.RANDOM_STATE))
            study.optimize(objective, n_trials=self.base_config.HPO_N_TRIALS, callbacks=[mlflow_callback], show_progress_bar=True)

            # 5. Report and log the final best results to the parent run
            DataUtils.print_header("Hyperparameter Optimization Finished")
            print(f"  Number of finished trials: {len(study.trials)}")
            best_trial = study.best_trial
            print(f"  Best trial value (AUC): {best_trial.value:.4f}")
            print("  Best hyperparameters found:")
            for key, value in best_trial.params.items():
                print(f"    - {key}: {value}")
                mlflow.log_param(f"best_{key}", value)
            mlflow.log_metric("best_auc", best_trial.value)
            return best_trial.params