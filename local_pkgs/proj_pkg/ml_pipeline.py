import os
import json
import joblib
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV
from local_pkgs.proj_pkg import preprocessing as preprocess
import numpy as np


class ModelTrainer:
    def __init__(self, base_folder="results", random_seed=42):
        self.base_folder = base_folder
        self.random_seed = random_seed

    def setup_save_folder(self, model_name, descriptions=None):
        date_str = datetime.now().strftime('%Y-%m-%d')
        if descriptions is None:
            self.save_folder = os.path.join(self.base_folder, model_name, f"{date_str}")
        else:
            self.save_folder = os.path.join(self.base_folder, model_name, f"{date_str}", f"{descriptions}")
        os.makedirs(self.save_folder, exist_ok=True)
    
    def train_model(self, X, y, reduced_col_data, model, search_space, descriptions_df=None, optimizer="random", scale=None, n_splits=5, n_iter=30, n_points=15, save=False):
        """
        Trains a model using nested cross-validation with hyperparameter tuning.

        Parameters
        ----------
        X : pd.DataFrame
            The feature DataFrame.
        y : pd.Series
            The target series.
        reduced_col_data : pd.Series
            The column data used for splitting the folds.
        model : estimator object
            The machine learning model to be trained.
        search_space : dict
            The hyperparameter search space.
        optimizer : str, optional
            The optimization technique to use ('random' or 'bayesian'). Default is 'random'.
        scale : object, optional
            The scaler object to use for scaling the data. Default is None.
        n_splits : int, optional
            The number of outer cross-validation splits. Default is 5.
        n_iter : int, optional
            The number of iterations for hyperparameter search. Default is 30.
        n_points : int, optional
            The number of points for Bayesian optimization. Default is 15.
        save: bool, optional
            Whether to save the model and hyperparameters. Default is False.

        Returns
        -------
        tuple
            A tuple containing the cross-validation results, actual vs predicted DataFrame, and metrics summary.
        """
        print(f"Scaler: {scale}")

        # Prepare cross-validation
        outer_kfold = preprocess.CompositionKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)

        results = []
        actual_vs_predicted_df = pd.DataFrame()
        # Outer loop for cross-validation
        for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_kfold.split(reduced_col_data)):
            print(f"Outer Fold {outer_fold_idx + 1}/{outer_kfold.n_splits}")

            X_training = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            if descriptions_df is not None:
                descriptions_df_test = descriptions_df.iloc[test_idx].reset_index(drop=True)
            else:
                descriptions_df_test = pd.DataFrame()
            y_test = y.iloc[test_idx]

            test_compositions_unique = outer_kfold.fold_compositions_[outer_fold_idx]['test_compositions']

            # Inner cross-validation for hyperparameter tuning
            inner_kfold = preprocess.CompositionKFold(n_splits=3, shuffle=True, random_state=self.random_seed)

            best_model = None
            best_score = float('inf')
            best_scaler = None

            for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_kfold.split(reduced_col_data.iloc[train_idx])):

                X_inner_train = X_training.iloc[inner_train_idx]
                y_inner_train = y_train.iloc[inner_train_idx]
                X_inner_val = X_training.iloc[inner_val_idx]
                y_inner_val = y_train.iloc[inner_val_idx]

                # Check that the compositions in each set are unique
                inner_train_compositions_unique = inner_kfold.fold_compositions_[inner_fold_idx]['train_compositions']
                inner_val_compositions_unique = inner_kfold.fold_compositions_[inner_fold_idx]['test_compositions']
                preprocess.check_composition_intersections(inner_train_compositions_unique, inner_val_compositions_unique, test_compositions_unique)

                # Scale the inner sets
                scaled_inner_train_data, scaler = preprocess.scale_df(X_inner_train, scaler=scale, fit=True)
                scaled_inner_val_data, _ = preprocess.scale_df(X_inner_val, scaler=scaler, fit=False)

                # Choose optimization technique
                if optimizer == "bayesian":
                    opt = BayesSearchCV(model, search_space, n_iter=n_iter, n_points=n_points, cv=3,
                                        random_state=self.random_seed, n_jobs=-1, scoring='neg_mean_squared_error')
                elif optimizer == "random":
                    opt = RandomizedSearchCV(model, search_space, n_iter=n_iter, cv=3, random_state=self.random_seed, n_jobs=-1, scoring='neg_mean_squared_error')

                # Fit on inner train set and optimize hyperparameters
                opt.fit(scaled_inner_train_data, y_inner_train)

                # Best model from this fold
                best_inner_model = opt.best_estimator_

                # Score the inner validation set
                y_inner_val_pred = best_inner_model.predict(scaled_inner_val_data)
                inner_fold_score = root_mean_squared_error(y_inner_val, y_inner_val_pred)

                print(f"  Inner Fold {inner_fold_idx + 1}/{inner_kfold.n_splits}: MSE = {inner_fold_score:.4f}")

                if inner_fold_score < best_score:
                    best_score = inner_fold_score
                    best_model = best_inner_model
                    best_scaler = scaler  # Save the scaler from the best inner fold

            # Scale the outer test data using the scaler from the best inner fold
            scaled_test_data, _ = preprocess.scale_df(X_test, scaler=best_scaler, fit=False)

            # Predict on the outer test set
            y_test_pred = best_model.predict(scaled_test_data)

            # Save the model and hyperparameters
            if save:
                self.save_model_and_hyperparams(best_model, outer_fold_idx + 1)

            # Calculate evaluation metrics
            mae = mean_absolute_error(y_test, y_test_pred)
            rmse = root_mean_squared_error(y_test, y_test_pred)
            r2 = r2_score(y_test, y_test_pred)

            print(f"Outer Fold {outer_fold_idx + 1} - Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}, Test R-squared: {r2:.4f}")

            # Store results
            results.append({
                'outer_fold': outer_fold_idx + 1,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'best_hyperparameters': best_model.get_params()
            })

            # Collect actual and predicted values along with their indices
            fold_actual_vs_predicted = pd.DataFrame({
                'K_Fold': outer_fold_idx + 1,
                'Random Seed': np.full(len(y_test), self.random_seed),
                'Actual': y_test.tolist(),
                'Predicted': y_test_pred.tolist()
            })

            result_df = pd.concat([fold_actual_vs_predicted, descriptions_df_test], axis=1)

            actual_vs_predicted_df = pd.concat([actual_vs_predicted_df, result_df], axis=0)

            print("-" * 40)

        # Sort the DataFrame by the index to preserve the original order
        actual_vs_predicted_file = os.path.join(self.save_folder, "actual_vs_predicted.csv")
        actual_vs_predicted_df.to_csv(actual_vs_predicted_file, index=False)
        print(f"Saved Actual vs Predicted values to {actual_vs_predicted_file}")

        # Calculate the mean and standard deviation of the metrics
        metrics_df = pd.DataFrame(results)
        metrics_summary = {
            'mean_mae': metrics_df['mae'].mean(),
            'std_mae': metrics_df['mae'].std(),
            'mean_rmse': metrics_df['rmse'].mean(),
            'std_rmse': metrics_df['rmse'].std(),
            'mean_r2': metrics_df['r2'].mean(),
            'std_r2': metrics_df['r2'].std()
        }

        # Save results and metrics summary to CSV
        results_file = os.path.join(self.save_folder, "cv_results.csv")
        metrics_df.to_csv(results_file, index=False)
        print(f"Saved Cross-Validation results to {results_file}")
        metrics_summary_file = os.path.join(self.save_folder, "metrics_summary.csv")
        pd.DataFrame([metrics_summary]).to_csv(metrics_summary_file, index=False)
        print(f"Saved Metrics Summary to {metrics_summary_file}")
        return results, actual_vs_predicted_df, metrics_summary

        
    
    # def single_loop_cv(self, df_clean, target_series_clean, model, n_splits=5):
    #     # Prepare cross-validation
    #     kfold = preprocess.CompositionKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        
    #     results = []
    #     actual_vs_predicted_df = pd.DataFrame(columns=['Index', 'Actual', 'Predicted'])

    #     # Loop for cross-validation
    #     for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(df_clean)):
    #         print(f"Fold {fold_idx + 1}/{kfold.n_splits}")
            
    #         X_train = df_clean.iloc[train_idx, 2:]  # Exclude the first two columns (composition objects)
    #         y_train = target_series_clean.iloc[train_idx]
    #         X_test = df_clean.iloc[test_idx, 2:]
    #         y_test = target_series_clean.iloc[test_idx]

    #         # Scale the data
    #         scaled_train_data, scaler = preprocess.scale_df(X_train, scaler=None, fit=True)
    #         scaled_test_data, _ = preprocess.scale_df(X_test, scaler=scaler, fit=False)

    #         # Train the model
    #         model.fit(scaled_train_data, y_train)
            
    #         # Predict on the test set
    #         y_test_pred = model.predict(scaled_test_data)

    #         # Calculate evaluation metrics
    #         mae = mean_absolute_error(y_test, y_test_pred)
    #         rmse = root_mean_squared_error(y_test, y_test_pred)
    #         r2 = r2_score(y_test, y_test_pred)
            
    #         print(f"Fold {fold_idx + 1} - Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}, Test R-squared: {r2:.4f}")

    #         # Store results
    #         results.append({
    #             'fold': fold_idx + 1,
    #             'mae': mae,
    #             'rmse': rmse,
    #             'r2': r2
    #         })

    #         # Collect actual and predicted values along with their indices
    #         fold_actual_vs_predicted = pd.DataFrame({
    #             'Index': df_clean.iloc[test_idx].index,
    #             'Actual': y_test.tolist(),
    #             'Predicted': y_test_pred.tolist()
    #         })
    #         actual_vs_predicted_df = pd.concat([actual_vs_predicted_df, fold_actual_vs_predicted], ignore_index=True)
            
    #         print("-" * 40)

    #     # Sort the DataFrame by the index to preserve the original order
    #     actual_vs_predicted_df.sort_values(by='Index', inplace=True)
    #     actual_vs_predicted_file = os.path.join(self.save_folder, "actual_vs_predicted_single_loop.csv")
    #     actual_vs_predicted_df.to_csv(actual_vs_predicted_file, index=False)
    #     print(f"Saved Actual vs Predicted values to {actual_vs_predicted_file}")

    #     return results, actual_vs_predicted_df
    
    def save_model_and_hyperparams(self, model, fold_idx):
        model_filename = os.path.join(self.save_folder, f"best_model_outer_fold_{fold_idx}.joblib")
        hyperparams_filename = os.path.join(self.save_folder, f"best_model_outer_fold_{fold_idx}_hyperparams.json")

        # Save the model
        joblib.dump(model, model_filename)
        
        # Save the hyperparameters
        hyperparameters = model.get_params()
        with open(hyperparams_filename, 'w') as json_file:
            json.dump(hyperparameters, json_file, indent=4)

        print(f"Saved model and hyperparameters for Outer Fold {fold_idx} as {model_filename}")