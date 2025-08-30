import os
from pymatgen.core.composition import Composition
import local_pkgs.proj_pkg.data_handler as dh
import pandas as pd
import numpy as np
# from datetime import datetime
# import proj_pkg.utils as utils
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from local_pkgs.proj_pkg.ml_pipeline import ModelTrainer
from local_pkgs.proj_pkg import preprocessing as preprocess
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
import yaml

with open("thermoelectric_properties.yaml", "r", encoding="utf-8") as file:
    properties_dict = yaml.safe_load(file)

scaler_name = "NoScaler" 
scaler = None

for random_seed in [1, 42, 88, 123, 1201]:
    for target_name, v in properties_dict.items():
        target_col = v['column_name']
        n_splits = 5
        type_of_features = "Direct Composition"
        descriptions = f"Seed{random_seed}_OuterFold{n_splits}_{scaler_name}_{type_of_features}_{target_name}" 

        te_file = "data/sysTEm_dataset.xlsx"
        temp_col = 'Temperature (K)'
        composition_col = "Pymatgen Composition"


        te_df = pd.read_excel(te_file)
        te_df[composition_col] = te_df[composition_col].map(lambda x: Composition(x))

        feature_df = pd.read_csv('data/matminer_features.csv')

        # Print the initial shape of the DataFrame
        print(f"Initial shape: {feature_df.shape}")

        # Drop columns with missing values
        feature_df = feature_df.dropna(axis=1)

        # Print the final shape of the DataFrame
        print(f"Final shape: {feature_df.shape}")

        feature_cols = feature_df.columns.tolist()

        # # check for missing values
        # print(preprocess.columns_with_na(feature_df))

        df = pd.concat([te_df['#'], te_df[composition_col], te_df[temp_col], te_df[target_col], feature_df], axis=1)

        # # when we want to test on a smaller dataset
        # truncated_size = 50
        # df = df.head(truncated_size)

        # generate a column with reduced compositions
        reduced_col = "reduced_compositions"
        preprocess.reduce_comps_in_df(df, composition_col, inplace=True)

        # Drop rows where target is NaN before doing K-Fold cross-validation
        print(f"Shape before dropping NaNs: {df.shape}")
        df_clean = df.dropna(subset=[target_col])
        print(f"Shape after dropping NaNs: {df_clean.shape}")


        df_clean_copy = df_clean.copy()
        compositions_and_temp_count = preprocess.count_identical_rows(df, reduced_col, temp_col)

        avg_ls = feature_cols + [target_col]
        
        df_clean = preprocess.average_duplicates(df_clean_copy, compositions_and_temp_count, avg_ls)
        avg_replacements = ['AVG' + str(i + 1) for i in range(df_clean['#'].isna().sum())]
        df_clean.loc[df_clean['#'].isna(), '#'] = avg_replacements


        y = df_clean[target_col]
        X = df_clean.drop([target_col, composition_col, reduced_col, '#'], axis=1)
        reduced_comp_df = df_clean[reduced_col]
        extra_df = df_clean[['#', reduced_col]]

        # Random Forest
        print(f"Random Forest for {target_name}")

        # For Random Search
        search_space = {
            'n_estimators': [200, 500, 1000, 1500],
            'max_depth': [10, 30, 50, None],
            'min_samples_split': [2, 5, 10]        
            }

        # Initialize the trainer
        trainer = ModelTrainer(random_seed=random_seed)
        trainer.setup_save_folder(model_name="Random Forest", descriptions=descriptions)

        model = RandomForestRegressor(n_jobs=-1, random_state=random_seed)
        results, actual_vs_predicted, results_summary = trainer.train_model(X, y, reduced_comp_df, model, search_space, optimizer="random", n_iter=20, scale=scaler, descriptions_df=extra_df, n_splits=n_splits, save=False)