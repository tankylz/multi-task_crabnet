from local_pkgs.crabnet_pkg.crabnet.crabnet_ import CrabNet
import pandas as pd
import numpy as np
import local_pkgs.proj_pkg.preprocessing as preprocess
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pymatgen.core import Composition
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
import yaml
import random

random_ls = [1, 42, 88, 123, 1201]

with open("thermoelectric_properties.yaml", "r", encoding="utf-8") as file:
    properties = yaml.safe_load(file)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

force_cpu = False # Force CPU usage

for rnd_seed in random_ls:
    torch.manual_seed(rnd_seed)
    np.random.seed(rnd_seed)
    random.seed(rnd_seed)

    for how_to_extend in ['concat_at_output']: # options are ['concat_at_input', 'tile_at_input', 'concat_at_output']
        for target_name, v in properties.items():
            print(f"Training model for {v['full_name']}")
            target_property = v['column_name']
            temp_col = 'Temperature (K)'

            composition_col = "Pymatgen Composition"
            reduced_col = "reduced_compositions"

            te_file = "data/sysTEm_dataset.xlsx" # latest TE file

            te_df = pd.read_excel(te_file)

            df = pd.concat([te_df['#'], te_df[composition_col], te_df[temp_col], te_df[target_property]], axis=1)

            # obtain reduced compositions
            preprocess.reduce_comps_in_df(df, composition_col, inplace=True) # generates a column with reduced compositions

            # rename the composition and target columns to match the requirements of the model
            # why this is required: see utils.py
            df = df.rename(columns={reduced_col: 'formula', target_property: 'target'})


            # try a small subset
            # truncated_size = 700
            # df = df.head(truncated_size)

            df['formula'] = df['formula'].apply(lambda comp: preprocess.scientific_to_numeric_compositions(comp, decimal_places=8))

            # drop NaN target values
            print(f"Original data size: {len(df)}")
            df = df.dropna(subset=['target'])
            print(f"Data size after removing rows with missing target values: {len(df)}")

            # handle duplicates by averaging
            df_copy = df.copy()
            compositions_and_temp_count = preprocess.count_identical_rows(df, 'formula', temp_col)
            df = preprocess.average_duplicates(df_copy, compositions_and_temp_count, 'target')
            avg_replacements = ['AVG' + str(i + 1) for i in range(df['#'].isna().sum())]


            df_copy = df.copy()
            val_proportion = 0.1
            n_splits = 5
            elem_prop = 'matscibert'
            criterion = 'RobustL2'


            target_scaler = 'Standard'
            transform_str = 'Standard'



            model_name = f'SingleTask_{transform_str}Transform_{target_name}_{criterion}_DopedCrab_{elem_prop}_{how_to_extend}_{n_splits}splits_Seed{rnd_seed}'



            comp_k_fold = preprocess.CompositionKFold(n_splits=n_splits, shuffle=True, random_state=rnd_seed)

            results = []
            actual_vs_predicted_df = pd.DataFrame(columns=['Index', 'Random Seed', 'K_fold', '#', 'Actual', 'Predicted', 'Uncertainty'])


            reduced_series = df_copy['formula']
            df = df_copy.drop(columns=[composition_col])

            for fold_idx, (train_idx, test_idx) in enumerate(comp_k_fold.split(reduced_series)):
                print(f"Fold {fold_idx + 1} / {comp_k_fold.n_splits}")


                # create a new model for each fold (so that it doesn't remember the test data)
                model = CrabNet(model_name=f"{model_name}_{fold_idx+1}of{comp_k_fold.n_splits}", mat_prop=target_property, random_state=rnd_seed, 
                                losscurve=False, learningcurve=False, save=True, verbose=True, force_cpu=force_cpu, extend_features=[temp_col], elem_prop=elem_prop, 
                                how_to_extend=how_to_extend, fudge=0.02, dropout=0.5, weight_decay=1e-4, heads=4, N=3, bias=False, fractional=True, 
                                extra_scaler=1.0, extra_scaler_log=1.0, pos_scaler=1.0, pos_scaler_log=1.0, criterion=criterion, d_model=512, lr=1e-5,
                                extra_enc_log_resolution=20000, extra_enc_resolution=20000, pe_resolution=20000, ple_resolution=20000,
                                target_scaling=target_scaler, num_properties=1)
                

                # split the train into validation as well
                training_data = df.iloc[train_idx]
                training_comps = reduced_series.iloc[train_idx]
                unique_compositions = training_comps.unique()
                train_comps, val_comps = train_test_split(unique_compositions, test_size=val_proportion, random_state=rnd_seed, shuffle=True)

                # check that train-val compositions are unique
                test_comps = reduced_series.iloc[test_idx].unique()
                preprocess.check_composition_intersections(train_comps, val_comps, test_comps)

                # generate the train, val and test dataframes
                train_mask = training_comps.isin(train_comps)
                val_mask = training_comps.isin(val_comps)

                train_df = training_data[train_mask]
                # get location of the temperature column
                temp_col_idx = train_df.columns.get_loc(temp_col)
                val_df = training_data[val_mask]
                test_df = df.iloc[test_idx]

                scale = MinMaxScaler()

                train_df, scaler = preprocess.scale_df(train_df, fit=True, cols_to_scale=[temp_col_idx], scaler=scale)
                val_df, _ = preprocess.scale_df(val_df, scaler=scaler, fit=False, cols_to_scale=[temp_col_idx])
                test_df, _ = preprocess.scale_df(test_df, scaler=scaler, fit=False, cols_to_scale=[temp_col_idx])


                model.fit(train_df, val_df)

                test_pred, test_sigma, test_true = model.predict(test_df, return_uncertainty=True, return_true=True)
                print("MAE: ", mean_absolute_error(test_true, test_pred))
                print("RMSE: ", np.sqrt(mean_squared_error(test_true, test_pred)))
                print("R2: ", r2_score(test_true, test_pred))

                # save y predictions and true values
                fold_actual_vs_predicted = pd.DataFrame({
                    'Index': df.iloc[test_idx].index,
                    'Random Seed': rnd_seed, 
                    '#': df.iloc[test_idx]['#'],
                    'Actual':test_true.tolist(),
                    'Predicted': test_pred.tolist(),
                    'Uncertainty': test_sigma.tolist(),
                    'K_fold': fold_idx + 1
                })

                actual_vs_predicted_df = pd.concat([actual_vs_predicted_df, fold_actual_vs_predicted], axis=0, ignore_index=True)

            actual_vs_predicted_df.sort_values(by='Index', inplace=True)
            actual_vs_predicted_file = os.path.join('results/ST-Crabnet', f"{model_name}.csv")
            actual_vs_predicted_df.to_csv(actual_vs_predicted_file, index=False)

            print("\nResults:")
            y_true_ls = actual_vs_predicted_df['Actual'].tolist()
            y_pred_ls = actual_vs_predicted_df['Predicted'].tolist()
            mae = mean_absolute_error(y_true_ls, y_pred_ls)
            rmse = np.sqrt(mean_squared_error(y_true_ls, y_pred_ls))
            r2 = r2_score(y_true_ls, y_pred_ls)

            print(f"Property: {v['full_name']}")
            print(f"MAE: {mae}")
            print(f"RMSE: {rmse}")
            print(f"R2: {r2}\n")