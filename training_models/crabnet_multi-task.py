import yaml
from local_pkgs.crabnet_pkg.crabnet.crabnet_ import CrabNet
import pandas as pd
import numpy as np
import local_pkgs.proj_pkg.preprocessing as preprocess
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from local_pkgs.proj_pkg.data_handler import add_one_hot_column
import torch
import os
import random

rnd_ls = [1, 42, 88, 123, 1201]


with open("thermoelectric_properties.yaml", "r", encoding="utf-8") as file:
    properties_dict = yaml.safe_load(file)

te_df = pd.read_excel('data/dataset_expanded.xlsx')
composition_col = "Pymatgen Composition"
reduced_col = "reduced_compositions"
temp_col = "Temperature (K)"
target_name_col = "target_name"
target_one_hot_col = "target_one_hot"

df = pd.concat([te_df['#'],te_df[composition_col], te_df[temp_col], te_df[target_name_col], te_df['target']], axis=1)

# try a smaller subset
# np.random.seed(1)
# torch.manual_seed(1)
# random.seed(1)
# truncated_size = 700
# df = df.sample(n=truncated_size, random_state=1)

# obtain reduced compositions
preprocess.reduce_comps_in_df(df, composition_col, inplace=True) # generates a column with reduced compositions
df_copy = df.copy()
df = df_copy.drop(columns=[composition_col])
df = df.rename(columns={reduced_col: 'formula'}) # rename the column as per the CrabNet requirements
df['formula'] = df['formula'].apply(lambda comp: preprocess.scientific_to_numeric_compositions(comp, decimal_places=8)) # convert the scientific compositions to numeric compositions

# deal with duplicates (same temperature, composition and target property)
# Check the columns in the dataframe
print(df.columns)
identical_df = preprocess.count_identical_rows(df, 'formula', temp_col, target_name_col)
# Averaging strategy
df_copy = df.copy()
df = preprocess.average_duplicates(df_copy, identical_df, 'target')
# fill # column for averaged results
avg_replacements = ['AVG' + str(i + 1) for i in range(df['#'].isna().sum())]
df.loc[df['#'].isna(), '#'] = avg_replacements

# use StandardScaler for all properties

# one hot encode the target property
df, encoder = add_one_hot_column(df, target_name_col, target_one_hot_col)
# test that the one hot encoding is working
unique_props = df[target_name_col].unique()
for prop in unique_props:
    df_prop = df[df[target_name_col] == prop]
    one_hot_unique = len(np.unique(df_prop[target_one_hot_col].tolist(),axis=0))
    if one_hot_unique != 1:
        print(f"Error: {prop} has {one_hot_unique} one hot vectors")

# use StandardScaler for all properties

target_scaler = {}

for prop in unique_props:
    target_scaler[prop] = 'Standard'

# setting CV
n_splits = 5
elem_prop = 'matscibert'
one_hot_layer = 'concat_at_attn' # or integers (from 0 to the # of residual block hidden layers + 1)
# 0 means concat at the input of residual block
val_proportion = 0.1
criterion = 'RobustL2'
force_cpu = True # Force CPU usage

results = []
df_copy = df.copy()
reduced_series = df_copy['formula']

# get number of properties
num_properties = df[target_name_col].nunique()

for rnd_seed in rnd_ls:
        for how_to_extend in ['concat_at_input', 'tile_at_input', 'concat_at_output', ]:

            np.random.seed(rnd_seed)
            torch.manual_seed(rnd_seed)
            random.seed(rnd_seed)

            comp_k_fold = preprocess.CompositionKFold(n_splits=n_splits, shuffle=True, random_state=rnd_seed)

            # results df
            actual_vs_predicted_df = pd.DataFrame()

            model_name = f'MultiTask_{criterion}_{elem_prop}_{how_to_extend}_onehotappend{one_hot_layer}_{n_splits}splits_seed{rnd_seed}'

            for fold_idx, (train_idx, test_idx) in enumerate(comp_k_fold.split(reduced_series)):
                print(f"Fold {fold_idx + 1} / {comp_k_fold.n_splits}")

                # create a new model for each fold (so that it doesn't remember the test data)
                model = CrabNet(model_name=f"{model_name}_{fold_idx+1}of{comp_k_fold.n_splits}", random_state=rnd_seed, 
                                losscurve=False, learningcurve=False, save=True, verbose=True, force_cpu=force_cpu, extend_features=[temp_col], elem_prop=elem_prop, 
                                how_to_extend=how_to_extend, fudge=0.02, dropout=0.5, weight_decay=1e-4, heads=4, N=3, bias=False, fractional=True, 
                                extra_scaler=1.0, extra_scaler_log=1.0, pos_scaler=1.0, pos_scaler_log=1.0, criterion=criterion, d_model=512, lr=1e-5, 
                                extra_enc_log_resolution=20000, extra_enc_resolution=20000, pe_resolution=20000, ple_resolution=20000,
                                target_scaling=target_scaler, num_properties=num_properties, property_name_col=target_name_col, property_one_hot_col=target_one_hot_col, one_hot_layer=one_hot_layer)

                # split the train into validation as well
                training_data = df.iloc[train_idx]
                training_comps = reduced_series.iloc[train_idx]
                unique_compositions = training_comps.unique()
                train_comps, val_comps = train_test_split(
                    unique_compositions,
                    test_size=val_proportion,
                    random_state=rnd_seed,
                    shuffle=True
                )

                # check that train-val compositions are unique
                test_comps = reduced_series.iloc[test_idx].unique()
                preprocess.check_composition_intersections(train_comps, val_comps, test_comps)

                # generate the train, val and test dataframes
                train_mask = training_comps.isin(train_comps)
                val_mask = training_comps.isin(val_comps)
                train_df = training_data[train_mask]
                val_df = training_data[val_mask]
                test_df = df.iloc[test_idx]

                # scale the temperature col
                temp_col_idx = train_df.columns.get_loc(temp_col) # get the index of the temperature column
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
                    'Property': df.iloc[test_idx][target_name_col],
                    'Actual':test_true.tolist(),
                    'Predicted': test_pred.tolist(),
                    'Uncertainty': test_sigma.tolist(),
                    'K_fold': fold_idx + 1
                })
                if not fold_actual_vs_predicted.empty:
                    actual_vs_predicted_df = pd.concat([actual_vs_predicted_df, fold_actual_vs_predicted], axis=0, ignore_index=True)
                else:
                    actual_vs_predicted_df = fold_actual_vs_predicted


            actual_vs_predicted_df.sort_values(by='Index', inplace=True)
            actual_vs_predicted_file = os.path.join('results/MT-Crabnet', f"{model_name}.csv")
            actual_vs_predicted_df.to_csv(actual_vs_predicted_file, index=False)

            print("\nResults:")
            for prop in actual_vs_predicted_df['Property'].unique():
                prop_results = actual_vs_predicted_df[actual_vs_predicted_df['Property'] == prop]
                y_true_ls = prop_results['Actual'].tolist()
                y_pred_ls = prop_results['Predicted'].tolist()
                mae = mean_absolute_error(y_true_ls, y_pred_ls)
                rmse = np.sqrt(mean_squared_error(y_true_ls, y_pred_ls))
                r2 = r2_score(y_true_ls, y_pred_ls)

                print(f"Property: {prop}")
                print(f"MAE: {mae}")
                print(f"RMSE: {rmse}")
                print(f"R2: {r2}\n")
