# code adapted from original DopNet example repo 
# DOI: 10.1038/s41524-021-00564-y

import numpy
import torch
import itertools
import pandas
import local_pkgs.dopnet_pkg.autoencoder as ae
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from local_pkgs.dopnet_pkg.ml import get_k_folds_list, get_k_folds_by_index_ls
import os
import local_pkgs.dopnet_pkg.dopnet as dp 
from local_pkgs.proj_pkg.preprocessing import CompositionKFold
import yaml
import pandas as pd
from datetime import datetime
import random

max_dops = 5 # maximum number of dopants
n_folds = 5
dopnet_init_lr = 1e-2 # model initial learning rate
batch_size = 256
emb_init_lr = 1e-3
embedding_epochs = 300
dopnet_epochs = 600
ratio = 10 # doping threshold

with open("thermoelectric_properties.yaml", "r", encoding="utf-8") as file:
    properties = yaml.safe_load(file)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('Device: ' + str(device))

date_str = datetime.now().strftime('%Y-%m-%d')


for random_seed in [1, 42, 88, 123, 1201]:
    print(f'random seed: {random_seed}')
    torch.manual_seed(random_seed)
    numpy.random.seed(random_seed)
    random.seed(random_seed)

    for target, v in properties.items():
        print('Target: ' + target)

        dataset_path = f'data/TE_reduced_comp_DopNet_{target}.xlsx'

        df = pd.read_excel(dataset_path)

        target_col = df.columns.get_loc(v['column_name'])
        temp_col = 'Temperature (K)'
        reduced_col = "reduced_compositions"

        cond_cols = [df.columns.get_loc('Temperature (K)')]
        comp_col = df.columns.get_loc('reduced_compositions')

        # dataset loading
        dataset, cleaned_comps = dp.load_dataset(dataset_path, comp_idx=comp_col, target_idx=target_col, max_dops=max_dops, cond_idx=cond_cols, doping_threshold=ratio/100)

        # we want to separate the dataset into k folds, ensuring that each normalised composition is only in one fold
        comp_k_fold_obj = CompositionKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        split_ls = []
        for train_idx, test_idx in comp_k_fold_obj.split(cleaned_comps):
            split_ls.append(test_idx)

        k_folds = get_k_folds_by_index_ls(dataset, split_ls)

        # list objects storing prediction results
        list_test_mae = list()
        list_test_rmse = list()
        list_test_r2 = list()
        list_preds = list()
        list_embs = list()

        # train and evaluate DopNet for k-fold dataset
        df_preds = pd.DataFrame()
        df_embs = pd.DataFrame()

        for k in range(0, n_folds):
            print('---------------------- Fold [{}/{}] ----------------------'.format(k + 1, n_folds))

            # load training dataset
            dataset_train = list(itertools.chain(*(k_folds[:k] + k_folds[k + 1:])))
            comps_train = [x.comp for x in dataset_train]
            targets_train = numpy.array([x.target for x in dataset_train]).reshape(-1, 1)
            dop_dataset_train = dp.get_dataset(dataset_train, max_dops)
            data_loader_train = DataLoader(dop_dataset_train, batch_size=batch_size, shuffle=True)
            data_loader_calc = DataLoader(dop_dataset_train, batch_size=batch_size)

            # load test dataset
            dataset_test = k_folds[k]
            comps_test = [x.comp for x in dataset_test]
            targets_test = numpy.array([x.target for x in dataset_test]).reshape(-1, 1)
            dop_dataset_test = dp.get_dataset(dataset_test, max_dops)
            data_loader_test = DataLoader(dop_dataset_test, batch_size=32)

            # define host embedding network and its optimizer
            emb_host = ae.Autoencoder(dataset[0].host_feat.shape[0], 64).to(device)
            optimizer_emb = torch.optim.Adam(emb_host.parameters(), lr=emb_init_lr, weight_decay=1e-5)

            # train the host embedding network
            for epoch in range(0, embedding_epochs):
                train_loss = ae.train(emb_host, data_loader_train, optimizer_emb, device)
                if (epoch + 1) % 100 == 0:
                    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, embedding_epochs, train_loss))

            # calculate host embeddings
            host_embs_train = ae.test(emb_host, data_loader_calc, device)
            host_embs_test = ae.test(emb_host, data_loader_test, device)

            # load dataset for DopNet
            dop_dataset_train.host_feats = host_embs_train
            dop_dataset_test.host_feats = host_embs_test
            data_loader_train = DataLoader(dop_dataset_train, batch_size=batch_size, shuffle=True)
            data_loader_calc = DataLoader(dop_dataset_train, batch_size=batch_size)
            data_loader_test = DataLoader(dop_dataset_test, batch_size=batch_size)

            # define DopNet and its optimizer
            pred_model = dp.DopNet(host_embs_train.shape[1], dataset[0].dop_feats.shape[1], dim_out=1, max_dops=max_dops).to(device)
            optimizer = torch.optim.SGD(pred_model.parameters(), lr=dopnet_init_lr, weight_decay=1e-7)
            criterion = torch.nn.L1Loss()

            # train DopNet
            for epoch in range(0, dopnet_epochs):
                if (epoch + 1) % 200 == 0:
                    for g in optimizer.param_groups:
                        g['lr'] *= 0.5

                train_loss = dp.train(pred_model, data_loader_train, optimizer, criterion, device)
                preds_test = dp.test(pred_model, data_loader_test, device).cpu().numpy()
                test_loss = mean_absolute_error(targets_test, preds_test)
                if (epoch + 1) % 100 == 0:
                    print('Epoch [{}/{}]\tTrain loss: {:.4f}\tTest loss: {:.4f}'.format(epoch + 1, dopnet_epochs, train_loss, test_loss))

            # calculate predictions, embeddings, and evaluation metrics
            preds_test = dp.test(pred_model, data_loader_test, device).cpu().numpy()
            embs_test = dp.emb(pred_model, data_loader_test, device).cpu().numpy()
            list_test_mae.append(mean_absolute_error(targets_test, preds_test))
            list_test_rmse.append(numpy.sqrt(mean_squared_error(targets_test, preds_test)))
            list_test_r2.append(r2_score(targets_test, preds_test))

            # save prediction and embedding results to the list objects
            idx_test = numpy.array([x.idx for x in dataset_test])
            data_df_test = df.iloc[idx_test]
            hash_arr = data_df_test['#'].to_numpy().reshape(-1, 1)
            temp_arr = data_df_test[temp_col].to_numpy().reshape(-1, 1) 
            comps_test = numpy.array(comps_test).reshape(-1, 1)

            random_seed_col = numpy.full((data_df_test.shape[0], 1), random_seed)
            n_splits_col = numpy.full((data_df_test.shape[0], 1), k + 1)

            # Create DataFrame for predictions
            preds = numpy.hstack([hash_arr, random_seed_col, n_splits_col, comps_test, temp_arr, targets_test, preds_test])
            embs = numpy.hstack([hash_arr, random_seed_col, n_splits_col,comps_test, temp_arr, targets_test, embs_test])

            embs_test_dim = embs_test.shape[1]
            numbered_columns = list(range(embs_test_dim))  # Create a list of numbers [0, 1, ..., n-1]

            if not df_preds.empty:
                df_preds = pd.concat([df_preds, pd.DataFrame(preds, columns=['#', 'Random Seed', 'K_fold', 'reduced_formula', 'Temperature (K)' , 'target', 'prediction'])])
            else:
                df_preds = pd.DataFrame(preds, columns=['#', 'Random Seed', 'K_fold', 'reduced_formula', 'Temperature (K)', 'target', 'prediction'])

            if not df_embs.empty:
                df_embs = pd.concat([df_embs, pd.DataFrame(embs, columns=['#', 'Random Seed', 'K_fold', 'reduced_formula', 'Temperature (K)', 'target'] + numbered_columns)])
            else:
                df_embs = pd.DataFrame(embs, columns=['#', 'Random Seed', 'K_fold', 'reduced_formula','Temperature (K)', 'target'] + numbered_columns)
        
        # save prediction end embedding results as files
        pandas.DataFrame(df_preds).to_csv(f'results/DopNet/{date_str}_{target}_preds_CompKFold_dopnet_ratio{ratio}_{random_seed}.csv')
        pandas.DataFrame(df_embs).to_csv(f'results/DopNet/{date_str}_{target}_embs_CompKFold_dopnet_ratio{ratio}_{random_seed}.csv')

        # print evaluation results
        print('Test MAE: ' + str(numpy.mean(list_test_mae)))
        print('Test RMSE: ' + str(numpy.mean(list_test_rmse)))
        print('Test R2: ' + str(numpy.mean(list_test_r2)))
