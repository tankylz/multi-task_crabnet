import pandas as pd
from matminer.featurizers.base import MultipleFeaturizer
import os
from matminer.featurizers.conversions import CompositionToOxidComposition
from pandas import DataFrame
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element
import matminer.featurizers.composition as cf
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple, List

# global variables
REDUCED_COMP_COL = "reduced_compositions"


def batch_matminer_featurize_comp_to_df(mat_arr, featurizers_arr, oxidation_states=True, drop_composition_cols=True, ignore_err=True):
    """
    Featurize a list or array of compositions using Matminer featurizers and return a DataFrame of features.

    Parameters
    ----------
    mat_arr : array-like of pymatgen.Composition or str
        Compositions to featurize. Can be pymatgen Composition objects or valid chemical formula strings.

    featurizers_arr : array-like of matminer Composition featurizer objects
        List of Matminer featurizers to apply to each composition.

    oxidation_states : bool, default=True
        If True, convert compositions to their oxidized forms before featurization (when supported).

    drop_composition_cols : bool, default=True
        If True, drop the original composition columns (and oxidized columns if present) from the output DataFrame.

    ignore_err : bool, default=True
        If True, ignore errors during featurization and return a separate DataFrame of error columns.

    Returns
    -------
    feat_df : pd.DataFrame
        DataFrame of featurized compositions. Each row corresponds to a composition, each column to a feature.
    errors_df : pd.DataFrame
        DataFrame of error columns (if ignore_err=True), otherwise empty.

    Notes
    -----
    - Uses MultipleFeaturizer to combine all provided featurizers.
    - Columns are flattened to single-level names for compatibility.
    - If AtomicOrbitals featurizer is used, additional postprocessing is applied to convert categorical features to numeric.
    - Prints the number of features generated and the shape of the resulting DataFrame.
    """

    # Convert mat_arr to a DataFrame
    df = pd.DataFrame({'mat': mat_arr})

    # Try to add oxidation states to the composition
    if oxidation_states:
        cto = CompositionToOxidComposition()
        df = cto.featurize_dataframe(df, 'mat')
        col_id = 'composition_oxid'  # Use the oxidized compositions
    else:
        col_id = 'mat'  # Use the original compositions

    # Fit the featurizers if needed
    fitted_feat_ls = []
    for featurizer in featurizers_arr:
        if hasattr(featurizer, 'fit'):
            # print(f"Fitting featurizer: {featurizer.__class__.__name__}")
            fitted_feat_ls.append(featurizer.fit(df[col_id]))
        else:
            fitted_feat_ls.append(featurizer)

    # Use MultipleFeaturizer to combine all featurizers
    combined_featurizer = MultipleFeaturizer(fitted_feat_ls)

    # Check the number of features that will be generated
    features_len = sum(len(f.feature_labels()) for f in combined_featurizer.featurizers)
    print(f"There will be a maximum of {features_len} types of features generated.")

    combined_featurizer.set_n_jobs(os.cpu_count())

    # Featurize the DataFrame using the correct column
    df = combined_featurizer.featurize_dataframe(df, col_id=col_id, multiindex=True, ignore_errors=ignore_err,
                                                 return_errors=ignore_err)

    error_df = pd.DataFrame()

    if ignore_err:
        error_cols = [col for col in df.columns if any('Exceptions' in str(level) for level in col)]
        error_df = df[error_cols].copy()

        df = df.drop(columns=error_cols)

    # Drop the original 'mat' and 'composition_oxid' columns if they exist
    if drop_composition_cols:
        # df = df.drop(columns=[col for col in ['mat', 'composition_oxid'] if col in df.columns])
        df = df.drop(
            columns=[col for col in [('Input Data', 'mat'), ('Input Data', 'composition_oxid')] if col in df.columns])
    else:
        df = df.rename(columns={"Input Data": ""})  # just a preference to remove the 'Input Data' level

    # Convert the MultiIndex columns to a single level
    df.columns = df.columns.map("|".join).str.strip("|")  # this line is copied from MODNet's feauturizers.py

    if any(isinstance(feat, cf.AtomicOrbitals) for feat in featurizers_arr):
        _orbitals = {"s": 1, "p": 2, "d": 3, "f": 4}
        df["AtomicOrbitals|HOMO_character"] = df[
            "AtomicOrbitals|HOMO_character"
        ].map(_orbitals)
        df["AtomicOrbitals|LUMO_character"] = df[
            "AtomicOrbitals|LUMO_character"
        ].map(_orbitals)

        df["AtomicOrbitals|HOMO_element"] = df["AtomicOrbitals|HOMO_element"].apply(
            lambda x: None if not isinstance(x, str) else Element(x).Z
        )
        df["AtomicOrbitals|LUMO_element"] = df["AtomicOrbitals|LUMO_element"].apply(
            lambda x: None if not isinstance(x, str) else Element(x).Z
        )

    print(f"Shape of the DataFrame: {df.shape}")
    print(df.info())
    return df, error_df


def clean_df_cols(df: pd.DataFrame, composition_type: str = None) -> pd.DataFrame:
    """
    Cleans the featurized DataFrame and appends the composition type to the column names

    Parameters
    ----------
    df: pd.DataFrame
        The Featurized Dataframe

    composition_type: str, optional
        The type of composition used in the featurization (e.g. 'H' for host, 'D' for dopant)

    Returns
    -------
    pd.DataFrame: The cleaned DataFrame with the composition type appended to the column names

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'ElementFraction|O': [0.2, 0.25],
    ...     'ElementFraction|H': [0.5, 0.5],
    ...     'ConstantCol': [1, 1],
    ...     'AllNaNCol': [np.nan, np.nan]
    ... })
    >>> clean_df = clean_df_cols(df, 'H')
    >>> print(clean_df)

       H|ElementFraction|O  H|ElementFraction|H
    0                  0.2                 0.5
    1                 0.25                 0.5

    Notes
    -----
    - The function removes columns where all values are NaN.
    - It also selects numerical columns only.
    - It also removes columns with constant values as they do not provide useful information.
    - Infinite values are replaced with NaN before dropping all-NaN columns.
    """
    df = df.select_dtypes(include="number")  # select only numerical columns

    df = df.replace([np.inf, -np.inf], np.nan)  # replace inf with NaN

    df = df.dropna(axis=1, how="all")  # drop columns that are all NaN

    # drop constant columns
    constant_cols = df.columns[df.nunique() == 1]
    df = df.drop(columns=constant_cols)

    # append the composition type to the column names
    if composition_type:
        df.columns = [f"{composition_type}|{col}" for col in df.columns]

    return df


def impute_missing(df: pd.DataFrame, strategy: str = 'constant', fill_value=0) -> pd.DataFrame:
    """
    Imputes missing values in the DataFrame

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame with missing values

    strategy: str, optional
        The imputation strategy to use. Default is 'constant'.

    fill_value: optional
        The value to replace missing values with when strategy='constant'.
        Default is 0. Ignored for other strategies.

    Returns
    -------
    pd.DataFrame: The DataFrame with missing values imputed

    Notes
    -----
    - The function uses the SimpleImputer from scikit-learn to impute missing values.
    - By default, missing values are replaced with 0 if strategy='constant'.
    """

    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df


def _process_cols_to_scale(cols_to_scale, num_columns):
    """
    Helper function to process cols_to_scale and handle a mix of integers and ranges.
    
    Parameters
    ----------
    cols_to_scale: list
        A list of column indices or ranges to process.
        
    num_columns: int
        The total number of columns in the DataFrame.
        
    Returns
    -------
    List of column indices expanded from the ranges and integers.
    """
    if cols_to_scale is None:
        # If no columns are provided, scale all columns
        return list(range(num_columns))

    # Expand ranges into individual indices
    cols_to_scale_expanded = []
    for col in cols_to_scale:
        if isinstance(col, range):
            cols_to_scale_expanded.extend(list(col))
        elif isinstance(col, int):
            cols_to_scale_expanded.append(col)
        else:
            raise ValueError("cols_to_scale should only contain integers or ranges")

    # Ensure the column indices are unique and sorted
    return sorted(set(cols_to_scale_expanded))


def scale_df(df: pd.DataFrame, scaler=StandardScaler(), fit=True, cols_to_scale: list = None) -> tuple[
    DataFrame, StandardScaler | object]:
    """
    Scales the DataFrame using a specified scaler and allows scaling of specific columns.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to be scaled.

    scaler: object, optional
        The pre-initialized scaler object to use (e.g., StandardScaler(), MinMaxScaler()). Default is None.

    fit: bool, optional
        If True, fit the scaler on the provided data. If False, only transform the data using the provided scaler.

    cols_to_scale: list, optional
        List of column indices or ranges to scale. Default is None, which means the entire DataFrame is scaled.

    Returns
    -------
    pd.DataFrame: The scaled DataFrame.
    scaler: The fitted scaler (if `fit=True`). This can be reused for future transformations.
    """        

    # Create a copy of the DataFrame to avoid modifying the original directly
    df_copy = df.copy()

    if scaler is None:
        return df_copy, None

    # Process cols_to_scale to handle lists, ranges, or None
    cols_to_scale = _process_cols_to_scale(cols_to_scale, df_copy.shape[1])

    # Ensure the specified columns are converted to float64
    df_copy.iloc[:, cols_to_scale] = df_copy.iloc[:, cols_to_scale].astype(float)

    # Scale the specified columns
    if fit:
        df_copy.iloc[:, cols_to_scale] = scaler.fit_transform(df_copy.iloc[:, cols_to_scale])
    else:
        df_copy.iloc[:, cols_to_scale] = scaler.transform(df_copy.iloc[:, cols_to_scale])

    return df_copy, scaler


def columns_with_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies columns with missing values in the DataFrame

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to be checked for missing values

    Returns
    -------
    pd.DataFrame: A DataFrame with columns that have 1 or more NaN values and the corresponding count of NaNs
    """

    # Select columns with at least one missing value
    na_columns = df.columns[df.isna().any()].tolist()

    # Create a DataFrame showing the count of missing values per column
    na_counts = df[na_columns].isna().sum().reset_index()
    na_counts.columns = ['Column', 'MissingCount']

    return na_counts


def reduce_comps_ls(compositions : list) -> list:
    """
    Normalize the list or array of compositions by converting each entry to its reduced formula.
    This ensures that compositions are order-independent and standardized.
    
    Parameters
    ----------
    compositions : list or array-like
        A list or array-like structure containing `pymatgen.Composition` objects or valid strings.
    
    Returns
    -------
    normalized_compositions : list of str or pymatgen.Composition
        A list of string or pymatgen.Composition representations of the reduced formulas of the compositions.

        Examples
    --------
    >>> compositions = [Composition("NaCl"), Composition("ClNa"), Composition("Na2Cl2")]
    >>> normalized = reduce_comps_ls(compositions)
    >>> print([str(comp) for comp in normalized])
    ['NaCl', 'NaCl', 'NaCl']
    """
    try:
        normalized_compositions = [
            comp.reduced_formula if isinstance(comp, Composition) else Composition(comp).reduced_formula
            for comp in compositions
        ]
        return normalized_compositions
    except Exception as e:
        raise ValueError("Invalid Composition data found. Ensure all entires are valid chemical formula.")



def reduce_comps_in_df(df, composition_column, inplace=False):
    """
    Normalize the compositions in the DataFrame by converting each entry to its reduced formula.
    This ensures that the compositions are order-independent and standardized for further grouping.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing composition data.
    
    composition_column : str
        The name of the column containing `pymatgen.Composition` or str objects.
    
    inplace : bool, optional
        If True, modifies the DataFrame directly and adds the 'normalized_composition' column in place.
        If False, returns a new DataFrame with the normalized compositions. Default is False.
    
    Returns
    -------
    pd.DataFrame
        If `inplace` is False, returns a DataFrame with an additional 'normalized_composition' column,
        containing the reduced formula of the compositions.
    
    Notes
    -----
    - The function uses `normalize_compositions()` to normalize the compositions.
    
    Examples
    --------
    >>> from pymatgen.core import Composition
    >>> import pandas as pd
    >>> data = {'composition': [Composition("NaCl"), Composition("ClNa"), Composition("Na2Cl2")]}
    >>> df = pd.DataFrame(data)
    >>> reduce_comps_in_df(df, 'composition', inplace=True)
    >>> print(df)
    """
    if not inplace:
        # Create a copy of the DataFrame to avoid modifying the original
        df = df.copy()

    # Apply normalize_compositions to the specified column
    df[REDUCED_COMP_COL] = reduce_comps_ls(df[composition_column])
    composition_idx = df.columns.get_loc(composition_column)
    df.insert(composition_idx + 1, REDUCED_COMP_COL, df.pop(REDUCED_COMP_COL))  # insert next to the composition column

    # Return the modified DataFrame if inplace is False
    if not inplace:
        return df


def count_identical_rows(df: pd.DataFrame, *columns, sort=True, normalize_comp=False):
    """
    Groups the DataFrame by the specified columns and counts the number of identical rows.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be grouped.
        
    *columns : str
        The column names by which to group the DataFrame.
        
    sort : bool, optional
        If True, the results are sorted by the count in descending order. Default is True.
        
    normalize_comp : bool, optional
        If True, normalizes composition columns using pymatgen's Composition.
        This ensures order-independent grouping of compositions (e.g., NaCl == ClNa).
    
    Returns
    -------
    pd.DataFrame
        A DataFrame showing the unique groups and the count of identical rows for each group.
    
    Raises
    ------
    ValueError
        If `normalize_compositions` is True but no composition columns are found.
    
    Notes
    -----
    - The function groups the DataFrame by the provided columns and counts how many identical rows 
      are present for each group.
    - If `sort` is set to True, the DataFrame is sorted by the count in descending order.
    
    Examples
    --------
    >>> data = {'A': [1, 2, 1, 1, 3, 2, 2, 3],
                'B': ['x', 'y', 'x', 'x', 'z', 'y', 'y', 'z']}
    >>> df = pd.DataFrame(data)
    >>> count_identical_rows(df, 'A', 'B', sort=True)
    
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Convert the passed columns into a list
    columns = list(columns)

    # Check if we need to normalize compositions
    if normalize_comp:
        composition_columns_found = False
        for col in columns:
            # If the column contains Composition objects, normalize it
            if df[col].apply(lambda x: isinstance(x, Composition)).any():
                reduce_comps_in_df(df, col, inplace=True)
                # Replace the composition column with the normalized composition column
                columns[columns.index(col)] = REDUCED_COMP_COL  # assuming normalized_comp_col is a global variable
                composition_columns_found = True

        # If no Composition columns were found for normalization, raise an error
        if not composition_columns_found:
            raise ValueError(
                "No composition column found for normalization. Please provide a composition column to normalize.")

    # Group the dataframe by the specified columns and get the size of each group
    grouped = df.groupby(columns).size().reset_index(name='count')

    # If sort is True, sort the grouped dataframe by count in descending order
    if sort:
        grouped = grouped.sort_values(by='count', ascending=False).reset_index(drop=True)

    return grouped

def keep_highest_duplicates(df, identical_counts_df, target_col):
    """
    Removes duplicate rows based on all columns in identical_counts_df except target_col,
    keeping only the row with the highest target_col value.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame.
    identical_counts_df : pd.DataFrame
        DataFrame containing the count of identical rows and the columns used for filtering. 
        It is typically generated from count_identical_rows().
    target_col : str
        The name of the column based on which duplicates will be filtered.

    Returns
    -------
    pd.DataFrame
        A DataFrame with duplicates removed, keeping only the rows with the highest target_col value.
    """
    # Determine columns to group by
    columns = [col for col in identical_counts_df.columns if col != target_col and col != 'count']

    # Filter for rows with 'count' > 1 in identical_counts_df
    duplicate_combinations = identical_counts_df[identical_counts_df['count'] > 1][columns]

    # Create a mask to identify rows in df matching the duplicate combinations
    mask = df[columns].apply(tuple, axis=1).isin(duplicate_combinations.apply(tuple, axis=1))

    # Filter the original DataFrame for rows matching the duplicate combinations
    duplicates_df = df[mask]

    # Sort the duplicates DataFrame by columns and target_col in descending order
    sorted_duplicates_df = duplicates_df.sort_values(by=columns + [target_col], ascending=[True] * len(columns) + [False])

    # Drop duplicate rows, keeping only the first occurrence (highest target_col value)
    unique_highest_df = sorted_duplicates_df.drop_duplicates(subset=columns, keep='first')

    # Combine unique rows with non-duplicate rows
    final_df = pd.concat([df[~mask], unique_highest_df], ignore_index=True)

    return final_df

def average_duplicates(df, identical_counts_df, target_col):
    """
    Removes duplicate rows based on all columns in identical_counts_df except 'count',
    averaging the target_col value(s) for duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame.
    identical_counts_df : pd.DataFrame
        DataFrame containing the count of identical rows and the columns used for filtering. 
        It is typically generated from count_identical_rows().
    target_col : str or list of str
        The name(s) of the column(s) based on which duplicates will be averaged.

    Returns
    -------
    pd.DataFrame
        A DataFrame with duplicates removed, with the target_col value(s) averaged for duplicates.
    """
    # Determine columns to group by
    columns = [col for col in identical_counts_df.columns if col != 'count']

    # Ensure data types of columns in df match those in identical_counts_df
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(identical_counts_df[col].dtype)

    # Filter for rows with 'count' > 1 in identical_counts_df
    duplicate_combinations = identical_counts_df[identical_counts_df['count'] > 1][columns]

    # Create a mask to identify rows in df matching the duplicate combinations
    duplicate_combinations_set = set(duplicate_combinations.apply(tuple, axis=1))
    mask = df[columns].apply(tuple, axis=1).isin(duplicate_combinations_set)

    # Prepare target columns for aggregation
    if isinstance(target_col, str):
        agg_dict = {target_col: 'mean'}
    else:
        agg_dict = {col: 'mean' for col in target_col}

    # Filter duplicate rows and compute their averages
    duplicates_df = df[mask]
    averaged_duplicates_df = duplicates_df.groupby(columns, as_index=False).agg(agg_dict)

    # Combine averaged duplicates with non-duplicate rows
    non_duplicates_df = df[~mask]
    final_df = pd.concat([non_duplicates_df, averaged_duplicates_df], ignore_index=True)

    print(f"Removed {len(duplicates_df) - len(averaged_duplicates_df)} duplicate rows. Generated {len(averaged_duplicates_df)} averaged rows.")
    return final_df

class CompositionKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        """
        Initializes the CompositionKFold instance.

        Parameters
        ----------
        n_splits : int, optional
            Number of folds for cross-validation. Default is 3.
        
        shuffle : bool, optional
            Whether to shuffle the compositions before splitting. Default is True.
        
        random_state : int, optional
            Random seed for shuffling. Default is None.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.fold_compositions_ = []  # This will store unique compositions for each fold, 

    def split(self, compositions):
        """
        Split the dataset into mutually exclusive folds based on normalized compositions.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The input dataset. This is not used directly for splitting, but can be useful to align the indices.
        
        compositions : array-like
            Array-like structure containing the composition information for each row in X.
            Compositions should be `pymatgen.Composition` objects.
        
        Yields
        ------
        train_idx, test_idx : np.array
            The indices for the train and test sets for each fold.
        """

        # Normalize compositions using the new normalize_compositions function
        normalized_compositions = reduce_comps_ls(compositions)

        # Group indices by unique normalized compositions
        unique_compositions, composition_indices = np.unique(normalized_compositions, return_inverse=True)

        # Optionally shuffle the unique compositions
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)

            # Shuffle the indices
            shuffled_idx = np.arange(len(unique_compositions))
            rng.shuffle(shuffled_idx)

            shuffled_unique_compositions = unique_compositions[shuffled_idx]

            # Create a mapping of original to shuffled indices
            index_mapping = {old: new for new, old in enumerate(shuffled_idx)}

            shuffled_composition_indices = np.array([index_mapping[idx] for idx in composition_indices])

        # Create the KFold object to split unique compositions
        kf = KFold(n_splits=self.n_splits, shuffle=False)

        # Clear previous fold compositions
        self.fold_compositions_ = []

        # Split based on unique compositions
        for train_compositions, test_compositions in kf.split(shuffled_unique_compositions):
            train_idx = np.where(np.isin(shuffled_composition_indices, train_compositions))[0]
            # np.isin will return a boolean array corresponding to whether each element in composition_indices is in
            # train_compositions np.where will return the indices of the True values in the boolean array in a single
            # length
            test_idx = np.where(np.isin(shuffled_composition_indices, test_compositions))[0]

            # Store unique compositions in this fold
            self.fold_compositions_.append({
                'train_compositions': shuffled_unique_compositions[train_compositions],
                'test_compositions': shuffled_unique_compositions[test_compositions]
            })

            yield train_idx, test_idx

    def get_fold_compositions(self, fold_index):
        """
        Get the unique compositions in the training and test sets for a given fold.

        Parameters
        ----------
        fold_index : int
            The index of the fold (0-based).
        
        Returns
        -------
        dict
            A dictionary with 'train_compositions' and 'test_compositions' as keys, which hold the unique compositions for the respective sets.
        """
        if fold_index >= len(self.fold_compositions_):
            raise ValueError(f"Invalid fold index {fold_index}. There are only {len(self.fold_compositions_)} folds.")
        return self.fold_compositions_[fold_index]


def check_composition_intersections(inner_train_compositions_unique, inner_val_compositions_unique,
                                    test_compositions_unique):
    """
    Check for intersections between train, validation, and test compositions.

    Parameters:
    inner_train_compositions_unique (array-like): Unique compositions in the inner training set.
    inner_val_compositions_unique (array-like): Unique compositions in the inner validation set.
    test_compositions_unique (array-like): Unique compositions in the test set.

    Raises:
    ValueError: If there are intersections between train, validation, and test sets.
    """
    # Intersection checks using sets (unique compositions from fold_compositions_)
    train_test_overlap = set(inner_train_compositions_unique) & set(test_compositions_unique)
    val_test_overlap = set(inner_val_compositions_unique) & set(test_compositions_unique)
    train_val_overlap = set(inner_train_compositions_unique) & set(inner_val_compositions_unique)

    if train_test_overlap:
        print(f"Overlap between train and test compositions: {train_test_overlap}")
        raise ValueError("There is an intersection between train and test compositions!")
    if val_test_overlap:
        print(f"Overlap between validation and test compositions: {val_test_overlap}")
        raise ValueError("There is an intersection between validation and test compositions!")
    if train_val_overlap:
        print(f"Overlap between train and validation compositions: {train_val_overlap}")
        raise ValueError("There is an intersection between train and validation compositions!")


def drop_missing_targets(df: pd.DataFrame, target_data: Union[pd.DataFrame, pd.Series, str, List[str]]) -> tuple:
    """
    Remove rows from the DataFrame and target data that contain missing values in the target data or target columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the main dataset.
        
    target_data : Union[pd.DataFrame, pd.Series, str, List[str]]
        The target dataset or the name(s) of the target columns. If given as a DataFrame or Series, any row with 
        missing values in the target data will be removed from both `df` and `target_data`. If given as a string or
        list of strings, rows will be removed if any value in those columns contains a missing value.

    Returns
    -------
    tuple
        A tuple containing:
        - `df_clean` (pd.DataFrame): The cleaned DataFrame with rows containing missing target values removed.
        - `target_data_clean` (Union[pd.DataFrame, pd.Series]): The cleaned target data with rows containing 
          missing values removed.

    Notes
    -----
    If `target_data` is a DataFrame, rows will be dropped if any value in that row contains a missing value.
    If `target_data` is a Series, rows will be dropped if the corresponding value in the Series is missing.
    If `target_data` is a string or list of strings, rows will be dropped based on missing values in those columns.
    
    Examples
    --------
    >>> import pandas as pd
    >>> data = {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], 'Target': [10, None, 30, 40]}
    >>> df = pd.DataFrame(data)
    >>> df_clean, target_clean = drop_missing_targets(df, 'Target')
    >>> print(df_clean)
       A  B  Target
    0  1  5    10.0
    2  3  7    30.0
    3  4  8    40.0
    >>> print(target_clean)
    0    10.0
    2    30.0
    3    40.0
    Name: Target, dtype: float64

    >>> target_series = pd.Series([10, None, 30, 40], name='Target')
    >>> df_clean, target_clean = drop_missing_targets(df, target_series)
    >>> print(df_clean)
       A  B
    0  1  5
    2  3  7
    3  4  8
    >>> print(target_clean)
    0    10.0
    2    30.0
    3    40.0
    Name: Target, dtype: float64
    """
    if isinstance(target_data, (pd.DataFrame, pd.Series)):
        # Original behavior: drop rows based on NaNs in the target DataFrame or Series
        non_missing_idx = target_data.dropna().index
        df_clean = df.loc[non_missing_idx]
        target_data_clean = target_data.loc[non_missing_idx]
        return df_clean, target_data_clean
    elif isinstance(target_data, (str, list)):
        # New behavior: drop rows based on NaNs in the specified target columns
        if isinstance(target_data, str):
            target_data = [target_data]
        df_clean = df.dropna(subset=target_data)
        target_data_clean = df_clean[target_data]

        df_clean = df_clean.drop(columns=target_data)
        
        # If there's only one target column, return it as a Series
        if len(target_data) == 1:
            target_data_clean = target_data_clean[target_data[0]]

        return df_clean, target_data_clean
    else:
        raise TypeError("target_data must be a DataFrame, Series, str, or list of column names")


def scientific_to_numeric_compositions(composition, decimal_places=6):
    """
    Convert subscripts in a Composition from scientific (e.g. 1e-3) to numeric (e.g. 0.001) notation. This is to ensure the compositions are parsed property by some models.

    Parameters
    ----------
    composition : Composition
        The Composition object to convert.

    decimal_places : int, optional
        The maximum number of decimal places to use for the numeric notation. Default is 6.

    Returns
    -------
    str
        The Composition as a string with numeric notation for the subscripts.

    """
    if not isinstance(composition, Composition):
        try:
            composition = Composition(composition)
        except Exception as e:
            raise ValueError("Invalid Composition data found. Ensure all entires are valid chemical formula.")

    # Create a dictionary with elements and their amounts
    element_amounts = composition.get_el_amt_dict()
    # Convert each amount to a string with regular notation
    format_string = f"{{:.{decimal_places}f}}"
    converted = {
        el: format_string.format(amt).rstrip('0').rstrip('.') if amt < 1 or amt != int(amt) else str(int(amt))
        for el, amt in element_amounts.items()
    }
    # Join the elements with their amounts as a string
    return ''.join([f"{el}{amt}" for el, amt in converted.items()])

def normalize_comp(comp, return_type='composition'):
    """
    Normalize a pymatgen Composition object so the sum of atomic fractions equals 1.

    Args:
        comp (Composition or str): The pymatgen Composition object or string to normalize.
        return_type (str): The return type - 'composition' for a pymatgen Composition 
                           or 'dict' for a dictionary of normalized atomic fractions.

    Returns:
        Composition or dict: Normalized composition as specified by `return_type`.
    """
    # Convert string input to Composition, ignore case
    if isinstance(comp, str):
        comp = Composition(comp)

    if not isinstance(comp, Composition):
        raise TypeError("Input must be a pymatgen Composition object or a valid chemical formula string.")
    
    # Normalize the composition
    normalized_comp = {el: amt / comp.num_atoms for el, amt in comp.items()}
    
    if return_type.lower() == 'composition':
        return Composition(normalized_comp)
    elif return_type.lower() == 'dict':
        return normalized_comp
    else:
        raise ValueError("Invalid return_type. Use 'composition' or 'dict'.")


def normalize_comp_array(comp_array, return_type='composition'):
    """
    Normalize an array of pymatgen Composition objects or string representations.

    Args:
        comp_array (list, pd.Series, or similar): Array-like structure containing compositions.
        return_type (str): The return type - 'composition' for pymatgen Composition objects
                           or 'dict' for dictionaries of normalized atomic fractions.

    Returns:
        pd.Series: Normalized compositions as specified by `return_type`.
    """
    try:
        normalized_array = [
            normalize_comp(comp, return_type=return_type) for comp in comp_array
        ]
        return pd.Series(normalized_array, name="Normalized Composition")
    except Exception as e:
        raise ValueError(f"Error processing compositions: {e}")
