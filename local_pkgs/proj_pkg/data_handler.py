from pymatgen.core.composition import Composition
import pandas as pd
import re
from . import utils
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def generate_one_hot(df, property_col):
    """
    Generate a dictionary of one-hot encoded vectors for unique properties in a column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the property column.
    property_col : str
        Column name with categorical properties to be one-hot encoded.

    Returns
    -------
    dict
        A dictionary where keys are unique property values and values are one-hot encoded vectors.
    """
    # Get unique properties
    unique_properties = df[property_col].unique()
    
    # Generate one-hot encoded vectors
    one_hot_dict = {prop: np.eye(len(unique_properties))[i] for i, prop in enumerate(unique_properties)} # one-hot encoding using an identity matrix
    
    return one_hot_dict

def add_one_hot_column(df, property_col, output_col="one_hot_vector"):
    """
    Adds a column of one-hot encoded vectors to the DataFrame based on a specified column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - property_col (str): The name of the column containing property names.

    Returns:
    - pd.DataFrame: DataFrame with an added column of one-hot encoded vectors.
    - OneHotEncoder: The fitted OneHotEncoder for potential reuse.
    """
    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  
    
    # Fit and transform the property column
    one_hot_encoded = encoder.fit_transform(df[[property_col]])
    
    # Convert each row of the one-hot encoding into a list and add it as a new column
    df[output_col] = list(one_hot_encoded)
    
    return df, encoder 


def split_chemicals(formula):
    """
    Split a chemical formula into individual chemical components based on '+' or '-' separators.

    This function splits the input string `formula` into a list of individual chemical components.
    The formula is split at occurrences of '+' or '-' (considered as separators), and any surrounding
    whitespace is removed. If neither '+' nor '-' is present, the entire formula is returned as a 
    single element in the list.

    Parameters
    ----------
    formula : str
        The chemical formula string to be split, where '+' or '-' separate the components.

    Returns
    -------
    list of str
        A list of chemical components from the formula, split at '+' or '-'. If no separators are 
        found, the entire formula is returned as a single element list.

    Examples
    --------
    >>> split_chemicals("H2O + NaCl - CO2")
    ['H2O', 'NaCl', 'CO2']
    
    >>> split_chemicals("C6H12O6")
    ['C6H12O6']
    """

    if '+' in formula or '-' in formula:
        # Split based on + or -. Note that + or - are just separators, not operators
        chemicals = re.split(r'\s*[\+\-]\s*', formula)
    else:
        # Return the whole formula as one piece
        chemicals = [formula]
    return chemicals

def process_mixed_formula_to_composition(formula):
    """
    Convert a mixed chemical formula into a composition based on weight percentages.

    This function takes a chemical formula string, which may contain multiple components 
    separated by '+' or '-' and with optional weight percentages (wt%). It calculates 
    the molar fractions for each component based on their weight percentages and combines 
    them into a single composition. If no weight percentage is specified for a component, 
    it is assumed to contribute 100 wt%.

    Parameters
    ----------
    formula : str
        A mixed chemical formula string with optional weight percentages (e.g., 
        "50wt% H2O + 50wt% NaCl").

    Returns
    -------
    Composition
        A composition object representing the total composition derived from the 
        components in the formula, with elements and their molar fractions.

    Examples
    --------
    # Example 1: H2O and NaCl with equal weight percentages
    # wt% (or wt %) can be specified before or after the chemical formula
    >>> formula = "50 wt% (H2O) + NaCl 50wt%"
    >>> result = process_mixed_formula_to_composition(formula)
    >>> result
    Composition({'H': 5.551, 'O': 2.775, 'Na': 0.855, 'Cl': 0.855})

    # Example 2: H2O and NaCl. H2O is assumed to be 100 wt% if no wt% is specified
    >>> formula = " H2O + 50wt% H"
    >>> result = process_mixed_formula_to_composition(formula)
    >>> result
    Composition({'H': 60.708 , 'O': 5.550})

    # Example 3: Same as example 2. Note only the ratios of the chemical formula matters. Thus the result is the same even if H2 is used instead of H
    >>> formula = " H2O + 50wt% H2"
    >>> result = process_mixed_formula_to_composition(formula)
    >>> result
    Composition({'H': 60.708 , 'O': 5.550})

    # Example 4: Same as example 2. Note only the ratios of the chemical formula matters. Thus the result is the same even if (H2O)2 is used instead of H2O
    >>> formula = " (H2O)2 + 50wt% H2"
    >>> result = process_mixed_formula_to_composition(formula)
    >>> result
    Composition({'H': 60.708 , 'O': 5.550})

    # Example 5: O2 and H2O. O2 is assumed to be 100 wt% if no wt% is specified
    >>> formula = "O2 + H2O 15 wt%"
    >>> result = process_mixed_formula_to_composition(formula)
    >>> result
    Composition({'O': 7.083, 'H': 1.665})

    Notes
    -----
    - The function assumes that each component's weight percentage is specified using the 'wt%' 
      notation. If a component does not have a specified 'wt%' value, it is assumed to represent 
    - The calculation of molar fractions is based on the total weight percent and the molar mass 
      of each component.
    """

    component_dict = {}

    # Split based on + or - first. Note - is a hyphen sign, not minus
    chemicals = split_chemicals(formula)

    for chem in chemicals:
        # Check if wt% is in the item
        if 'wt%' in chem:
            # Extract wt% value and the formula
            wt_match = re.search(r'(\d*\.?\d+)\s*wt%', chem)
            formula_match = re.search(r'[A-Z][a-z]?\d*\.?\d*(\([\w\d]+\))?[A-Za-z\d]*', chem)

            if wt_match and formula_match:
                wt_percent = float(wt_match.group(1))

                # Extract the chemical formula
                formula_str = formula_match.group(0)
                comp = Composition(formula_str)
                molar_mass = comp.weight
                mol_fraction = wt_percent / molar_mass

                for element, amount in comp.as_dict().items():
                    component_dict[element] = component_dict.get(element, 0) + amount * mol_fraction 
        else:
            # take this whole part as 100g or 100wt% of compound 
            wt_percent =  100

            comp = Composition(chem)
            molar_mass = comp.weight

            mol_fraction = wt_percent / molar_mass

            for element, amount in comp.as_dict().items():
                component_dict[element] = component_dict.get(element, 0) + amount * mol_fraction
    
    
    final_comp = Composition(component_dict)

    return final_comp


def convert_to_composition(formula: pd.Series, formula_type: pd.Series, output_name: str = "pymatgen Composition") -> pd.Series:
    """
    Converts a given series of chemical formulas into pymatgen Composition objects based on their type.
    
    Parameters
    ----------
    formula : pd.Series
        Series containing the chemical formulas.
    formula_type : pd.Series
        Series indicating whether the formula is "Stoichiometric Formula" or "Mixed Formula".
    output_name : str, optional
        Name of the output series. Default is "pymatgen Composition".

    Returns
    -------
    pd.Series
        A new series with pymatgen Composition objects. Entries that could not be converted are set to None.
    
    Raises
    ------
    ValueError
        If the input series are of different lengths or if the formula type is not recognized.
    
    Notes
    -----
    - "Stoichiometric Formula" types are directly converted to pymatgen Composition objects.
    - "Mixed Formula" types are expected to contain 'wt%' and are processed with `process_mixed_formula_to_composition`.
    - The function will print errors for individual entries that cannot be converted, while still returning a Series of the same length by replacing errors with None.

    """
    # Check if both series are of the same size
    if len(formula) != len(formula_type):
        raise ValueError("The input series must be of the same length.")

    # Initialize an empty list to store the results
    compositions = []
    iter = 0
    # Iterate through each formula and type
    for formula, ftype in zip(formula, formula_type):
        try:
            if ftype == "Stoichiometric Formula":
                # Direct conversion for stoichiometric formulas
                try:
                    # converts the percentage to a fraction, if any
                    formula = re.sub(r'(\d+\.?\d*)%', lambda match: str(utils.convert_percentage_to_fraction(match)), formula)

                    comp = Composition(formula)
                    compositions.append(comp)
                except Exception as e:
                    compositions.append(None)
                    print(f"Error converting stoichiometric formula '{formula}', entry {iter} to Composition: {e}")

            elif ftype == "Mixed Formula":
                if 'wt%' in formula:
                    try:
                        comp = process_mixed_formula_to_composition(formula)
                        compositions.append(comp)
                    except Exception as e:
                        compositions.append(None)
                        print(f"Error converting mixed formula '{formula}', entry {iter} to Composition: {e}")

                else: 
                    raise ValueError("The formula is not in the expected format for mixed formula. It should contain wt%")

            else:
                raise ValueError(f"Formula type '{ftype}' is not recognized.")
            
        except ValueError as ve:
            # Log the error and append None to keep the series length consistent
            compositions.append(None)
            print(f"ValueError encountered at entry {iter}: {ve}")
        iter += 1

    # Convert the list to a pandas series with the desired output name
    return pd.Series(compositions, name=output_name)


# Function to classify dopant and host
def classify_host_dopant(composition, threshold):
    """
    Classifies elements in a chemical composition as either dopants or hosts based on a given threshold.

    Parameters
    ----------
    composition : pymatgen.core.Composition
        The chemical composition to classify, represented as a pymatgen Composition object.
    threshold : str or float
        The threshold used to classify elements as dopants. Can be a percentage string (e.g., '5%')
        or a decimal fraction (e.g., 0.05). Elements with a fraction of the total composition
        at or below this threshold are classified as dopants.

    Returns
    -------
    hosts : list of str 
        Elements classified as hosts.
    dopants : list of str or None)
        Elements classified as dopants, or None if no dopants.

    Raises
    ------
    ValueError
        If no host element is found, indicating the threshold may be too high.

    Notes
    -----
    The function ensures at least one host element is present in the composition. If all elements
    fall below the threshold, a ValueError is raised.

    Examples
    --------
    >>> from pymatgen.core import Composition
    >>> comp = Composition("NaCl")
    >>> classify_host_dopant(comp, '5%')
    (['Na1.0', 'Cl1.0'], None)
    """

    # Check and update the threshold
    threshold = utils.convert_percentage_to_fraction(threshold)

    # Get the total amount of all elements in the composition
    total_amount = sum(composition.get_el_amt_dict().values())

    dopants = []
    hosts = []

    # Classify each element
    for element, amount in composition.get_el_amt_dict().items():
        fraction = amount / total_amount
        element_amt = f"{element}{amount}"

        if fraction <= threshold:
            dopants.append(element_amt)
        else:
            hosts.append(element_amt)

    # Ensure there is at least one host
    if not hosts:
        raise ValueError("No host element found in the composition. The threshold may be too high.")

    # Return None for dopants if empty
    return hosts, dopants if dopants else None

# Function to apply classify_dopant_host to a series of compositions
def classify_host_dopant_bulk(composition_series, threshold):
    """
    Applies `classify_host_dopant` to a series of compositions, returning a DataFrame with hosts and dopants.

    Parameters
    ----------
    composition_series : pd.Series
        A pandas Series containing pymatgen Composition objects to classify.
    threshold : str or float
        The threshold used to classify elements as dopants, either as a percentage string (e.g., '5%')
        or a decimal fraction (e.g., 0.05). The threshold is applied uniformly across all compositions.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns:
        - 'Host(x%)': Lists of host elements for each composition, where x is the host percentage.
        - 'Dopant(y%)': Lists of dopant elements for each composition, where y is the dopant percentage.
          If no dopants are present for a composition, the entry will be None.

    Notes
    -----
    - This function iterates over the provided series of compositions and classifies elements as hosts
      or dopants based on the given threshold.
    - The returned DataFrame column names will reflect the dopant and host percentages.

    Examples
    --------
    >>> from pymatgen.core import Composition
    >>> import pandas as pd
    >>> comps = pd.Series([Composition("NaCl"), Composition("KBr"), Composition("LiF")])
    >>> classify_host_dopant_bulk(comps, '1%')
        Host(99.0%) Dopant(1.0%)
    0   Na1.0 Cl1.0         None
    1   K1.0 Br1.0          None
    2   Li1.0 F1.0          None
    """
    threshold = utils.convert_percentage_to_fraction(threshold)

    host_percent = (1 - threshold) * 100
    dopant_percent = threshold * 100

    # Lists to store results
    hosts_list = []
    dopants_list = []

    # map to composition
    composition_series = composition_series.map(lambda x: Composition(x))

    # Apply the classify_dopant_host function to each composition in the series
    for comp in composition_series:
        hosts, dopants = classify_host_dopant(comp, threshold)
        hosts_list.append(' '.join(hosts))  # Convert to string for the DataFrame
        dopants_list.append(' '.join(dopants) if dopants else None)

    # Create a DataFrame with Hosts and Dopants columns
    result_df = pd.DataFrame({
        f'Host({host_percent}%)': hosts_list,
        f'Dopant({dopant_percent}%)': dopants_list
    })

    return result_df


def thermal_conductivity_verification(df, total_col="Total Thermal Conductivity (W/mK)", 
                                      lattice_col="Lattice Thermal Conductivity (W/mK)", 
                                      electronic_col="Electronic Thermal Conductivity (W/mK)", verbose=0):
    """
    Verifies and fills missing thermal conductivity values in a DataFrame based on the relationship:
    Total Thermal Conductivity = Lattice Thermal Conductivity + Electronic Thermal Conductivity.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing thermal conductivity data.
    total_col : str, optional
        Column name for total thermal conductivity. Default is "Total Thermal Conductivity (W/mK)".
    lattice_col : str, optional
        Column name for lattice thermal conductivity. Default is "Lattice Thermal Conductivity (W/mK)".
    electronic_col : str, optional
        Column name for electronic thermal conductivity. Default is "Electronic Thermal Conductivity (W/mK)".
    verbose : int, optional
        Verbosity level:
        - 0 (default): Minimal output.
        - 1: Prints a summary of modifications and verification results.

    Returns
    -------
    pd.DataFrame
        A modified DataFrame with an additional column indicating modifications and verification results:
        - "False" if less than two values are present.
        - "k_total added" if total thermal conductivity was calculated.
        - "k_lattice added" if lattice thermal conductivity was calculated.
        - "k_electronic added" if electronic thermal conductivity was calculated.
        - Absolute percentage difference when all values are present.

    Notes
    -----
    - If any two values are present, the third value is calculated to satisfy the equation:
      Total = Lattice + Electronic.
    - When all three values are present, the function computes the absolute percentage difference to check
      for consistency.
    - If `verbose` is set to 1, a summary of the total rows, counts of each modification type, and fully filled
      entries is printed.

    Examples
    --------
    >>> data = {
    ...     "Total Thermal Conductivity (W/mK)": [3.0, None, 5.0, 2.5],
    ...     "Lattice Thermal Conductivity (W/mK)": [1.5, 2.0, None, 1.0],
    ...     "Electronic Thermal Conductivity (W/mK)": [1.5, 1.0, 2.5, None]
    ... }
    >>> df = pd.DataFrame(data)
    >>> thermal_conductivity_verification(df, verbose=1)
       Total Thermal Conductivity (W/mK)  Lattice Thermal Conductivity (W/mK)  Electronic Thermal Conductivity (W/mK) Thermal Conductivities Modification / Verification
    0                                3.0                                 1.5                                1.5                                      0.0
    1                                3.0                                 2.0                                1.0                               k_total added
    2                                5.0                                 2.5                                2.5                              k_lattice added
    3                                2.5                                 1.0                                1.5                           k_electronic added
    """
    if verbose not in [0, 1]:
        print("Verbose should be 0 or 1 ... setting to 0")

    new_df = df.copy()

    # Ensure the required columns are present in the dataframe
    if all(col in new_df.columns for col in [total_col, lattice_col, electronic_col]):
        # Define the index positions for the relevant columns
        total_idx = new_df.columns.get_loc(total_col)
        lattice_idx = new_df.columns.get_loc(lattice_col)
        electronic_idx = new_df.columns.get_loc(electronic_col)
        
        # Determine where to insert the new column
        insert_position = max(total_idx, lattice_idx, electronic_idx) + 1
        
        # Prepare a list to store the results of the verification
        modification_verification = []

        # Counters for the different types of modifications
        false_count = k_total_count = k_lattice_count = k_electronic_count = fully_filled_count = 0
        
        # Iterate through each row in the dataframe
        for idx, row in new_df.iterrows():
            total, lattice, electronic = row[total_col], row[lattice_col], row[electronic_col]
            
            filled_count = sum(pd.notna([total, lattice, electronic]))


            
            if filled_count == 0 or filled_count == 1:
                false_count += 1
                modification_verification.append("False")
            
            elif filled_count == 2:
                if pd.isna(total):
                    new_df.at[idx, total_col] = round(lattice + electronic, 8)
                    k_total_count += 1
                    modification_verification.append("k_total added")
                elif pd.isna(lattice):
                    new_df.at[idx, lattice_col] = round(total - electronic, 8)
                    k_lattice_count += 1
                    modification_verification.append("k_lattice added")
                else:
                    new_df.at[idx, electronic_col] = round(total - lattice, 8)
                    k_electronic_count += 1
                    modification_verification.append("k_electronic added")
            
            elif filled_count == 3:
                abs_perc_diff = abs(total - (lattice + electronic)) / total 
                modification_verification.append(abs_perc_diff)

                fully_filled_count += 1

                # if abs(total - (lattice + electronic)) <= threshold * total:
                #     verfied_correct_count += 1
                #     modification_verification.append("Verified Correct")
                # else:
                #     verfied_wrong_count += 1
                #     if verbose >= 2:
                #         print(f"Row {idx}: Total thermal conductivity does not match the sum of lattice and electronic")
                #     modification_verification.append("Verification Error")
        
        # Insert the new column into the dataframe
        new_df.insert(insert_position, "Thermal Conductivities Modification / Verification", modification_verification)

        if verbose >= 1:
            print(f"Total rows: {len(new_df)}")
            print(f"Unmodified rows: {false_count}")
            print(f"Total Thermal Conductivity added: {k_total_count}")
            print(f"Lattice Thermal Conductivity added: {k_lattice_count}")
            print(f"Electronic Thermal Conductivity added: {k_electronic_count}")
            print(f"Number of Fully Filled Entries: {fully_filled_count}")
    
    return new_df

def power_factor_verification(df, 
                              power_factor_col="Power Factor (µW/cmK²)", 
                              seebeck_col="Seebeck Coefficient (µV/K)", 
                              electric_con_col="Electrical Conductivity (S/cm)", verbose=0):
    """
    Verifies and fills missing values related to the power factor in a DataFrame based on the relationship:
    Power Factor = Electrical Conductivity * (Seebeck Coefficient)^2 * 10^(-6).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing power factor, Seebeck coefficient, and electrical conductivity data.
    power_factor_col : str, optional
        Column name for the power factor. Default is "Power Factor (µW/cmK²)".
    seebeck_col : str, optional
        Column name for the Seebeck coefficient. Default is "Seebeck Coefficient (µV/K)".
    electric_con_col : str, optional
        Column name for the electrical conductivity. Default is "Electrical Conductivity (S/cm)".
    verbose : int, optional
        Verbosity level:
        - 0 (default): Minimal output.
        - 1: Prints a summary of modifications and verification results.

    Returns
    -------
    pd.DataFrame
        A modified DataFrame with an additional column indicating modifications and verification results:
        - "False" if less than two values are present.
        - "Power Factor added" if the power factor was calculated.
        - "Seebeck Coefficient added. Remember to check the sign." if the Seebeck coefficient was calculated.
        - "Electrical Conductivity added" if the electrical conductivity was calculated.
        - Absolute percentage difference when all values are present.

    Notes
    -----
    - If any two values are present, the third value is calculated using the power factor formula.
    - When all three values are present, the function computes the absolute percentage difference to check
      for consistency.
    - If `verbose` is set to 1, a summary of the total rows, counts of each modification type, and fully filled
      entries is printed.
    - For Seebeck coefficient calculations, check the sign manually to ensure accuracy. Default assumption is that the Seebeck coefficient is positive.

    Examples
    --------
    >>> data = {
    ...     "Power Factor (µW/cmK²)": [10.0, None, 15.0, 5.0],
    ...     "Seebeck Coefficient (µV/K)": [200, 150, None, 100],
    ...     "Electrical Conductivity (S/cm)": [2, 1.5, 1.8, None]
    ... }
    >>> df = pd.DataFrame(data)
    >>> power_factor_verification(df, verbose=1)
       Power Factor (µW/cmK²)  Seebeck Coefficient (µV/K)  Electrical Conductivity (S/cm) Power Factor Modification / Verification
    0                    10.0                        200.0                             2.0                                      0.0
    1                    10.0                        150.0                             1.5                           Power Factor added
    2                    15.0                        204.1                             1.8               Seebeck Coefficient added. Remember to check the sign.
    3                     5.0                        100.0                             0.5                    Electrical Conductivity added
    """
    if verbose not in [0, 1]:
        print("Verbose should be 0 or 1 ... setting to 0")
        verbose = 0

    new_df = df.copy()

    # Ensure the required columns are present in the dataframe
    if all(col in new_df.columns for col in [power_factor_col, seebeck_col, electric_con_col]):
        # Define the index positions for the relevant columns
        power_factor_idx = new_df.columns.get_loc(power_factor_col)
        seebeck_idx = new_df.columns.get_loc(seebeck_col)
        electric_con_idx = new_df.columns.get_loc(electric_con_col)
        
        # Determine where to insert the new column
        insert_position = max(power_factor_idx, seebeck_idx, electric_con_idx) + 1
        
        # Prepare a list to store the results of the verification
        modification_verification = []

        # Counters for the different types of modifications
        false_count = pf_count = sc_count = ec_count = fully_filled_count = 0
        
        # Iterate through each row in the dataframe
        for idx, row in new_df.iterrows():
            power_factor, seebeck_coeff, electric_con = row[power_factor_col], row[seebeck_col], row[electric_con_col]
            
            filled_count = sum(pd.notna([power_factor, seebeck_coeff, electric_con]))

            if filled_count == 0 or filled_count == 1:
                false_count += 1
                modification_verification.append("False")
            
            elif filled_count == 2:
                if pd.isna(power_factor):
                    new_power_factor = round(electric_con * (seebeck_coeff ** 2) * 10**-6, 8)
                    new_df.at[idx, power_factor_col] = new_power_factor
                    pf_count += 1
                    modification_verification.append("Power Factor added")
                elif pd.isna(seebeck_coeff):
                    new_seebeck_coeff = round((power_factor / (electric_con * 10**-6))**0.5, 8)
                    new_df.at[idx, seebeck_col] = new_seebeck_coeff
                    sc_count += 1
                    modification_verification.append("Seebeck Coefficient added. Remember to check the sign.")
                else:
                    new_electric_con = round(power_factor / (seebeck_coeff ** 2 * 10**-6), 8)
                    new_df.at[idx, electric_con_col] = new_electric_con
                    ec_count += 1
                    modification_verification.append("Electrical Conductivity added")
            
            elif filled_count == 3:
                calculated_pf = electric_con * (seebeck_coeff ** 2) * 10**-6
                abs_perc_diff = abs(power_factor - calculated_pf) / power_factor
                modification_verification.append(abs_perc_diff)

                fully_filled_count += 1
        
        # Insert the new column into the dataframe
        new_df.insert(insert_position, "Power Factor Modification / Verification", modification_verification)

        if verbose >= 1:
            print(f"Total rows: {len(new_df)}")
            print(f"Unmodified rows: {false_count}")
            print(f"Power Factor added: {pf_count}")
            print(f"Seebeck Coefficient added: {sc_count}. Remember to check the sign.")
            print(f"Electrical Conductivity added: {ec_count}")
            print(f"Number of Fully Filled Entries: {fully_filled_count}")
    
    return new_df

def zt_verification(df, 
                    zt_col="zT", 
                    power_factor_col="Power Factor (µW/cmK²)", 
                    temp_col="Temperature (K)", 
                    k_total_col="Total Thermal Conductivity (W/mK)", 
                    verbose=0):
    """
    Verifies and fills missing values related to the dimensionless figure of merit (ZT) in a DataFrame 
    based on the relationship: ZT = Power Factor * Temperature / Total Thermal Conductivity * 10^(-4).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing ZT, power factor, temperature, and total thermal conductivity data.
    zt_col : str, optional
        Column name for the dimensionless figure of merit (ZT). Default is "ZT".
    power_factor_col : str, optional
        Column name for the power factor. Default is "Power Factor (µW/cmK²)".
    temp_col : str, optional
        Column name for the temperature. Default is "Temperature (K)".
    k_total_col : str, optional
        Column name for the total thermal conductivity. Default is "Total Thermal Conductivity (W/mK)".
    verbose : int, optional
        Verbosity level:
        - 0 (default): Minimal output.
        - 1: Prints a summary of modifications and verification results.

    Returns
    -------
    pd.DataFrame
        A modified DataFrame with an additional column indicating modifications and verification results:
        - "False" if fewer than three values are present.
        - "ZT added" if the ZT was calculated.
        - "Power Factor added" if the power factor was calculated.
        - "Temperature added" if the temperature was calculated.
        - "Total Thermal Conductivity added" if the total thermal conductivity was calculated.
        - Absolute percentage difference when all values are present.

    Notes
    -----
    - If three values are present, the fourth value is calculated using the ZT formula.
    - When all four values are present, the function computes the absolute percentage difference to check for consistency.
    - If `verbose` is set to 1, a summary of the total rows, counts of each modification type, and fully filled entries is printed.

    Examples
    --------
    >>> data = {
    ...     "ZT": [0.8, None, 1.2, 0.6],
    ...     "Power Factor (µW/cmK²)": [12.0, 10.0, None, 5.0],
    ...     "Temperature (K)": [300, 500, 400, None],
    ...     "Total Thermal Conductivity (W/mK)": [1.5, 2.0, 1.2, 0.8]
    ... }
    >>> df = pd.DataFrame(data)
    >>> zt_verification(df, verbose=1)
       ZT  Power Factor (µW/cmK²)  Temperature (K)  Total Thermal Conductivity (W/mK) ZT Modification / Verification
    0  0.8                   12.0              300                                1.5                            0.0
    1  0.8                   10.0              500                                2.0                      ZT added
    2  1.2                   12.0              400                                1.2               Power Factor added
    3  0.6                    5.0              750                                0.8             Temperature added
    """
    
    if verbose not in [0, 1]:
        print("Verbose should be 0 or 1 ... setting to 0")
        verbose = 0

    new_df = df.copy()

    # Ensure the required columns are present in the dataframe
    if all(col in new_df.columns for col in [zt_col, power_factor_col, temp_col, k_total_col]):
        # Define the index positions for the relevant columns
        zt_idx = new_df.columns.get_loc(zt_col)
        power_factor_idx = new_df.columns.get_loc(power_factor_col)
        temp_idx = new_df.columns.get_loc(temp_col)
        k_total_idx = new_df.columns.get_loc(k_total_col)
        
        # Determine where to insert the new column
        insert_position = max(zt_idx, power_factor_idx, temp_idx, k_total_idx) + 1
        
        # Prepare a list to store the results of the verification
        modification_verification = []

        # Counters for the different types of modifications
        false_count = zt_count = pf_count = temp_count = k_total_count = fully_filled_count = 0
        
        # Iterate through each row in the dataframe
        for idx, row in new_df.iterrows():
            zt, power_factor, temp, k_total = row[zt_col], row[power_factor_col], row[temp_col], row[k_total_col]
            
            filled_count = sum(pd.notna([zt, power_factor, temp, k_total]))

            if filled_count <= 2:
                false_count += 1
                modification_verification.append("False")
            
            elif filled_count == 3:
                if pd.isna(zt):
                    new_zt = round(power_factor * temp / k_total * 10**-4, 8)
                    new_df.at[idx, zt_col] = new_zt
                    zt_count += 1
                    modification_verification.append("ZT added")
                elif pd.isna(power_factor):
                    new_power_factor = round(zt * k_total / (temp * 10**-4), 8)
                    new_df.at[idx, power_factor_col] = new_power_factor
                    pf_count += 1
                    modification_verification.append("Power Factor added")
                elif pd.isna(temp):
                    new_temp = round(zt * k_total / (power_factor * 10**-4), 8)
                    new_df.at[idx, temp_col] = new_temp
                    temp_count += 1
                    modification_verification.append("Temperature added")
                else:
                    new_k_total = round(power_factor * temp * 10**-4 / zt, 8)
                    new_df.at[idx, k_total_col] = new_k_total
                    k_total_count += 1
                    modification_verification.append("Total Thermal Conductivity added")
            
            elif filled_count == 4:
                calculated_zt = power_factor * temp / k_total * 10**-4
                abs_perc_diff = abs(zt - calculated_zt) / zt
                modification_verification.append(abs_perc_diff)

                fully_filled_count += 1
        
        # Insert the new column into the dataframe
        new_df.insert(insert_position, "zT Modification / Verification", modification_verification)

        if verbose >= 1:
            print(f"Total rows: {len(new_df)}")
            print(f"Unmodified rows: {false_count}")
            print(f"zT added: {zt_count}")
            print(f"Power Factor added: {pf_count}")
            print(f"Temperature added: {temp_count}")
            print(f"Total Thermal Conductivity added: {k_total_count}")
            print(f"Number of Fully Filled Entries: {fully_filled_count}")
    
    return new_df

def filter_by_threshold(data, threshold, column=None, index_column=None):
    """
    Filters a DataFrame or Series based on a numeric threshold. Returns two DataFrames:
    one with numeric values below the threshold and all non-numeric values, and another
    with numeric values above or equal to the threshold.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        The input data to filter. If a DataFrame is provided, the `column` parameter
        should specify the column to filter.
    threshold : float
        The numeric threshold for filtering. Numeric values above or equal to this threshold
        will be filtered into one DataFrame, and numeric values below or non-numeric into another.
    column : str, optional
        The column in the DataFrame to apply the threshold filter to. If `data` is a Series,
        this parameter should be None.
    index_column : str, optional
        The column to display in print statements for identifying rows. This is useful
        for identifying entries by a specific column (e.g., an index column).

    Returns
    -------
    tuple of pd.DataFrame
        A tuple containing two DataFrames:
        - The first DataFrame includes numeric values below the threshold and all non-numeric values.
        - The second DataFrame includes numeric values above or equal to the threshold.

    Notes
    -----
    - Non-numeric entries are retained as-is in the below-threshold DataFrame.
    - If `data` is a DataFrame with one column and `column` is not specified, it automatically
      selects the only column available.
    """
    if column is not None and not isinstance(column, str):
        raise ValueError("Column must be a string or None")

    # Determine if input is DataFrame or Series, and select the column or series for filtering
    if isinstance(data, pd.DataFrame):
        selected_data = data[column] if column else data.iloc[:, 0]
    elif isinstance(data, pd.Series):
        selected_data = data
    else:
        raise ValueError("Input data must be a pandas DataFrame or Series")
    
    # Create masks for numeric values
    numeric_mask = pd.to_numeric(selected_data, errors='coerce').notna()
    
    # Apply threshold filtering only to numeric values
    below_threshold_mask = numeric_mask & (selected_data[numeric_mask].astype(float) < threshold)
    above_threshold_mask = numeric_mask & (selected_data[numeric_mask].astype(float) >= threshold)
    
    # Filter data based on masks, retaining non-numeric values in below-threshold DataFrame
    below_threshold_df = data[below_threshold_mask | ~numeric_mask]
    above_threshold_df = data[above_threshold_mask]

    if column:
        # Sort by the specified column
        above_threshold_df = above_threshold_df.sort_values(by=column, ascending=False)
    else:
        if not above_threshold_df.empty:
            first_col = above_threshold_df.columns[0]
            above_threshold_df = above_threshold_df.sort_values(by=first_col, key=pd.to_numeric, ascending=False, errors='coerce')


    # Print removed entries for above/equal threshold values
    removed_entries = above_threshold_df
    if not removed_entries.empty:
        print(f"Removed entries (values above or equal to {threshold}):")
        if index_column:
            print(removed_entries[[index_column, column]])
        else:
            print(removed_entries[[column]])

    return below_threshold_df, above_threshold_df


def check_doi_source(df, doi_col="Source Paper", init_dataset_col="Initial Dataset"):
    """
    Checks for possibly repeated datapoints by comparing whether the DOIs exist in more than one source.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing dataset
    doi_col : str, optional
        The column name for DOIs. Default is "Source Paper".
    init_dataset_col : str, optional
        The initial dataset column name for sources. Default is "Initial Dataset".

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the rows where DOIs are associated with more than one initial dataset.
        If all DOIs are from a single initial dataset, returns an empty DataFrame.

    """
    # Group by DOI and check if the number of unique sources is more than 1
    inconsistent_dois = df.groupby(doi_col)[init_dataset_col].nunique() > 1
    
    # Filter the DOIs with inconsistent sources
    inconsistent_doi_list = inconsistent_dois[inconsistent_dois].index
    
    if not inconsistent_doi_list.empty:
        # Return the rows corresponding to DOIs with different sources
        inconsistent_entries = df[df[doi_col].isin(inconsistent_doi_list)]
        print("Warning: Some DOIs have different sources.")
        return inconsistent_entries
    else:
        print("All DOIs come from one source.")
        return pd.DataFrame() # return an empty DataFrame if all DOIs have the same source

def check_seebeck_sign_consistency(df, doi_col="Source Paper", seebeck_col="Seebeck Coefficient (µV/K)"):
    """
    Checks for inconsistencies in the sign of Seebeck coefficients within each DOI group and returns rows with such inconsistencies.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing DOIs and Seebeck coefficient values.
    doi_col : str, optional
        The column name for DOIs. Default is "Source Paper".
    seebeck_col : str, optional
        The column name for the Seebeck coefficient values. Default is "Seebeck Coefficient (µV/K)".

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the rows where DOIs have inconsistent Seebeck coefficient signs.
        If all Seebeck coefficients within a DOI group are consistent in sign, returns an empty DataFrame.

    Notes
    -----
    - The function groups the data by DOI and checks for sign inconsistencies within each group.
    - If any DOI has both positive and negative Seebeck coefficient values, it is flagged as inconsistent.
    - The function ignores NaN values, and consistency is checked only if there are at least two non-zero values in a group.

    Examples
    --------
    >>> data = {
    ...     "DOI": ["10.1000/xyz123", "10.1000/xyz123", "10.1000/abc456", "10.1000/abc456"],
    ...     "Seebeck Coefficient (µV/K)": [100, -50, 200, 150]
    ... }
    >>> df = pd.DataFrame(data)
    >>> check_seebeck_sign_consistency(df)
           DOI  Seebeck Coefficient (µV/K)
    0  10.1000/xyz123                    100
    1  10.1000/xyz123                    -50

    """
    # Function to check if signs are inconsistent within a group
    def has_inconsistent_signs(group):
        non_nan_values = group.dropna()
        # Check if there are at least two non-zero values to compare
        if len(non_nan_values) > 1:
            # Check if all values are positive or all are negative
            if (non_nan_values > 0).all() or (non_nan_values < 0).all():
                return False  # All signs are consistent
            return True  # Signs are inconsistent
        return False  # No inconsistency if there's only one value or all are NaN/zero
    
    # Apply the function to each group of DOI
    inconsistent_dois = df.groupby(doi_col)[seebeck_col].apply(has_inconsistent_signs)
    
    # Filter the rows where the DOI has inconsistent signs
    inconsistent_dois = inconsistent_dois[inconsistent_dois]  # Select DOIs where inconsistency is True
    
    # Return the rows corresponding to these DOIs
    return df[df[doi_col].isin(inconsistent_dois.index)]


def filter_zero_entries(df, column_name, index_column=None):
    """
    Filters out rows with zero values in a specified column, while keeping blanks (NaN) intact.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    column_name : str
        The name of the column to check for zero values.
    index_column : str, optional
        The name of an additional column to display for identifying removed entries. If specified,
        the removed entries are printed along with this column for easier identification.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with rows containing zero values in the specified column removed. Rows with NaN
        values in the specified column are retained.

    Notes
    -----
    - The function prints the removed entries with their values and optional index information.
    - If `index_column` is provided, the output will include this column for each removed entry. Otherwise,
      only the values in `column_name` are shown.

    Examples
    --------
    >>> data = {
    ...     "Index": ["A", "B", "C", "D"],
    ...     "Value": [1, 0, None, 3]
    ... }
    >>> df = pd.DataFrame(data)
    >>> filter_zero_entries(df, column_name="Value", index_column="Index")
    Removed entries:
      Index  Value
    1     B    0.0
       Index  Value
    0     A    1.0
    2     C    NaN
    3     D    3.0

    """
    # Create a mask to filter out zero values, keeping blanks (NaN) intact
    mask = (df[column_name] != 0) | df[column_name].isna()
    
    # Identify the rows to be removed (those with exact zero values)
    removed_entries = df[~mask]
    
    # Print removed entries and their indices
    if not removed_entries.empty:
        print("Removed entries:")
        if index_column:
            print(removed_entries[[index_column, column_name]])
        else:
            print(removed_entries[[column_name]])
    
    # Return the filtered dataframe
    filtered_df = df[mask].copy()
    
    return filtered_df