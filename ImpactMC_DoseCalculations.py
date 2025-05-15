import numpy as np
import os
import glob
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
import pydicom
from pydicom.dataset import Dataset, FileDataset
import pydicom.uid
import spekpy as sp
import re
import matplotlib.cm as cm
import ImpactMC_CBCTinput as IMC
import xml.etree.ElementTree as ET
from collections import defaultdict
from scipy.stats import ttest_rel

''' 
This code can be used to calculate the mean absorbed organ doses, effective doses and CTDI values
from ImpactMC (Erlangen, Germany) Raw-Dose output. It uses information from the 
ImpactMC_CBCTinput.py code, so all parameters there need to correspond to the output generated 
with ImpactMC.

F. van Wolferen
'''

''' Loading and preprocessing of the RAW Dose data '''
#########################################################################
# Path to your raw file and mifx XML file
if IMC.child_phantom == False:
    raw_file_path = f'H:\ImpactMC_CBCT\Dose_RawData_{IMC.sex_of_phantom}\Binary_file.raw'
    mifx_file_path = f'H:\ImpactMC_CBCT\Dose_RawData_{IMC.sex_of_phantom}\Binary_file.mifx'
    if IMC.CTDI_phantom == True:
        raw_file_path = f'H:\ImpactMC_CBCT\Dose_RawData_CTDI\Binary_file_32cm_6.raw'
        mifx_file_path = f'H:\ImpactMC_CBCT\Dose_RawData_CTDI\Binary_file_32cm_6.mifx'
elif IMC.child_phantom == True :
    raw_file_path = f'H:\ImpactMC_CBCT\Dose_RawData_{IMC.sex_of_phantom}\Child\Binary_file_F10_CP_check2.raw'
    mifx_file_path = f'H:\ImpactMC_CBCT\Dose_RawData_{IMC.sex_of_phantom}\Child\Binary_file_F10_CP_check2.mifx'
    if IMC.CTDI_phantom == True:
        raw_file_path = f'H:\ImpactMC_CBCT\Dose_RawData_CTDI\Binary_file_32cm_6.raw'
        mifx_file_path = f'H:\ImpactMC_CBCT\Dose_RawData_CTDI\Binary_file_32cm_6.mifx'

# Read and extract the dimensions from the .mifx XML file
def read_mifx(mifx_path):
    tree = ET.parse(mifx_path)
    root = tree.getroot()
    
    # Extract dimensions from XML tags
    width = int(root.find('NumberOfXVoxels').text)  # Get width from NumberOfXVoxels
    height = int(root.find('NumberOfYVoxels').text)  # Get height from NumberOfYVoxels
    depth = int(root.find('NumberOfZVoxels').text)  # Get depth from NumberOfZVoxels

    return width, height, depth

# Get the volume dimensions
width, height, depth = read_mifx(mifx_file_path)
#print(f"Width: {width}, Height: {height}, Depth: {depth}")

# Read the raw binary data as 32-bit floating point (little-endian byte order)
volume = np.fromfile(raw_file_path, dtype=np.float32)

# Check the total size of the raw data
#print(f"Raw data size: {volume.size}")

# Ensure the size matches with the expected dimensions (width * height * depth)
expected_size = width * height * depth
if volume.size == expected_size:
    volume = volume.reshape((depth, height, width))  # Reshape to 3D volume
    print(f"Reshaped data to volume with shape: {volume.shape}")
else:
    print(f"Mismatch in data size! Expected size: {expected_size}, got: {volume.size}")

''' Load and store are weighting factor data '''
#########################################################################

weighting_factors = [['Bladder', 1, 0.04],['Brain', 2, 0.01],['Breast', 3, 0.12],['Colon', 4, 0.12],['Oesophagus', 5, 0.04],
                     ['Gonads', 6, 0.08],['Liver', 7, 0.04],['Lung', 8, 0.12],['Red bone marrow', 9, 0.12],['Salivary glands', 10, 0.01],
                     ['Skeleton', 11, 0.01],['Skin', 12, 0.01],['Stomach', 13, 0.12],['Thyroid', 14, 0.04]]
remainder_tissues = [['Adrenals',15], ['Extrathoraxic tissue', 16], ['Gall bladder',17], ['Heart',18], ['Kidney',19], ['Lymphatic nodes',20], 
                     ['Muscle',21],['Oral mucosa',22], ['Pancreas',23], ['Prostate',24], ['Small intestine',25], ['Spleen',26], ['Thymus',27], 
                     ['Uterus',28]]
remainder_factor = 0.12
remainder_tissues_TLD = 0.009
remainder_tissues_names = [43, 49, 45, 33, 35, 47, 29, 29, 31, 46, 37, 39, 45, 46]
remainder_tissues_names = [29, 31, 33, 35, 37, 39, 43, 45, 46, 47, 49]

# Load the text files into a DataFrame
if IMC.CTDI_phantom == False:
    if IMC.child_phantom == False:
        if IMC.sex_of_phantom == 'Female':
            spongiosa_organ_weighting_file = pd.read_csv(r'H:\ImpactMC_CBCT\AF_weighting_spongiosa.txt', delimiter=',')  # Use delimiter=',' for comma-separated values
        else:
            spongiosa_organ_weighting_file = pd.read_csv(r'H:\ImpactMC_CBCT\AM_weighting_spongiosa.txt', delimiter=',')

        with open(Path(f"H:\ImpactMC_CBCT\Table 3.2.txt"), "r") as file:
        # Read all lines, but only take the first two
            spongiosa_file = file.readlines()

        spongiosa_data = []
        spongiosa_file = spongiosa_file[5:]
        spongiosa_file = spongiosa_file[:-2]
        
        for line in spongiosa_file:
            parts = line.strip().split()  # Split by spaces

            if not parts:  # Skip empty lines
                continue

            # Extract the first item as an integer
            first_integer = int(parts[0])  # The first value is always an integer

            text_part = []
            numeric_cols = []

            # Identify where the first float occurs (rest of the numbers)
            for i in range(1, len(parts)):  # Start from index 1 since 0 is the integer
                if re.match(r'^-?\d+\.\d+$', parts[i]):  # First float detected
                    text_part = " ".join(parts[1:i])  # Everything before the float is text
                    numeric_cols = list(map(float, parts[i:]))  # Convert remaining to float
                    break
            else:
                # If no floats exist, treat everything after the first integer as text
                text_part = " ".join(parts[1:])
                numeric_cols = []

            # Store extracted values
            spongiosa_data.append([first_integer, text_part] + numeric_cols)

        if IMC.sex_of_phantom == 'Male':
            spongiosa_values_RBM = [float(row[3]) for row in spongiosa_data]
            spongiosa_values_TM50 = [float(row[5]) for row in spongiosa_data]
        else: 
            spongiosa_values_RBM = [float(row[7]) for row in spongiosa_data]
            spongiosa_values_TM50 = [float(row[9]) for row in spongiosa_data]
    
    else:
        with open(IMC.folder_path5, "r") as file:
        # Read all lines, but only take the first two
            spongiosa_file = file.readlines()
        
        if IMC.sex_of_phantom == 'Female':
            spongiosa_organ_weighting_file = pd.read_csv(r'H:\ImpactMC_CBCT\CF_weighting_spongiosa.txt', delimiter=',')
        else:
            spongiosa_organ_weighting_file = pd.read_csv(r'H:\ImpactMC_CBCT\CM_weighting_spongiosa.txt', delimiter=',')

        spongiosa_data = []
        spongiosa_file = spongiosa_file[:-2]
        spongiosa_file = spongiosa_file[2:]
        for line in spongiosa_file:
            parts = line.strip().split()  # Split by spaces

            if not parts:  # Skip empty lines
                continue

            # Identify where the numbers start
            for i in range(len(parts)):
                if parts[i].replace('.', '', 1).isdigit():  # First numeric value found
                    text_part = " ".join(parts[:i])  # Everything before first number is text
                    numeric_cols = list(map(float, parts[i:]))  # Convert remaining to float
                    break
            else:
                # If no numeric values found, treat the entire line as text
                text_part = " ".join(parts)
                numeric_cols = []

            # Store extracted values
            spongiosa_data.append([text_part] + numeric_cols)

        spongiosa_values_RBM = [float(row[5]) for row in spongiosa_data[:-1]]
        spongiosa_values_TM50 = [float(row[7]) for row in spongiosa_data[:-1]]


    # Extract the first and third columns (organ number and weighting number)
    organ_numbers = spongiosa_organ_weighting_file.iloc[:, 0].values  # First column (organ number)
    weighting_numbers = spongiosa_organ_weighting_file.iloc[:, 2].values.astype(float)  # Ensure weighting numbers are floats

    if IMC.child_phantom == False:
        spongiosa_numbers_perorgan = spongiosa_organ_weighting_file.iloc[:, 3].values 
        organ_numbers_perfraction = [float(row[0]) for row in spongiosa_data]

        combined_array2 = np.column_stack((organ_numbers, spongiosa_numbers_perorgan)).astype(float)
        combined_array3 = np.column_stack((organ_numbers_perfraction, spongiosa_values_RBM)).astype(float)
        combined_array4 = np.column_stack((organ_numbers_perfraction, spongiosa_values_TM50)).astype(float)
    else:
        organ_numbers_perfraction1 = [8,13,16,21,15,17,18,19,20,14]
        spongiosa_values_RBM1 = spongiosa_values_RBM[:10]

        organ_numbers_perfraction2 = [3,4]
        spongiosa_values_RBM2 = np.append(spongiosa_values_RBM1, [spongiosa_values_RBM[11], spongiosa_values_RBM[12]])

        organ_numbers_perfraction3 = [5,6]
        spongiosa_values_RBM3 = np.append(spongiosa_values_RBM2, [(spongiosa_values_RBM[13] + spongiosa_values_RBM[14]), spongiosa_values_RBM[15]])

        organ_numbers_perfraction4 = [9]
        spongiosa_values_RBM4 = np.append(spongiosa_values_RBM3, [spongiosa_values_RBM[16]])

        organ_numbers_perfraction5 = [10]
        spongiosa_values_RBM5 = np.append(spongiosa_values_RBM4, [spongiosa_values_RBM[17]])

        organ_numbers_perfraction6 = [11]
        spongiosa_values_RBM6 = np.append(spongiosa_values_RBM5, [(spongiosa_values_RBM[18] + spongiosa_values_RBM[19] + spongiosa_values_RBM[20])])

        organ_numbers_perfraction7 = [12]
        spongiosa_values_RBM7 = np.append(spongiosa_values_RBM6, [spongiosa_values_RBM[21]])

        ########################

        organ_numbers_perfraction1 = [8,13,16,21,15,17,18,19,20,14]
        spongiosa_values_TM1 = spongiosa_values_TM50[:10]

        organ_numbers_perfraction2 = [3,4]
        spongiosa_values_TM2 = np.append(spongiosa_values_TM1, [spongiosa_values_TM50[11], spongiosa_values_TM50[12]])

        organ_numbers_perfraction3 = [5]
        spongiosa_values_TM3 = np.append(spongiosa_values_TM2, [(spongiosa_values_TM50[13] + spongiosa_values_TM50[14])])

        organ_numbers_perfraction4 = [6]
        spongiosa_values_TM4 = np.append(spongiosa_values_TM3, [spongiosa_values_TM50[15]])

        organ_numbers_perfraction5 = [9,10]
        spongiosa_values_TM5 = np.append(spongiosa_values_TM4, [spongiosa_values_TM50[16], spongiosa_values_TM50[17]])

        organ_numbers_perfraction6 = [11]
        spongiosa_values_TM6 = np.append(spongiosa_values_TM5, [(spongiosa_values_TM50[18] + spongiosa_values_TM50[19] + spongiosa_values_TM50[20])])

        organ_numbers_perfraction7 = [12]
        spongiosa_values_TM7 = np.append(spongiosa_values_TM6, [spongiosa_values_TM50[21]])

        total_RBM = float(spongiosa_data[-1][5])
        total_TM50 = float(spongiosa_data[-1][7])

        spongiosa_values_RBM_final = [value / total_RBM for value in spongiosa_values_RBM7]
        #print(spongiosa_values_RBM_final)
        spongiosa_values_TM50_final = [value / total_TM50 for value in spongiosa_values_TM7]
        #print(spongiosa_values_TM50_final)

        spongiosa_numbers_perorgan = spongiosa_organ_weighting_file.iloc[:, 3].values
        combined_array2 = np.column_stack((organ_numbers, spongiosa_numbers_perorgan)).astype(float)

        organ_numbers_perfraction_final = np.append(organ_numbers_perfraction1, np.concatenate([organ_numbers_perfraction2, organ_numbers_perfraction3, 
                        organ_numbers_perfraction4, organ_numbers_perfraction5, 
                        organ_numbers_perfraction6, organ_numbers_perfraction7]))

        combined_array3 = np.column_stack((organ_numbers_perfraction_final, spongiosa_values_RBM_final)).astype(float)
        combined_array4 = np.column_stack((organ_numbers_perfraction_final, spongiosa_values_TM50_final)).astype(float)


    # Combine them into a 2D array (organ number, weighting number)
    combined_array = np.column_stack((organ_numbers, weighting_numbers)).astype(float)

    # Count how many rows in combined_array have a weighting number in the range 1 to 14
    count = np.sum((combined_array[:, 1] >= 1) & (combined_array[:, 1] <= 14))

    # Print the original combined array (for reference)
    #print("Original Combined Array:")
    #print(combined_array)

''' Compute the mean absorbed organ dose '''
#########################################################################

if IMC.CTDI_phantom == False:
    tissue_volume = IMC.volume
    dose_volume = volume

    # Ensure dose_volume is flattened to match organ_volume shape
    dose_volume = np.ravel(dose_volume)
    tissue_volume = np.ravel(tissue_volume)

    # Convert `combined_array` into a dictionary for quick lookup
    weighting_number_dict = {int(row[0]): float(row[1]) for row in combined_array}
    organ_number_dict = {int(row[1]): float(row[0]) for row in combined_array}
    #print(weighting_number_dict)

    tissue_to_organ_number_dict = {int(row[1]): float(row[0]) for row in IMC.organ_number_medium_number}
    # Vectorized replacement using a fallback for unmapped values
    replace_func = np.vectorize(lambda x: tissue_to_organ_number_dict.get(int(x), 0))  # Default to 0 if not found

    # Apply mapping
    organ_volume = replace_func(tissue_volume)

    # Convert `combined_array2` into a dictionary for quick lookup
    spongiosa_number_dict = {int(row[0]): float(row[1]) for row in combined_array2} # spong to organ
    # Use defaultdict with lists to store multiple values for each key
    organ_to_spong_number_dict = defaultdict(list)

    for row in combined_array2:
        organ_to_spong_number_dict[int(row[0])].append(int(row[1]))  # Append instead of overwrite

    # Convert defaultdict to a regular dict (optional)
    organ_to_spong_number_dict = dict(organ_to_spong_number_dict)

    # Convert `combined_array3` into a dictionary for quick lookup
    spongiosa_RBM_dict = {int(row[0]): float(row[1]) for row in combined_array3}

    # Convert `combined_array4` into a dictionary for quick lookup
    spongiosa_TM50_dict = {int(row[0]): float(row[1]) for row in combined_array4}

    # Replace organ numbers in organ_volume with corresponding weighting numbers
    organ_volume_new = np.array([weighting_number_dict.get(int(org), 0) for org in organ_volume])  # Default to 0 if not found

    # Make an empty Effective Dose array for each organ
    mao_doses_tissues = np.zeros(62)
    mao_doses_organs = np.zeros(142)
    # Create a counter array to track how many times a dose is added to each organ
    mao_doses_count_tissues = np.zeros(62)
    mao_doses_count_organs = np.zeros(142)

    # Create a lookup dictionary mapping p3 to w_T
    w_T_dict = {row[1]: row[2] for row in weighting_factors}

    # Loop through each corresponding pixel
    for p1, p2, p3 in zip(dose_volume, tissue_volume, organ_volume):
        D_pix = p1 
        
        # Store D_pix in the corresponding organ index (adjust for zero-based indexing)
        tissue_index = int(p2)
        mao_doses_tissues[tissue_index] += D_pix
        mao_doses_count_tissues[tissue_index] += 1  # Increment the count for this organ

        organ_index = int(p3)
        mao_doses_organs[organ_index] += D_pix
        mao_doses_count_organs[organ_index] += 1  # Increment the count for this organ

    # After the loop, calculate the average for each organ
    for i in range(len(mao_doses_tissues)):
        if mao_doses_count_tissues[i] > 0:
            mao_doses_tissues[i] /= mao_doses_count_tissues[i]  # Calculate the average dose for each organ

    for dose in mao_doses_tissues:
        print(f"{dose:.8f}")

    # After the loop, calculate the average for each organ
    for i in range(len(mao_doses_organs)):
        if mao_doses_count_organs[i] > 0:
            mao_doses_organs[i] /= mao_doses_count_organs[i]

    #print(mao_doses)
    #print(len(mao_doses))

    # Print effective_doses_final with corresponding organ numbers
    for i, dose in enumerate(mao_doses_tissues):
        print(f"Tissue {i:03d} - Mean Absorbed Dose = {dose:.8f} mGy")


''' Calculate the effective energy of the spectrum '''
#########################################################################

if IMC.CTDI_phantom == False:
    # Getting the effective energy
    E_eff = IMC.E_eff2


    print("Mean Energy (E_mean):", E_eff, "keV")

''' Get the MEAC ratios for RBM '''
#########################################################################

if IMC.CTDI_phantom == False:
    if IMC.child_phantom == False:
        # Load the file into a pandas DataFrame, assuming it's space-separated
        df = pd.read_csv(f'H:\ImpactMC_CBCT\ICRP {IMC.sex_of_phantom}\MEAC Ratios {IMC.sex_of_phantom}.txt', sep=' ', header=None)

        # Initialize an empty list to store rows with decimal values
        rows_with_floats = []

        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():
            # Convert the row to a string (join the columns into a single string)
            row_str = ' '.join(map(str, row.values))  # Ensure all values are treated as strings
            
            # Check if the row string contains a float (decimal number)
            if re.search(r'\d+\.\d+', row_str):  # Regex to find a decimal number
                rows_with_floats.append(row_str.strip())  # Add the row to the list

        # Print the rows that contain decimal numbers
        #print(rows_with_floats)

        # Now split the rows into columns (separated by spaces)
        split_rows = [row.split() for row in rows_with_floats]
        #print(split_rows)

        # Select specific columns: 0, 1, 3, 5, 7, 9, 11, and 13
        RBM_columns = [0, 1, 3, 5, 7, 9, 11, 13]
        RBM_columns2 = [1, 3, 5, 7, 9, 11, 13]
        RBM_columns3 = [1, 3, 5, 7]

        # Select specific columns: 
        TM_columns = [0, 2, 4, 6, 8, 10, 12, 14]
        TM_columns2 = [2, 4, 6, 8, 10, 12, 14]
        TM_columns3 = [2, 4, 6, 8]

        # Extract the first 11 rows and only the selected columns
        array1 = [[row[i] for i in RBM_columns] for row in split_rows[:11]]
        array2 = [[row[i] for i in RBM_columns2] for row in split_rows[11:22]]
        array3 = [[row[i] for i in RBM_columns2] for row in split_rows[22:33]]
        array4 = [[row[i] for i in RBM_columns3] for row in split_rows[33:44]]

        # Extract the first 11 rows and only the selected columns
        array12 = [[row[i] for i in TM_columns] for row in split_rows[:11]]
        array22 = [[row[i] for i in TM_columns2] for row in split_rows[11:22]]
        array32 = [[row[i] for i in TM_columns2] for row in split_rows[22:33]]
        array42 = [[row[i] for i in TM_columns3] for row in split_rows[33:44]]

        # Combine the arrays
        result_array_RBM = [row1 + row2 + row3 + row4 for row1, row2, row3, row4 in zip(array1, array2, array3, array4)]
        #print(result_array)

        # Combine the arrays
        result_array_TM = [row1 + row2 + row3 + row4 for row1, row2, row3, row4 in zip(array12, array22, array32, array42)]
        #print(result_array2)

        # Find the values that correspond closest to the effective energy
        def find_closest_row(result_array, E_eff):
            return min(result_array, key=lambda row: abs(float(row[0]) - (E_eff/1000)))

        # Find the closest row
        meac_ratios = find_closest_row(result_array_RBM, E_eff)
        meac_ratios2 = find_closest_row(result_array_TM, E_eff)
        #print(meac_ratios)
        #print(meac_ratios2)

        spong_icrp_file = pd.read_csv(r'H:\ImpactMC_CBCT\AFM_spongiosa_ICRP.txt', delimiter=',')
        spongiosa_num = spong_icrp_file.iloc[:, 0].values  # First column (organ number)
        icrp_num = spong_icrp_file.iloc[:, 1].values  # First column (organ number)
        arr = np.arange(3, 29)
        combined_icrpspong = np.column_stack((spongiosa_num, icrp_num)).astype(float)
        spongiosa_icrp_dict = {int(row[0]): int(row[1]) for row in combined_icrpspong}

        combined_meac = np.column_stack((meac_ratios, arr))
        combined_meac2 = np.column_stack((meac_ratios2, arr))

        meac_spong_dict = {int(row[1]): str(row[0]) for row in combined_meac}
        meac_spong_dict2 = {int(row[1]): str(row[0]) for row in combined_meac2}

    else:
        folder_path = Path(r"H:\ImpactMC_CBCT\NIST_data")

        # Dictionary to store attenuation coefficient arrays
        mu_all = {}
        element_name = []

        for file in folder_path.glob("*.txt"):
            name = file.stem  # Extract filename (e.g., "C" from "C.txt")
            element_name.append(name)

            # Load data as NumPy array (skip first two rows if needed)
            mu_data = np.loadtxt(file, skiprows=2)  
            
            # Store the NumPy array in dictionary under its element name
            mu_all[name] = mu_data

        with open(IMC.folder_path2, "r") as file:
            # Read all lines, but only take the first two
            media_data_all = file.readlines()  

        media_data = media_data_all[3:]
        media_numbers = [line.strip().split()[0] for line in media_data]
        #print(media_numbers)

        element_data = media_data_all[:2]
        element_numbers = element_data[0].strip().split()

        # Extract only the element symbol (remove numbers)
        element_names = [re.match(r"[A-Za-z]+", name).group() for name in element_data[0].strip().split()[2:-2]]

        # Sort element_name to match the order of element_names
        sorted_mu_all = {key: mu_all[key] for key in element_names if key in mu_all}

        # Print results
        #print("Sorted dictionary:")
        #for key, value in sorted_mu_all.items():
        #   print(f"{key}: {value}")

        processed_media = []

        for line in media_data:
            parts = line.strip().split()  # Split by default whitespace
            first_col = parts[0]  # First column (numeric)
            
            # Identify where the second column (text) ends
            for i in range(1, len(parts)):
                if parts[i].replace('.', '', 1).isdigit():  # First numeric column after text
                    second_col = " ".join(parts[1:i])  # Join everything between col 1 and first number
                    remaining_cols = parts[i:]  # Everything after
                    break
            
            # Store the row as a list
            processed_media.append([first_col, second_col] + remaining_cols)

        # Print first 5 rows to check structure
        #for i, row in enumerate(processed_media[:5]):
        #   print(f"Row {i+1}: {row}")

        # Initialize an empty list to store the second elements
        mu_E_list = []

        # Loop through each element in mu_all
        for names, array in sorted_mu_all.items():
            # Compute absolute differences between the first column and E_eff
            differences = np.abs(array[:, 0] - (E_eff/1000))  
            
            # Find the index of the row where the first element is closest to E_eff
            closest_row_index = np.argmin(differences)
            
            # Get the second element (attenuation coefficient) from the closest row
            mu_E_list.append(array[closest_row_index])

        # Convert to a NumPy array
        mu_E = np.array(mu_E_list)
        #print(mu_E)

        # Initialize an empty list to store all rows of fractions
        all_fractions = []

        # Getting the fractions for each medium
        for row in processed_media:
            raw_fractions = row[2:]  # Slice to get elements after the first two
            #print(raw_fractions)

            raw_fractions = np.array(raw_fractions, dtype=str)  # Ensure everything is string
            raw_fractions = np.char.strip(raw_fractions)  # Remove hidden spaces
            fractions = raw_fractions.astype(float)  # Convert to float

            # Append the current row of fractions to the all_fractions list
            all_fractions.append(fractions)

        # Convert the list of fractions into a NumPy array for easier handling
        all_fractions_array = np.array(all_fractions)
        #print(all_fractions_array)

        with open(IMC.folder_path3, "r") as file:
            # Read all lines, but only take the first two
            organ_data_all = file.readlines()[4:]  # Skip first 3 header rows

        # Getting the densities for each medium
        rho_media_raw = [line.strip().split()[-2:] for line in organ_data_all]
        #print(rho_media_raw)  # Check extracted data

        organ_numbers = [line.strip().split()[0] for line in organ_data_all]
        #print(organ_numbers)

        organ_number_vs_medium = [line.strip().split()[-2] for line in organ_data_all]
        #print(organ_number_vs_medium)

        # Extract values, skipping lines where organ_number_vs_medium is 'n/a'
        filtered_data = [
                (num, medium) for num, medium in zip(organ_numbers, organ_number_vs_medium) if medium.lower() != 'n/a'
        ]

        # Convert to NumPy array
        combined_arrayyy = np.array(filtered_data, dtype=float)
        organ_medium_dict = {int(row[0]): int(row[1]) for row in combined_arrayyy}

        rho_media = {}

        for medium, density in rho_media_raw:
            if medium.lower() != 'n/a' and density.lower() != 'n/a':  # Skip 'n/a' values
                rho_media[medium] = float(density)  # Convert density to float, keep medium as string
        
        # Sort medium numbers (strings) and retrieve the densities in the correct order
        sorted_rho_media = [rho_media[medium] for medium in sorted(rho_media.keys(), key=lambda x: float(x))]
        rho_spong = sorted_rho_media[2:25]
        #print(rho_spong)

        def mu_over_rho_medium(fraction, mu_E):
            return sum((fraction/100)*(mu_E))

        # Initialize lists to store the results for each medium
        mu_over_rho_medium_results = []

        # Loop through each medium in processed_media
        for i, medium in enumerate(processed_media):
            # Extract the fraction row for the current medium
            fractions = all_fractions_array[i]
            
            # Select the correct mu_E and rho_elements based on element index
            mu_E_for_medium = np.array([mu_E[j] for j in range(len(fractions[:-1]))])  
            #rho_elements_for_medium = np.array([rho_elements[j] for j in range(len(fractions))])
            #print(mu_E_for_medium)

            # Calculate mu_over_rho_medium for the current medium
            muoverrhomedium = mu_over_rho_medium(fractions[:-1], mu_E_for_medium[:, 1])

            # Optionally, print the result for each medium
            #print(f"mu_over_rho for medium {medium[1]}: {muoverrhomedium}")
            
            mu_over_rho_medium_results.append(muoverrhomedium)

        #print(mu_over_rho_medium_results)

        # Load the text files into a DataFrame
        weighting_file = spongiosa_organ_weighting_file

        # Extract the first and third columns (organ number and weighting number)
        organ_numbers = weighting_file.iloc[:, 0].values  # First column (organ number)
        weighting_numbers = weighting_file.iloc[:, 2].values.astype(float)  # Ensure weighting numbers are floats

        # Combine them into a 2D array (organ number, weighting number)
        combined_array5 = np.column_stack((organ_numbers, weighting_numbers)).astype(float)

        # Convert `combined_array` into a dictionary for quick lookup
        weighting_number_dict = {int(row[0]): float(row[1]) for row in combined_array5}
        organ_number_dict = {int(row[1]): float(row[0]) for row in combined_array5}

        mu_over_rho_spong = mu_over_rho_medium_results[:23]

        #print(mu_over_rho_spong)

        # Load the text files into a DataFrame
        with open(r'H:\ImpactMC_CBCT\AFM_spongiosa_compositions.txt', "r") as file:
            # Read all lines, but only take the first two
            spong_data_all = file.readlines()[4:]  # Skip first 3 header rows

        #print(spong_data_all)

        spong_data = [line.strip().split() for line in spong_data_all]
        if IMC.sex_of_phantom == 'Female':
            rbm_value = spong_data[1]
            rbm_value = rbm_value[1:]
            rbmvalue = []
            for i in rbm_value:
                value = float(i)
                rbmvalue.append(value)

            #print(rbmvalue)
            tm_value = spong_data[5]
            tm_value = tm_value[1:]
            tmvalue = []
            for i in tm_value:
                value = float(i)
                tmvalue.append(value)

            #print(tmvalue)
        else:
            rbm_value = spong_data[0]
            rbm_value = rbm_value[1:]
            rbmvalue = []
            for i in rbm_value:
                value = float(i)
                rbmvalue.append(value)

            #print(rbmvalue)

            tm_value = spong_data[4]
            tm_value = tm_value[1:]
            tmvalue = []
            for i in tm_value:
                value = float(i)
                tmvalue.append(value)

            #print(tmvalue)

        # Calculate mu_over_rho_medium for the current medium
        muoverrhorbm = mu_over_rho_medium(np.array(rbmvalue, dtype=float), mu_E[:, 1])
        #print(muoverrhorbm)
        muoverrhotm = mu_over_rho_medium(np.array(tmvalue, dtype=float), mu_E[:, 1])
        #print(muoverrhotm)

        meac_ratios_RBM = muoverrhorbm / mu_over_rho_spong
        meac_ratios_TM = muoverrhotm / mu_over_rho_spong

        # Define excluded indices based on age
        if IMC.age_of_child == '10':
            excluded_indices = {3, 9}
        elif IMC.age_of_child == '15':
            excluded_indices = {2, 3, 8, 9}
        else:
            excluded_indices = set()

        # Create arrays with 'NA' for excluded indices
        meac_ratios = np.array([
            0 if i in excluded_indices else val 
            for i, val in enumerate(meac_ratios_RBM)
        ], dtype=object)  # Use dtype=object to store mixed types (numbers & 'NA')

        meac_ratios2 = meac_ratios_TM

        #medium_numbers_meacr = np.arange(3, 26, 1)
        
        #combinated_array6_1 = np.column_stack((medium_numbers_meacr, meac_ratios)).astype(float)
        #combinated_array6_2 = np.column_stack((medium_numbers_meacr, meac_ratios2)).astype(float)
        
        #medium_to_meacr_dict = {float(row[0]): float(row[1]) for row in combinated_array6_1}
        #medium_to_meacr2_dict = {float(row[0]): float(row[1]) for row in combinated_array6_2}

        spong_icrp_file = pd.read_csv(r'H:\ImpactMC_CBCT\CFM_spongiosa_ICRP.txt', delimiter=',')
        spongiosa_num = spong_icrp_file.iloc[:, 0].values  # First column (organ number)
        icrp_num = spong_icrp_file.iloc[:, 1].values  # First column (organ number)
        arr = np.arange(1, 24)
        combined_icrpspong = np.column_stack((spongiosa_num, icrp_num)).astype(float)
        spongiosa_icrp_dict = {int(row[0]): int(row[1]) for row in combined_icrpspong}

        combined_meac = np.column_stack((meac_ratios, arr))
        combined_meac2 = np.column_stack((meac_ratios2, arr))

        meac_spong_dict = {int(row[1]): str(row[0]) for row in combined_meac}
        meac_spong_dict2 = {int(row[1]): str(row[0]) for row in combined_meac2}


''' Get the Dose Enhancement Factors for RBM '''
#########################################################################

if IMC.CTDI_phantom == False:
    # Load the file into a pandas DataFrame, assuming it's space-separated
    with open('H:\ImpactMC_CBCT\AFM_DEF.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Initialize an empty list to store rows with decimal values
    rows_with_floats = []

    # Iterate over each line in the file
    for line in lines:
        # Check if the line contains a float (decimal number)
        if re.search(r'\d+\.\d+', line):  # Regex to find a decimal number
            rows_with_floats.append(line.strip())  # Add the row to the list

    # Now split the rows into columns (separated by spaces)
    split_rows = [row.split() for row in rows_with_floats]

    # Select specific columns: 0, 1, 3, 5, 7, 9, 11, and 13
    RBM_columns = [0, 1, 3, 5, 7, 9, 11, 13]
    RBM_columns2 = [1, 3, 5, 7, 9, 11, 13]
    RBM_columns3 = [1, 3, 5, 7]

    # Select specific columns: 
    TM_columns = [0, 2, 4, 6, 8, 10, 12, 14]
    TM_columns2 = [2, 4, 6, 8, 10, 12, 14]
    TM_columns3 = [2, 4, 6, 8]

    # Extract the rows based on column indices
    array1 = [[row[i] for i in RBM_columns] for row in split_rows[:11]]
    array2 = [[row[i] for i in RBM_columns2] for row in split_rows[11:22]]
    array3 = [[row[i] for i in RBM_columns2] for row in split_rows[22:33]]
    array4 = [[row[i] for i in RBM_columns3] for row in split_rows[33:44]]

    # Extract the rows based on column indices
    array12 = [[row[i] for i in TM_columns] for row in split_rows[:11]]
    array22 = [[row[i] for i in TM_columns2] for row in split_rows[11:22]]
    array32 = [[row[i] for i in TM_columns2] for row in split_rows[22:33]]
    array42 = [[row[i] for i in TM_columns3] for row in split_rows[33:44]]

    # Combine the arrays
    result_array3 = [row1 + row2 + row3 + row4 for row1, row2, row3, row4 in zip(array1, array2, array3, array4)]
    result_array4 = [row1 + row2 + row3 + row4 for row1, row2, row3, row4 in zip(array12, array22, array32, array42)]

    # Function to find the row closest to the effective energy (E_eff)
    def find_closest_row(result_array3, E_eff):
        return min(result_array3, key=lambda row: abs(float(row[0]) - (E_eff / 1000)))

    # Find the closest row
    dose_efs = find_closest_row(result_array3, E_eff)
    dose_efs2 = find_closest_row(result_array4, E_eff)
    #print(dose_efs)
    #print(dose_efs2)

    if IMC.child_phantom == False:
        combined_des = np.column_stack((dose_efs, arr))
        des_spong_dict = {int(row[1]): str(row[0]) for row in combined_des}

        combined_des2 = np.column_stack((dose_efs2, arr))
        des_spong_dict2 = {int(row[1]): str(row[0]) for row in combined_des2}
    else:  
        spong_icrp_file2 = pd.read_csv(r'H:\ImpactMC_CBCT\CFM_spongiosa_ICRP_DES.txt', delimiter=',')
        spongiosa_num2 = spong_icrp_file2.iloc[:, 0].values  # First column (organ number)
        icrp_num2 = spong_icrp_file2.iloc[:, 1].values  # First column (organ number)
        arr2 = np.arange(1, 27)
        combined_icrpspong2 = np.column_stack((spongiosa_num2, icrp_num2)).astype(float)
        spongiosa_icrp_dict2 = {int(row[0]): int(row[1]) for row in combined_icrpspong2}

        combined_des = np.column_stack((dose_efs, arr2))
        des_spong_dict = {int(row[1]): str(row[0]) for row in combined_des}

        combined_des2 = np.column_stack((dose_efs2, arr2))
        des_spong_dict2 = {int(row[1]): str(row[0]) for row in combined_des2}


''' Calculate effective doses '''
#########################################################################

if IMC.CTDI_phantom == False:
    # Calculate final effective doses
    # Initialize second array to store the sum
    remainder_dose = np.zeros(1)  # Single-element array to store the sum
    remainder_dose_separate = np.zeros(13)
    effective_doses_final = np.zeros(count)
    spongiosa_doses_final = np.zeros(25)
    tldbones = [19,15,14]

    # Iterate over indices and values in effective_doses
    final_index = 0  # To keep track of where to store values in effective_doses_final
    final_index2 = 0  # To keep track of where to store values in spongiosa_doses_final
    final_index3 = 0  # To keep track of where to store values in remainder doses separate
    remainder_dose_count = 0
    total_remainder_dose = 0

    # Keep track of organ numbers for valid organs
    tissue_numbers_final = []
    tissue_numbers_spong_final = []

    all_effective_doses = []

    for i, dose in enumerate(mao_doses_tissues):
        # i = tissue number
        # Get the corresponding weighting number from the dictionary
        w_number = int(weighting_number_dict.get(i, 0))  

        if w_number == 0:
            all_effective_doses.append(dose)

        # Check if w_number is between 1 and 14 and store in effective_doses_final
        if 1 <= w_number <= 14:
            w_T = w_T_dict.get(w_number, 0)  # Get w_T for w_number, default to 0 if not found
            effectivedose = dose * w_T
            effective_doses_final[final_index] += effectivedose
            tissue_numbers_final.append(i)  # Append the corresponding organ number to the list
            final_index += 1  # Move to the next spot in effective_doses_final for valid organs
            all_effective_doses.append(effectivedose)

        # If w_number is 15 or higher, add to remainder_dose
        elif 15 <= w_number <= 28:
            E_dose = dose * remainder_tissues_TLD
            remainder_dose_separate[final_index3] += E_dose
            total_remainder_dose += dose  #Sum the dose into the second array
            remainder_dose_count += 1  # Increment the counter for doses added to remainder_dose
            final_index3 += 1
            all_effective_doses.append(E_dose)
        
        # Corrections for spongiosa regions
        elif w_number >= 29:
            # Getting the mass fractions for Bone and RBM for current organ number
            Rbm = spongiosa_RBM_dict.get(i+1,0)  # Mass fraction
            TM50 = spongiosa_TM50_dict.get(i+1,0) # Mass fraction
            w_T_Rbm = 0.12
            w_T_TM50 = 0.12

            # Getting the MEAC ratio using the effective energy
            spong_nr = int(organ_to_spong_number_dict.get(i+1, [0])[0])
            icrp_number = spongiosa_icrp_dict.get(spong_nr,0)
            meac_ratio = meac_spong_dict.get(icrp_number,0)
    
            if meac_ratio != 'NA' :
                meac_ratio = float(meac_ratio)
            else:
                meac_ratio = 1

            # Getting the MEAC ratio using the effective energy
            icrp_number2 = spongiosa_icrp_dict.get(spong_nr,0)
            meac_ratio2 = meac_spong_dict2.get(icrp_number2,0)

            if meac_ratio2 != 'NA' :
                meac_ratio2 = float(meac_ratio2)
            else:
                meac_ratio2 = 1

            # Getting the Dose Enhancement Factor using the effective energy
            icrp_number = spongiosa_icrp_dict2.get(spong_nr,0)
            dose_ef = des_spong_dict.get(icrp_number,0)

            if dose_ef != 'NA' :
                dose_ef = float(dose_ef)
            else:
                dose_ef = 1

            # Getting the Dose Enhancement Factor using the effective energy
            icrp_number2 = spongiosa_icrp_dict2.get(spong_nr,0)
            dose_ef2 = des_spong_dict2.get(icrp_number2,0)

            if dose_ef2 != 'NA' :
                dose_ef2 = float(dose_ef2)
            else:
                dose_ef2 = 1
            
            dose_rbm = dose * meac_ratio * dose_ef * Rbm
            dose_tm50 = dose * TM50 * meac_ratio2 * dose_ef2 

            E_Rbm = dose_rbm * w_T_Rbm
            E_tm50 = dose_tm50 * w_T_TM50
            E_final = E_Rbm + E_tm50
            spongiosa_doses_final[final_index2] += E_final
            tissue_numbers_spong_final.append(i)
            final_index2 += 1
            all_effective_doses.append(E_final)

            #if i in tldbones:
               # print(dose)
               # print(meac_ratio)
              ##  print(meac_ratio2)
              #  print(dose_ef)
              #  print(dose_ef2)
              #  print(Rbm)
              #  print(TM50)
              #  print(dose_rbm)
              #  print(dose_tm50)
              #  print(E_Rbm)
              #  print(E_tm50)
              #  print(E_final)


    # After the loop, if any doses were added to remainder_dose, calculate the average
    if remainder_dose_count > 0:
        remainder_dose[0] = total_remainder_dose / remainder_dose_count  # Calculate the average

    # Compute remainder effective dose
    remainder_effective_dose = remainder_dose * remainder_factor

    # Print the results
    print(f"Remainder Effective Dose: {remainder_effective_dose[0]:.8f} mGy")  # Ensure it's formatted correctly

    for organ, dose in zip(remainder_tissues_names, remainder_dose_separate):
        print(f"Tissue {organ}: Remainder Effective Dose = {dose:.8f} mGy")

    print("Effective Doses Final (for tissues with w_number between 1 and 14):")

    # Print effective_doses_final with corresponding organ numbers
    for organ, dose in zip(tissue_numbers_final, effective_doses_final):
        print(f"Tissue {organ}: Effective Dose = {dose:.8f} mGy")

    print("Effective Doses Final (for Tissues with w_number of 29):")

    # Print effective_doses_final with corresponding organ numbers
    for organ, dose in zip(tissue_numbers_spong_final, spongiosa_doses_final):
        print(f"Tissue {organ}: Effective Dose = {dose:.8f} mGy")

    for dose in all_effective_doses:
        print(f"{dose:.8f}")



''' Perform CTDI Calculations '''
#########################################################################

if IMC.CTDI_phantom == True:
    def compute_ctdi100(dose_profile, slice_thickness, center_z, shape_z):
        """
        Computes CTDI100 from a given dose profile along the z-axis.

        Parameters:
        - dose_profile: 1D numpy array of dose values along the z-axis
        - slice_thickness: float, the thickness of each slice in mm

        Returns:
        - CTDI100 value
        """
        # Number of slices in the dose profile
        num_slices = len(np.ravel(dose_profile))

        # Define the range for z (±50 mm around the center)
        z_range_min = int(center_z - (50/slice_thickness))
        z_range_max = int(center_z + (50/slice_thickness))

        # Generate z positions, centered around the actual center_z
        z = np.linspace(0, shape_z, num=num_slices)

        # Now filter the dose_profile and z values to include only those within the desired range
        # Assuming z is uniformly spaced and corresponds to the dose_profile
        mask = (z >= z_range_min) & (z <= z_range_max)

        # Apply the mask to the dose_profile and corresponding z values
        filtered_dose_profile = np.ravel(dose_profile)[mask]
        filtered_z = z[mask]

        # Perform numerical integration (trapezoidal rule) only over the filtered values
        dose_integral = np.trapezoid(filtered_dose_profile, filtered_z)

        # Normalize over a Collimation mm range for CTDI100
        return dose_integral / (IMC.total_beam_collimation/slice_thickness)

    def compute_ctdiw(ctdi100_center, ctdi100_periphery):
        """Computes the weighted CTDIw."""
        return (1/3) * ctdi100_center + (2/3) * ctdi100_periphery

    # Load dose_volume (3D numpy array)
    shape_z, shape_y, shape_x = volume.shape
    slice_thickness = IMC.slice_thickness

    # Define center and periphery locations
    center_z, center_x, center_y = shape_z // 2, shape_x // 2, shape_y // 2

    # Define the range for z (±50 mm around the center)
    z_range_min = int(center_z - (50/slice_thickness))
    z_range_max = int(center_z + (50/slice_thickness))

    # Find the edges of the phantom (assuming nonzero dose region)
    phantom_mask = volume > 0

    # Slice off the first 20 pixels on the y-axis before checking x and y
    phantom_mask_trimmed = phantom_mask[:, :-20, :]

    # X-axis: ignore first 20 y-pixels by slicing before reducing along axis 1
    x_nonzero = np.where(np.any(phantom_mask_trimmed, axis=(0, 1)))[0]

    # Y-axis: already trimmed, but we need to adjust indices back to match original volume
    y_nonzero = np.where(np.any(phantom_mask_trimmed, axis=(0, 2)))[0] 

    # Compute bounds
    x_min, x_max = x_nonzero[0], x_nonzero[-1]
    y_min, y_max = y_nonzero[0], y_nonzero[-1]

    # Peripheral points (3 mm from the edge)
    voxel_size = IMC.voxel_size_xy[0]
    periphery_points = [
        (int(center_y), int(x_min + (3/voxel_size) + ((13.1/2)/voxel_size))),  # Left
        (int(center_y), int(x_max - (3/voxel_size) -4)),  # Right
        (int(y_min + (3/voxel_size) + ((13.1/2)/voxel_size)), int(center_x)),  # Top
        (int(y_max - (3/voxel_size) -4), int(center_x))   # Bottom
    ]

    # Extract dose profiles
    center_profile = volume[:, center_y, center_x]  # Center
    periphery_profiles = [volume[:, y, x] for y, x in periphery_points]

    # Compute the average across the 4 arrays
    average_periphery_profile = np.mean(periphery_profiles, axis=0)

    # Compute CTDI100 values with updated center_z
    ctdi100_center = compute_ctdi100(center_profile, slice_thickness, center_z, shape_z)
    ctdi100_periphery = compute_ctdi100(average_periphery_profile, slice_thickness, center_z, shape_z)

    # Compute CTDIw
    ctdi_w = compute_ctdiw(ctdi100_center, ctdi100_periphery)

    # Ratio
    ratio = ctdi100_center / ctdi100_periphery

    print(f"CTDI100 (Center): {ctdi100_center:.2f} mGy")
    print(f"CTDI100 (Periphery): {ctdi100_periphery:.2f} mGy")
    print(f"CTDIw: {ctdi_w:.2f} mGy")
    print(f"Ratio: {ratio:.2f}")


''' Comparing CTDI '''
#########################################################################

if IMC.CTDI_phantom == True:
    # Ratios from manual
    ratio_real16 = 46.9 / 47.8
    ratio_real32 = 3.9 / 7.1
    ratio_CT16 = 22.00 / 17.91
    ratio_CT32 = 1.51 / 2.87

    IMC16_c = 22.54
    IMC16_p = 24.89
    IMC32_c = 1.20
    IMC32_p = 2.79
    CT16_c = 22.00
    CT16_p = 17.91
    CT32_c = 1.51
    CT32_p = 2.87
    ratio_IMC16 = IMC16_c / IMC16_p
    ratio_IMC32 = IMC32_c / IMC32_p

    # Relative difference
    rel_diff16 = (abs(ratio_real16 - ratio) / ratio_real16) * 100
    #rel_diff32 = (abs(ratio_real32 - ratio) / ratio_real32) * 100

    rel_diff16_IMCc = (abs(IMC16_c - ctdi100_center) / IMC16_c) * 100
    #rel_diff32_IMCc = (abs(IMC32_c - ctdi100_center) / IMC32_c) * 100
    rel_diff16_IMCp = (abs(IMC16_p - ctdi100_periphery) / IMC16_p) * 100
    #rel_diff32_IMCp = (abs(IMC32_p - ctdi100_periphery) / IMC32_p) * 100

    rel_diff16_CTc = (abs(ctdi100_center - CT16_c) / ctdi100_center) * 100
    #rel_diff32_CTc = (abs(ctdi100_center - CT32_c) / ctdi100_center) * 100
    rel_diff16_CTp = (abs(ctdi100_periphery - CT16_p) / ctdi100_periphery) * 100
    #rel_diff32_CTp = (abs(ctdi100_periphery - CT32_p) / ctdi100_periphery) * 100

    rel_diff16_IMC = (abs(ratio - ratio_IMC16) / ratio) * 100
    #rel_diff32_IMC = (abs(ratio - ratio_IMC32) / ratio) * 100

    rel_diff16_CT = (abs(ratio - ratio_CT16) / ratio) * 100
    #rel_diff32_CT = (abs(ratio - ratio_CT32) / ratio) * 100

    #print(f"Relative difference: {rel_diff16:.4f}%")
    print(f"Relative difference: {rel_diff16:.4f}%")

    print(f"Relative difference: {rel_diff16_IMCc:.4f}%")
    print(f"Relative difference: {rel_diff16_IMCp:.4f}%")

    print(f"Relative difference: {rel_diff16_CTc:.4f}%")
    print(f"Relative difference: {rel_diff16_CTp:.4f}%")

    print(f"Relative difference: {rel_diff16_IMC:.4f}%")

    print(f"Relative difference: {rel_diff16_CT:.4f}%")


''' Printing the middle 100 Dose values to check '''
#########################################################################

if IMC.CTDI_phantom == False:
    # Access and print =100 values of the 269th slice
    slice_index = 100
    slice_269 = volume[slice_index, :, :]

    # Get the middle 100 values (10x10 region)
    height, width = slice_269.shape
    center_y, center_x = height // 2, width // 2
    middle_100_values = slice_269[center_y - 5:center_y + 5, center_x - 5:center_x + 5]

    # Flatten and print the first 100 values
    flattened_values = middle_100_values.flatten()

    #print("Middle 100 values of the 269th slice (in mGy):")
    #print(flattened_values)


''' Plotting the Dose distribution to check '''
#########################################################################

plotting = True
if plotting == True:
    volume2 = volume.astype(np.float32)  # Convert to float first
    #volume1 = IMC.volume.astype(np.float32)
    #volume2 = np.round(volume).astype(np.int16)  # Round and convert to int

    # Define initial slice indices
    axial_idx2 = volume2.shape[0] // 2  # Middle slice for axial
    coronal_idx2 = volume2.shape[1] // 2  # Middle slice for coronal
    sagittal_idx2 = volume2.shape[2] // 2  # Middle slice for sagittal

    # Find the minimum and maximum pixel values in the entire volume
    min_val = np.min(volume2)
    max_val = np.max(volume2)
    #min_val2 = np.min(volume1)
    #max_val2 = np.max(volume1)

    # Create a custom colormap where the minimum value is black and the maximum is white
    cmap = plt.cm.gray  # Use the gray colormap, which already ranges from black to white

    # Create the figure and subplots for axial, coronal, and sagittal slices
    #fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows: volume1 (top), volume2 (bottom)

    # Update slice display with greyscale colors
    # Normalize the pixel values using min_val and max_val
    # First row: volume1 (top)
    #axial_img1 = axes2[0, 0].imshow(volume1[axial_idx2, :, :], cmap=cmap, vmax=max_val)
    #coronal_img1 = axes2[0, 1].imshow(volume1[:, coronal_idx2, :], cmap=cmap, vmin=min_val, vmax=max_val)
    #sagittal_img1 = axes2[0, 2].imshow(volume1[:, :, sagittal_idx2], cmap=cmap, vmin=min_val, vmax=max_val)

    # Second row: volume2 (bottom, as in your original code)
    axial_img2 = axes2[1, 0].imshow(volume2[axial_idx2, :, :], cmap=cmap, vmin=min_val, vmax=max_val)
    coronal_img2 = axes2[1, 1].imshow(volume2[:, coronal_idx2, :], cmap=cmap, vmin=min_val, vmax=max_val)
    sagittal_img2 = axes2[1, 2].imshow(volume2[:, :, sagittal_idx2], cmap=cmap, vmin=min_val, vmax=max_val)

    # Define the x, y coordinate in the axial slice where the line should be drawn
    if IMC.CTDI_phantom == True:
        x_axial = periphery_points[0][1]  # Example x-coordinate on the axial slice
        y_axial = periphery_points[0][0]  # Example y-coordinate on the axial slice
        x_axial2 = periphery_points[1][1]  # Example x-coordinate on the axial slice
        y_axial2 = periphery_points[1][0]  # Example y-coordinate on the axial slice
        x_axial3 = periphery_points[2][1]  # Example x-coordinate on the axial slice
        y_axial3 = periphery_points[2][0]  # Example y-coordinate on the axial slice
        x_axial4 = periphery_points[3][1]  # Example x-coordinate on the axial slice
        y_axial4 = periphery_points[3][0]  # Example y-coordinate on the axial slice
        x_axial5 = center_x  # Example x-coordinate on the axial slice
        y_axial5 = center_y  # Example y-coordinate on the axial slice

        # Plot vertical line on coronal slice at x_axial (spanning from top to bottom)
        coronal_line, = axes2[1].plot([x_axial, x_axial], [z_range_min, z_range_max], 'r-', linewidth=2)
        coronal_line2, = axes2[1].plot([x_axial2, x_axial2], [z_range_min, z_range_max], 'r-', linewidth=2)
        coronal_line3, = axes2[1].plot([x_axial3, x_axial3], [z_range_min, z_range_max], 'r-', linewidth=2)
        coronal_line4, = axes2[1].plot([x_axial4, x_axial4], [z_range_min, z_range_max], 'r-', linewidth=2)
        coronal_line5, = axes2[1].plot([x_axial5, x_axial5], [z_range_min, z_range_max], 'r-', linewidth=2)

        # Plot vertical line on sagittal slice at y_axial (spanning from top to bottom)
        sagittal_line, = axes2[2].plot([y_axial, y_axial], [z_range_min, z_range_max], 'r-', linewidth=2)
        sagittal_line2, = axes2[2].plot([y_axial2, y_axial2], [z_range_min, z_range_max], 'r-', linewidth=2)
        sagittal_line3, = axes2[2].plot([y_axial3, y_axial3], [z_range_min, z_range_max], 'r-', linewidth=2)
        sagittal_line4, = axes2[2].plot([y_axial4, y_axial4], [z_range_min, z_range_max], 'r-', linewidth=2)
        sagittal_line5, = axes2[2].plot([y_axial5, y_axial5], [z_range_min, z_range_max], 'r-', linewidth=2)

        # Plot the intersection point on the axial slice
        axial_point, = axes2[0].plot(x_axial, y_axial, 'ro', markersize=2)  # 'ro' = red circle marker
        axial_point2, = axes2[0].plot(x_axial2, y_axial2, 'ro', markersize=2)  # 'ro' = red circle marker
        axial_point3, = axes2[0].plot(x_axial3, y_axial3, 'ro', markersize=2)  # 'ro' = red circle marker
        axial_point4, = axes2[0].plot(x_axial4, y_axial4, 'ro', markersize=2)  # 'ro' = red circle marker
        axial_point5, = axes2[0].plot(x_axial5, y_axial5, 'ro', markersize=2)  # 'ro' = red circle marker

    # Titles
    #axes2[0, 0].set_title(f"Volume 1 - Axial Slice {axial_idx2}")
    #axes2[0, 1].set_title(f"Volume 1 - Coronal Slice {coronal_idx2}")
    #axes2[0, 2].set_title(f"Volume 1 - Sagittal Slice {sagittal_idx2}")
    axes2[1, 0].set_title(f"Volume 2 - Axial Slice {axial_idx2}")
    axes2[1, 1].set_title(f"Volume 2 - Coronal Slice {coronal_idx2}")
    axes2[1, 2].set_title(f"Volume 2 - Sagittal Slice {sagittal_idx2}")

    # Remove axis labels
    for ax_row in axes2:
        for ax in ax_row:
            ax.axis("off")

    # Add sliders
    axcolor = 'lightgoldenrodyellow'
    axial_slider_ax2 = plt.axes([0.2, 0.02, 0.65, 0.02], facecolor=axcolor)
    coronal_slider_ax2 = plt.axes([0.2, 0.05, 0.65, 0.02], facecolor=axcolor)
    sagittal_slider_ax2 = plt.axes([0.2, 0.08, 0.65, 0.02], facecolor=axcolor)

    axial_slider2 = Slider(axial_slider_ax2, 'Axial', 0, volume.shape[0] - 1, valinit=axial_idx2, valstep=1)
    coronal_slider2 = Slider(coronal_slider_ax2, 'Coronal', 0, volume.shape[1] - 1, valinit=coronal_idx2, valstep=1)
    sagittal_slider2 = Slider(sagittal_slider_ax2, 'Sagittal', 0, volume.shape[2] - 1, valinit=sagittal_idx2, valstep=1)

    # Function to update slices and lines
    def update(val):
        axial_idx2 = int(axial_slider2.val)
        coronal_idx2 = int(coronal_slider2.val)
        sagittal_idx2 = int(sagittal_slider2.val)

        # Update slice images
        axial_img2.set_data(volume2[axial_idx2, :, :])
        coronal_img2.set_data(volume2[:, coronal_idx2, :])
        sagittal_img2.set_data(volume2[:, :, sagittal_idx2])

        #axial_img1.set_data(volume1[axial_idx2, :, :])
        #coronal_img1.set_data(volume1[:, coronal_idx2, :])
        #sagittal_img1.set_data(volume1[:, :, sagittal_idx2])

        if IMC.CTDI_phantom == True:
            # Update line positions dynamically
            coronal_line.set_data([x_axial, x_axial], [z_range_min, z_range_max])
            sagittal_line.set_data([y_axial, y_axial], [z_range_min, z_range_max])
            coronal_line2.set_data([x_axial2, x_axial2], [z_range_min, z_range_max])
            sagittal_line2.set_data([y_axial2, y_axial2], [z_range_min, z_range_max])
            coronal_line3.set_data([x_axial3, x_axial3], [z_range_min, z_range_max])
            sagittal_line3.set_data([y_axial3, y_axial3], [z_range_min, z_range_max])
            coronal_line4.set_data([x_axial4, x_axial4], [z_range_min, z_range_max])
            sagittal_line4.set_data([y_axial4, y_axial4], [z_range_min, z_range_max])
            coronal_line5.set_data([x_axial5, x_axial5], [z_range_min, z_range_max])
            sagittal_line5.set_data([y_axial5, y_axial5], [z_range_min, z_range_max])

            # Update the axial point dynamically
            axial_point.set_data(x_axial, y_axial)
            axial_point2.set_data(x_axial2, y_axial2)
            axial_point3.set_data(x_axial3, y_axial3)
            axial_point4.set_data(x_axial4, y_axial4)
            axial_point5.set_data(x_axial5, y_axial5)

        # Update titles
        axes2[0].set_title(f"Axial Slice {axial_idx2}")
        axes2[1].set_title(f"Coronal Slice {coronal_idx2}")
        axes2[2].set_title(f"Sagittal Slice {sagittal_idx2}")

        # Redraw
        fig2.canvas.draw_idle()

    # Connect sliders
    axial_slider2.on_changed(update)
    coronal_slider2.on_changed(update)
    sagittal_slider2.on_changed(update)

    # Show the plot
    plt.show()



