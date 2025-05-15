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
import xml.etree.ElementTree as ET

''' 
This code can be used to transform the scan and simulation parameters of (CB)CT systems
into relevant input for ImpactMC (Erlangen, Germany). Also, this code is able to create an 
input volume using ICRP computational phantoms or CTDI phantoms. The input section underneath 
can be usedto set the parameters that correspond to your specific set-up. The code outputs 
different textor XML files, relevant plots and saves the input volume into a DICOM file.

F. van Wolferen
'''

''' In this section you can give all input necessary to create ImpactMC files for used filters, collimation,
tube current modulation, and more! '''
####################################################

# Input 
''' General input '''
sex_of_phantom = 'Female' # Male or Female
child_phantom = True # Is it a Child or Adult phantom?
age_of_child = '10' # Age of the child phantom (00, 01, 05, 10, 15)
cwd = os.getcwd()  # Get the current working directory (cwd)
show_plot = True # Do you want to show a plot?
measure_in_air = False # Do you want to measure in air? (no phantom)
ring = False # Do you want to simulate a ring?
CTDI_phantom = False # Are you simulating a CTDI phantom?
CTDI_diameter = 320 # Diameter of the CTDI phantom in mm (320 or 160)

''' Table input '''
with_table = True # Choose whether you want to model a table
table_back = False # Do you want the table to be located at the back or front of the phantom?
thickness_table = [2.5, 0.5, 4] # Thickness table in mm
material_table = ['C' , 'H2O', 'H2O'] # Materials of the table
name_atoms_table = 'C' 
name_atoms_table2 = ['H', 'O']
nr_atoms_table = 1
nr_atoms_table2 = [2,1] 
table_densities = [1.7, 1.0, 1.0] # Densities of the table materials in g/cm3
width_table = 500 # Width of the table in mm

''' Spectrum input '''
dir_spec = os.path.join(cwd, 'spekpy_spectra') # Specify path and name of folder containing spectral data
input_dict = {'name':'W-120', # X-ray spectrum specifications
         'kV': 117, # Tube Voltage
         'physics':'casim',
         'dk': 1.0, # Energy bin width,
         'filters': {'Al':3.5,
                     'Cu':0,
                     'Air': 0
                     }
         }

''' Collimation input '''
symmetric_collimation = True # Set this to False in the case of Asymmetric collimation or Dynamic collimation
total_beam_collimation = [270,210] # Total beam collimation value in mm
distance_from_middlepoint_fov = 0 # Distance from the symmetric middle point to the asymmetric middle point on the field of view in mm (can be either + or -, max total beam collimation / 2)

''' Filter input '''
type_filter = 'Bowtie' # Type of filter (Flat or Bowtie)
thickness_filter = [1.0, 0.4] # Thickness of the filter in mm
angular_increment = 0.1 # Angluar increment of the filter file, in degrees
filter_materials = ['Al', 'Cu'] # Material type 1 of the filter
density_filter = [2.7, 8.96] # Density of the filter material in g/cm^3
distance_source_isocenter = 595 # Distance from the source to the isocenter in mm


''' Tube current input '''
tube_current_modulation = True # Do you want to use Tube Current Modulation?
number_of_rotations = 0.5 # How many rotations does your set-up make?
rotation_time = 1.0 # Rotation time in s
nr_projection_angles = 360 # Number of projection angles for a full rotation

''' LOAD ALL FILES FOR GIVEN GENERAL INPUT '''
####################################################

slice_thicknesses = [['00', 0.663],['01', 1.400], ['05', 1.928], ['10', 2.425], ['15', 2.832]]
slice_thickness_F15 = 2.828
voxel_sizes = [['00', 0.663],['01', 0.663], ['05', 0.850], ['10', 0.990], ['15', 1.250]]
voxel_size_15F = 1.200
dimensions = [['00', 345, 211, 716],['01', 393, 248, 546], ['05', 419, 230, 572], ['10', 419, 226, 576], ['15', 407, 225, 586]]
dimensions_15F = [401, 236, 571]
slicethick_dict = {str(row[0]): float(row[1]) for row in slice_thicknesses}
voxsize_dict = {str(row[0]): float(row[1]) for row in voxel_sizes}
dim_dict = {str(row[0]): (float(row[1]), float(row[2]), float(row[3])) for row in dimensions}

if CTDI_phantom == True:
    slice_thickness = 1.00 # Slice thickness of the DICOM slices in mm
    voxel_size_xy = [1.00, 1.00] # Voxel size of the x and y directions of the voxels in mm
    width, height, depth = 299, 137, 346
    child_phantom = False
    if CTDI_diameter == 320:
        fit_start = 370
        fit_end = 418
    else:
        fit_start = 211
        fit_end = 258
    folder_path2 = Path("H:\ImpactMC_CBCT\ICRP Female\AF_media.dat") # Folder path to media data from computational phantom
    folder_path3 = Path("H:\ImpactMC_CBCT\ICRP Female\AF_organs.dat") # Folder path to organ data from computational phantom
    folder_path4 = Path("H:\ImpactMC_CBCT\ICRP Female\AF.dat") # Folder path to 3D voxel data from computational phantom
    folder_path5 = Path("H:\ImpactMC_CBCT\ICRP Female\AF_spongiosa.dat") # Folder path to spongiosa data from computational phantom
    if measure_in_air == True:
        if CTDI_phantom == True:
                output_folder = r"H:/ImpactMC_CBCT/DICOM_output_CTDI"
        else:
                output_folder = r"H:/ImpactMC_CBCT/DICOM_output_Air" # Folder path to output folder for DICOM files
else:
    if sex_of_phantom == 'Female' and child_phantom == False :
            folder_path2 = Path("H:\ImpactMC_CBCT\ICRP Female\AF_media.dat") # Folder path to media data from computational phantom
            folder_path3 = Path("H:\ImpactMC_CBCT\ICRP Female\AF_organs.dat") # Folder path to organ data from computational phantom
            folder_path4 = Path("H:\ImpactMC_CBCT\ICRP Female\AF.dat") # Folder path to 3D voxel data from computational phantom
            folder_path5 = Path("H:\ImpactMC_CBCT\ICRP Female\AF_spongiosa.dat") # Folder path to spongiosa data from computational phantom
            slice_thickness = 4.84 # Slice thickness of the DICOM slices in mm
            voxel_size_xy = [1.775, 1.775] # Voxel size of the x and y directions of the voxels in mm
            width, height, depth = 299, 137, 346
            if table_back == False:
                fit_start = 0
                fit_end = 0
            else:
                fit_start = 0
                fit_end = 0
            output_folder = r"H:/ImpactMC_CBCT/DICOM_output_Female" # Folder path to output folder for DICOM files
    elif sex_of_phantom == 'Male' and child_phantom == False :
            folder_path2 = Path("H:\ImpactMC_CBCT\ICRP Male\AM_media.dat") # Folder path to media data from computational phantom
            folder_path3 = Path("H:\ImpactMC_CBCT\ICRP Male\AM_organs.dat") # Folder path to organ data from computational phantom
            folder_path4 = Path("H:\ImpactMC_CBCT\ICRP Male\AM.dat") # Folder path to 3D voxel data from computational phantom
            folder_path5 = Path("H:\ImpactMC_CBCT\ICRP Male\AM_spongiosa.dat") # Folder path to spongiosa data from computational phantom
            slice_thickness = 8.0 # Slice thickness of the DICOM slices in mm
            voxel_size_xy = [2.137, 2.137] # Voxel size of the x and y directions of the voxels in mm
            width, height, depth = 254, 127, 220
            if table_back == False:
                fit_start = 0
                fit_end = 0
            else:
                fit_start = 0
                fit_end = 0
            output_folder = r"H:/ImpactMC_CBCT/DICOM_output_Male" # Folder path to output folder for DICOM files
    elif child_phantom == True :
        if sex_of_phantom == 'Male' :
                folder_path2 = Path(f"H:/ImpactMC_CBCT/ICRP Child/{age_of_child}M/{age_of_child}M_media.dat") # Folder path to media data from computational phantom
                folder_path3 = Path(f"H:/ImpactMC_CBCT/ICRP Child/{age_of_child}M/{age_of_child}M_organs.dat") # Folder path to organ data from computational phantom
                folder_path4 = Path(f"H:/ImpactMC_CBCT/ICRP Child/{age_of_child}M/{age_of_child}M_ascii.dat") # Folder path to 3D voxel data from computational phantom
                folder_path5 = Path(f"H:/ImpactMC_CBCT/ICRP Child/{age_of_child}M/{age_of_child}M_skeleton.dat") # Folder path to spongiosa data from computational phantom
                slice_thickness = slicethick_dict.get(age_of_child, 0) # Slice thickness of the DICOM slices in mm
                voxel_size_xy = [voxsize_dict.get(age_of_child, 0),voxsize_dict.get(age_of_child, 0)] # Voxel size of the x and y directions of the voxels in mm
                dimension = dim_dict.get(age_of_child, 0)
                width, height, depth = dimension
                if table_back == False:
                    fit_start = 0
                    fit_end = 0
                else:
                    fit_start = 0
                    fit_end = 0
                output_folder = r"H:/ImpactMC_CBCT/DICOM_output_Child" # Folder path to output folder for DICOM files
        elif sex_of_phantom == 'Female' and age_of_child != '15':
                folder_path2 = Path(f"H:/ImpactMC_CBCT/ICRP Child/{age_of_child}F/{age_of_child}F_media.dat") # Folder path to media data from computational phantom
                folder_path3 = Path(f"H:/ImpactMC_CBCT/ICRP Child/{age_of_child}F/{age_of_child}F_organs.dat") # Folder path to organ data from computational phantom
                folder_path4 = Path(f"H:/ImpactMC_CBCT/ICRP Child/{age_of_child}F/{age_of_child}F_ascii.dat") # Folder path to 3D voxel data from computational phantom
                folder_path5 = Path(f"H:/ImpactMC_CBCT/ICRP Child/{age_of_child}F/{age_of_child}F_skeleton.dat") # Folder path to spongiosa data from computational phantom
                slice_thickness = slicethick_dict.get(age_of_child, 0) # Slice thickness of the DICOM slices in mm
                voxel_size_xy = [voxsize_dict.get(age_of_child, 0),voxsize_dict.get(age_of_child, 0)] # Voxel size of the x and y directions of the voxels in mm
                dimension = dim_dict.get(age_of_child, 0)
                width, height, depth = dimension
                if table_back == False:
                    fit_start = 210
                    fit_end = 225
                else:
                    fit_start = 0
                    fit_end = 0
                output_folder = r"H:/ImpactMC_CBCT/DICOM_output_Child" # Folder path to output folder for DICOM files
        elif sex_of_phantom == 'Female' and age_of_child == '15':
                folder_path2 = Path(f"H:/ImpactMC_CBCT/ICRP Child/{age_of_child}F/{age_of_child}F_media.dat") # Folder path to media data from computational phantom
                folder_path3 = Path(f"H:/ImpactMC_CBCT/ICRP Child/{age_of_child}F/{age_of_child}F_organs.dat") # Folder path to organ data from computational phantom
                folder_path4 = Path(f"H:/ImpactMC_CBCT/ICRP Child/{age_of_child}F/{age_of_child}F_ascii.dat") # Folder path to 3D voxel data from computational phantom
                folder_path5 = Path(f"H:/ImpactMC_CBCT/ICRP Child/{age_of_child}F/{age_of_child}F_skeleton.dat") # Folder path to spongiosa data from computational phantom
                slice_thickness = slice_thickness_F15 # Slice thickness of the DICOM slices in mm
                voxel_size_xy = [voxel_size_15F, voxel_size_15F] # Voxel size of the x and y directions of the voxels in mm
                width, height, depth = dimensions_15F
                if table_back == False:
                    fit_start = 120
                    fit_end = 135
                else:
                    fit_start = 0
                    fit_end = 0
                output_folder = r"H:/ImpactMC_CBCT/DICOM_output_Child" # Folder path to output folder for DICOM files
path = r"H:/ImpactMC_CBCT/Parameter files/"


''' GET ENERGY SPECTRUM, SAVE IT IN A TEXT-FILE '''
####################################################
# Generate x-ray spectrum functions

def gen_spectrum(inputs, angle=30, mu_data='pene'):
    '''
    This function generates a Spekpy spectrum from a dictionary of input params
    Parameters
    ----------
    spectrum_inputs : Dict of input filtrations and applied voltage, in kV
        Keys include Mo, Al, Sn, Pb, Cu, and items are expressed in thickness (mm)
        Keys should aldo include the tube voltage, in kV
        Keys should also include the anode angle.

    Returns a Spekpy spectrum object
    '''
    
    phys = 'spekcalc'
    if (('physics' in inputs) and ('kV' in inputs)):
        phys = inputs['physics']
    s = sp.Spek(kvp=inputs['kV'], 
                th=angle,
                physics=phys, 
                mu_data_source = mu_data,
                comment=inputs['name']) 
    filter_list = [(k, v) for k, v in inputs['filters'].items()] 
    s.multi_filter(filter_list)
    if 'mu_data' in inputs:
        s.set(mu_data_source = inputs['mu_data'])
    if 'dk' in inputs:
        s.set(dk=inputs['dk'])
    if 'angle' in inputs:
        s.angle= inputs['angle']
    return s

def spectrum_filename(dic):
    '''

    Parameters
    ----------
    dic : Dictionary of spectrum input data
        Creates a file name via string concatenation, using
        the dictionary as a source of information

    Returns
    -------
    String: filename
    '''
    name = ''
    for key, item in dic['filters'].items(): # zip(WMo28.keys(), WMo28.values()):
        name=name+' ' + key+str(item)
        spek_name = dic['name'] + ' ' + str(dic['kV']) + 'kV' + name + '.spec'
    return spek_name

def scarto_rel(a,b):
    return np.abs(a-b) / np.min([a,b])

# Give the input parameters for the spectrum
fname_qual = spectrum_filename(input_dict)
spectrum = gen_spectrum(input_dict)
input_dict['spekpy_spectrum'] = spectrum
''' 
outputs spectrum to .spec file for later uses. Note that this includes
information such as HVL1, HVL2, mean energy etc 
''' 
kbins, spk = spectrum.get_spectrum(edges=False) # returns mid-bin energy and freq.
input_dict['spectrum_df'] = pd.DataFrame({'keV': kbins, 'freq': spk}, index = None)
input_dict['spectrum_df'].to_csv(os.path.join(dir_spec,fname_qual+'.txt'), index=False, sep='\t')
input_dict['mean_E'] = spectrum.get_emean()

dfinput_dict = pd.concat([pd.Series(input_dict)])
dfinput_dict = dfinput_dict.transpose()
dfinput_dict.drop(columns=['spekpy_spectrum', 'spectrum_df'], inplace=True)
#print(dfAllQualities)

dfinput_dict.to_csv(os.path.join(cwd,'spekpy_qualities_summary.csv'), index=False, sep=',')

# Simulate with 15 cm of water
# Water attenuation coefficients (approx. from NIST, in mm^-1)
def get_water_mu(energies_keV):
    # Basic approximation: real implementation should interpolate from actual data
    # Example values (not accurate): you should replace with precise NIST mu/rho * density
    mu_over_rho = np.interp(energies_keV,
                            [10, 20, 30, 40, 50, 60, 80, 100, 120],
                            [4.5, 1.2, 0.5, 0.3, 0.2, 0.15, 0.08, 0.05, 0.035])  # cm^2/g
    density = 1.0  # g/cm^3
    mu = mu_over_rho * density / 10  # convert to mm^-1
    return mu

# Apply water attenuation
water_thickness_mm = 100  # 15 cm
mu_water = get_water_mu(kbins)
attenuated_spk = spk * np.exp(-mu_water * water_thickness_mm)

# Store attenuated spectrum
input_dict['spectrum_df_water'] = pd.DataFrame({'keV': kbins, 'freq': attenuated_spk})
input_dict['spectrum_df_water'].to_csv(os.path.join(dir_spec, fname_qual + '_water.txt'), index=False, sep='\t')

mean_energy_water = np.sum(kbins * attenuated_spk) / np.sum(attenuated_spk)
#print("Mean Energy (E_mean) attenuated:", mean_energy_water, "keV")
E_eff2 = mean_energy_water

# Open the file with energy spectrum data
# Create full file path
file_path = os.path.join(dir_spec, fname_qual+'.txt')

# Open the file
with open(file_path, 'r') as file:
    spectrum_data = file.readlines()

# Open a file in write mode
with open(path+"user_defined_spectrum.txt", "w") as file:
    # Write the header lines
    file.write("# Spectral Data File\n")
    file.write("# Energy, Photons/(4Pi*s)\n")
    
    # Loop through the data (skipping the first row)
    for row in spectrum_data[1:]:
        # Strip the row of any leading/trailing whitespace and newlines
        row = row.strip()
        
        # Replace the tab character with a comma
        row = row.replace('\t', ', ')
        
        # Write the row to the file with a newline at the end
        file.write(row + '\n')

print("Spectrum file written succesfully!")

# Parse the data
x = []
y = []

for line in spectrum_data:
    parts = line.strip().split('\t')
    if len(parts) >= 2:
        try:
            x_val = float(parts[0])
            y_val = float(parts[1])
            x.append(x_val)
            y.append(y_val)
        except ValueError:
            # Skip lines that don't convert to float
            continue

# Plot the data
plt.plot(x, y, marker='o', label='Spectrum Data')

# Add labels and title
plt.xlabel('X Axis (e.g., Energy)')
plt.ylabel('Y Axis (e.g., Counts)')
plt.title('Spectrum Plot')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


''' GET DENSITIES FOR EACH MEDIUM, SAVE THEM IN A TEXT-FILE '''
################################################

# Loading medium data
with open(folder_path2, "r") as file:
    media_data_all = file.readlines()  

# Adjust starting line based on `child_phantom`
media_data = media_data_all[1:] if child_phantom else media_data_all[3:]

# Extract first column (numeric IDs)
media_numbers = [line.strip().split()[0] for line in media_data]

processed_media = []
remaining_columns = []  # Store numeric columns after text

for line in media_data:
    parts = line.strip().split()  # Split by spaces
    first_col = parts[0]  # First column (numeric)

    # Identify where the second column (text) ends
    for i in range(1, len(parts)):
        if parts[i].replace('.', '', 1).isdigit():  # First numeric value after text
            second_col = " ".join(parts[1:i])  # Extract text
            numeric_cols = list(map(float, parts[i:]))  # Convert remaining to float
            break  # Stop once we find the first numeric value

    # Store extracted values
    processed_media.append([first_col, second_col] + numeric_cols)
    remaining_columns.append(numeric_cols)  # Store just numeric values

# Print results for debugging
#print(media_numbers)  # First column
#print(processed_media)  # Full extracted data
#print(remaining_columns)  # Only numeric data after text

# Print first 5 rows to check structure
#for i, row in enumerate(processed_media[:5]):
 #   print(f"Row {i+1}: {row}")

# Loading organ data
with open(folder_path3, "r") as file:
    # Read all lines, but only take the first two
    organ_data_all = file.readlines()[4:]  # Skip first 1 header rows

# Getting the densities for each medium
rho_media_raw = [line.strip().split()[-2:] for line in organ_data_all]
#print(rho_media_raw)  # Check extracted data

organ_numbers = [line.strip().split()[0] for line in organ_data_all]
#print(organ_numbers)

organ_number_vs_medium = [line.strip().split()[-2] for line in organ_data_all]
#print(organ_number_vs_medium)

# Ensure both inputs are NumPy arrays
organ_numbers = np.array(organ_numbers)
organ_number_vs_medium = np.array(organ_number_vs_medium)

# Build a mask: keep values that are not 'n/a' or np.nan
mask = [
    val != "n/a" and not (isinstance(val, float) and np.isnan(val))
    for val in organ_number_vs_medium
]

# Apply mask
filtered_organ_numbers = organ_numbers[mask]
filtered_medium_raw = organ_number_vs_medium[mask]

# Convert the filtered values to integers
filtered_medium = np.array([int(float(val)) for val in filtered_medium_raw])

# Stack and print
organ_number_medium_number = np.column_stack((filtered_organ_numbers, filtered_medium)).astype(int)
#print(organ_number_medium_number)

rho_media = {}
for medium, density in rho_media_raw:
    if density != 'n/a':
        rho_media[medium] = float(density)  # Convert density to float, but keep medium as string

# Sort medium numbers (strings) and retrieve the densities in the correct order
sorted_rho_media = [rho_media[medium] for medium in sorted(rho_media.keys(), key=lambda x: float(x))]
#print(sorted_rho_media)

filename1 = "input_to_density.txt" # File name of the Input To Density Conversion file
scaled_rho = [x * 1000 for x in sorted_rho_media]
rho_final = scaled_rho + [0.001205 * 1000]
int_rho = [int(x) for x in rho_final]
int_media_numbers = [int(x) for x in media_numbers]
int_media_numbers_final = int_media_numbers + [len(media_numbers) + 1]

if with_table == True:
    number_of_tablemat = len(table_densities)
    
    for i in range(number_of_tablemat):
        int_media_numbers_final = int_media_numbers_final + [len(media_numbers) + (i+2)]
    
    for i in range(number_of_tablemat):
        new_table_densities = [int(x * 1000) for x in table_densities]
        int_rho = int_rho + [new_table_densities[i]]

if CTDI_phantom == True:
    int_media_numbers_final = int_media_numbers_final + [len(media_numbers) + len(table_densities)+2]
    int_rho.append(int(1.19*1000))
    int_media_numbers_final = int_media_numbers_final + [len(media_numbers) + len(table_densities)+3]
    int_rho.append(int(1.19*1000))

#print(int_media_numbers_final)

# Open a file in write mode
with open(path + filename1, "w") as file:
    for num1, num2 in zip(int_media_numbers_final, int_rho):
        file.write(f"{num1},{num2}\n")  # Writing numbers with a comma separator

print("Density file written successfully!")

''' GET MATERIAL NUMBERS FOR EACH MEDIUM NAME, SAVE THEM IN A TEXT-FILE '''
################################################

media_names = [
    re.sub(r'\((?!compressed_lungs\))', '', row[1])  # Remove '(' unless followed by 'compressed_lungs'
      .replace(')', '')  # Remove all ')'
      .replace('(', '')  # Remove all '('
      .replace(', ', '_')  # Replace commas with underscores
      .replace(' ', '_')  # Replace spaces with underscores
      .replace('.', '')  # Remove periods
      .replace('-', '')  # Remove hyphens
      .replace('compressed_lungs', '(compressed_lungs)')  # Ensure "compressed lungs" has parentheses
      + '_(ICRU110' + sex_of_phantom + ')'  # Append '_(ICRU110Female)' with parentheses
    for row in processed_media
]

media_names_final = [name.replace('__', '_') for name in media_names]
media_names_final_final = media_names_final + ['Air']

if with_table == True:
    number_of_tablemat = len(table_densities)

    for i in range(number_of_tablemat):
        media_names_final_final = media_names_final_final + [material_table[i]]

if CTDI_phantom == True:
    media_names_final_final.append('PMMA')
    media_names_final_final.append('PMMA')

#print(media_names_final_final)

# Open a file in write mode
with open(path + "material_to_HU.txt", "w") as file:
    for num1, num2 in zip(media_names_final_final, int_media_numbers_final):
        file.write(f"{num1}, {num2}\n")  # Writing numbers with a comma separator

print("Material file written successfully!")

''' MAKE THE COLLIMATION FILE AND SAVE AS A TEXT-FILE '''
#####################################################

if symmetric_collimation == False:
    FLo = -((total_beam_collimation[0] / 2) + distance_from_middlepoint_fov)
    FHi = (total_beam_collimation[0] / 2) - distance_from_middlepoint_fov
    ZLo = -((total_beam_collimation[1] / 2) + distance_from_middlepoint_fov)
    ZHi = (total_beam_collimation[1] / 2) - distance_from_middlepoint_fov

    # Open a file in write mode
    with open(path + "asymmetric_collimation.txt", "w") as file:
        # Write the header lines
        file.write("Asymmetric\n")
        file.write("# FanLo, FanHi, ZLo, ZHi [mm]\n")
        # Convert the numerical values to strings and then write them
        file.write(str(FLo) + ", " + str(FHi) + ", " + str(ZLo) + ", " + str(ZHi))

    print("Collimation file written successfully!")

''' MAKE THE FILTER FILE AND SAVE AS A TEXT-FILE '''
#####################################################

if type_filter == 'Flat':
    d0 = thickness_filter[0]
    all_d = []
    half_length = CTDI_diameter / 2
    n = 25
    for i in range(np.floor(n).astype(int)):
        d = d0 / np.cos(((i+1)*angular_increment)*(np.pi/180))
        all_d.append(d)

    # Open a file in write mode
    with open(path + "flat_filter.txt", "w") as file:
        # Write the header lines
        file.write("Filter\n")
        file.write("# Name of the material\n")
        file.write(filter_materials[0] + '\n')
        file.write("# Density of the material (g/cm^3)\n")
        file.write(str(density_filter[0]) + '\n')
        file.write("# Angular increment (°)\n")
        file.write(str(angular_increment) + '\n')
        file.write("# Thickness array (mm)\n")
        file.write(str(d0) + '\n')
        for row in all_d:
            file.write(str(row) + '\n')
        
        if len(filter_materials) >= 2:
            d0 = thickness_filter[1]
            all_d = []
            n = 25
            for i in range(np.floor(n).astype(int)):
                d = d0 / np.cos(((i+1)*angular_increment)*(np.pi/180))
                all_d.append(d)

            # Write the header lines
            file.write(" \n")
            file.write("Filter\n")
            file.write("# Name of the material\n")
            file.write(filter_materials[1] + '\n')
            file.write("# Density of the material (g/cm^3)\n")
            file.write(str(density_filter[1]) + '\n')
            file.write("# Angular increment (°)\n")
            file.write(str(angular_increment) + '\n')
            file.write("# Thickness array (mm)\n")
            file.write(str(d0) + '\n')
            for row in all_d:
                file.write(str(row) + '\n')

elif type_filter == 'Bowtie':
    d0 = thickness_filter[0] 
    all_d_plot = [d0]
    all_d = []
    x_values = [0]  # Distances from the center
    half_length = CTDI_diameter / 2
    n = 100
    for i in range(n):
        angle_deg = (i + 1) * angular_increment
        angle_rad = angle_deg * (np.pi / 180)
    
        # Calculate distance from center
        x = (distance_source_isocenter * np.tan(angle_rad))
        x_values.append(x)
    
        # Compute d value
        d = 1*(((CTDI_diameter - 2 * np.sqrt(half_length**2 - (distance_source_isocenter * np.sin(angle_rad))**2)) \
            + ((2 * np.log(np.cos(angle_rad))) / 0.2285) ) * 1 ) + d0

        all_d.append(d)
        all_d_plot.append(d)

    # Plot the data
    plt.plot(x_values, all_d_plot, marker='o', markersize=4, linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('Length of filter (mm)', fontsize=14)
    plt.ylabel('Thickness of filter (mm)', fontsize=14)
    plt.title('Bowtie Filter Thickness vs Length', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
    
    # Open a file in write mode
    with open(path + "bowtie_filter.txt", "w") as file:
        # Write the header lines
        file.write("Filter\n")
        file.write("# Name of the material\n")
        file.write("PMMA" + '\n')
        file.write("# Density of the material (g/cm^3)\n")
        file.write("1.190" + '\n')
        file.write("# Angular increment (°)\n")
        file.write(str(angular_increment) + '\n')
        file.write("# Thickness array (mm)\n")
        file.write(str(d0) + '\n')
        for row in all_d:
            file.write(str(row) + '\n')

print("Filter file written successfully!")

''' DETERMINE THE TUBE CURRENT MODULATION VALUES, SAVE THEM IN A TEXT-FILE '''
################################################

if tube_current_modulation == True:
    # Load the Excel file
    tbm_data = pd.read_csv('H:\ImpactMC_CBCT\child_abdomen_tube_current.csv', delimiter=';')
    #print(tbm_data)

    # Replace four white spaces with a period in all string entries of the DataFrame
    tbm_data_processed = tbm_data.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)

    #print(tbm_data_processed)

    angle_points = tbm_data_processed.iloc[:, 1].astype(float).to_numpy()
    kerma = tbm_data_processed.iloc[:, 2].astype(float).to_numpy()
    #print(kerma)

    # Split the array into 36 even sections
    sections = np.array_split(kerma, nr_projection_angles*number_of_rotations)

    # Calculate the average for each section
    averages = np.array([np.mean(section) for section in sections])
    #print(averages)

    # Compute the integral over the kerma vs angles
    area = np.sum(averages)
    #print(area)

    # Normalize the averages
    timestep = 1.0 / 360
    normalized_values = ((averages / area) * 100) * 180
    #print(normalized_values) 
    summed = np.sum(normalized_values)
    #print(summed)

    # Open a file in write mode
    with open(path + "tube_current_modulation.txt", "w") as file:
        number = int(nr_projection_angles * number_of_rotations)
        # Loop through the selected values and write them to the file
        for i in range(number):
            # Ensure we're only looping through the selected values (middle half or outside values)
            file.write(str(normalized_values[i]) + '\n')

    
    print('Tube Current Modulation file written succesfully!')


''' DEFINE THE CHILD PHANTOM MATERIALS, SAVE IN AN XML FILE '''
################################################

if child_phantom == True :
    # Function to create an XML structure for user-defined materials
    def create_material_xml(materials_data, output_file, with_table):
        # Create the root element
        root = ET.Element("UserDefinedMaterials")
        
        # Loop through each material data provided
        for material in materials_data:
            # Create a <Material> element with type attribute
            material_elem = ET.SubElement(root, "Material", type="MassFraction")
            
            # Add the <Name> element for the material name
            name_elem = ET.SubElement(material_elem, "Name")
            name_elem.text = material["name"]
            
            # Optionally, add the <Density> element if present
            if "density" in material:
                density_elem = ET.SubElement(material_elem, "Density", uom="g/mm^3")
                density_elem.text = str(material["density"])
            
            # Add <Component> elements for each component in the material
            for component in material["components"]:
                component_elem = ET.SubElement(material_elem, "Component")
                
                # Add the <Name> of the component
                component_name_elem = ET.SubElement(component_elem, "Name")
                component_name_elem.text = component["name"]
                
                # Add the <MassFraction> of the component
                mass_fraction_elem = ET.SubElement(component_elem, "MassFraction")
                mass_fraction_elem.text = str(component["mass_fraction"])

        # Create a <Material> element with type attribute
        material_elem = ET.SubElement(root, "Material", type="MassFraction")
            
        # Add the <Name> element for the material name
        name_elem = ET.SubElement(material_elem, "Name")
        name_elem.text = 'Air'
            
        # Optionally, add the <Density> element if present
        density_elem = ET.SubElement(material_elem, "Density", uom="g/mm^3")
        density_elem.text = str(0.001)

        component_elem = ET.SubElement(material_elem, "Component")
                
        # Add the <Name> of the component
        component_name_elem = ET.SubElement(component_elem, "Name")
        component_name_elem.text = 'N'
                
        # Add the <MassFraction> of the component
        mass_fraction_elem = ET.SubElement(component_elem, "MassFraction")
        mass_fraction_elem.text = str(0.8) 
        
        component_elem = ET.SubElement(material_elem, "Component")
                
        # Add the <Name> of the component
        component_name_elem = ET.SubElement(component_elem, "Name")
        component_name_elem.text = 'O'
                
        # Add the <MassFraction> of the component
        mass_fraction_elem = ET.SubElement(component_elem, "MassFraction")
        mass_fraction_elem.text = str(0.2) 

        if with_table == True:
            for material, density in zip(material_table, table_densities):
            # Create a <Material> element with type attribute
                material_elem = ET.SubElement(root, "Material", type="ElementalComposition")

                # Add the <Name> element for the material name
                name_elem = ET.SubElement(material_elem, "Name")
                name_elem.text = material
                
                # Optionally, add the <Density> element if present
                density_elem = ET.SubElement(material_elem, "Density", uom="g/mm^3")
                density_elem.text = str(density)
                
                if name_elem.text == 'C':
                    # Add <Component> elements for each component in the material
                    component_elem = ET.SubElement(material_elem, "Component")
                                
                    # Add the <Name> of the component
                    component_name_elem = ET.SubElement(component_elem, "Name")
                    component_name_elem.text = name_atoms_table
                                
                    # Add the <MassFraction> of the component
                    nratoms_elem = ET.SubElement(component_elem, "NumberOfAtoms")
                    nratoms_elem.text = str(nr_atoms_table)
                else:
                    for i in nr_atoms_table2:
                        # Add <Component> elements for each component in the material
                        component_elem = ET.SubElement(material_elem, "Component")
                                    
                        # Add the <Name> of the component
                        component_name_elem = ET.SubElement(component_elem, "Name")
                        component_name_elem.text = str(name_atoms_table2[i-1])
                                    
                        # Add the <MassFraction> of the component
                        nratoms_elem = ET.SubElement(component_elem, "NumberOfAtoms")
                        nratoms_elem.text = str(nr_atoms_table2[i-1])

        
        # Create the tree from the root and write it to the output file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)


    # Prepare the material data using processed_media
    materials_data = []

    for i, row in enumerate(processed_media):
        material_name = media_names[i]  # Get the material name from the media_names array
        
        # Get atomic percentages (from the 3rd to 15th columns) and divide by 100
        atomic_percentages = {
            "H": row[2] / 100,  # H in the 3rd column
            "C": row[3] / 100,  # C in the 4th column
            "N": row[4] / 100,  # N in the 5th column
            "O": row[5] / 100,  # O in the 6th column
            "Na": row[6] / 100,  # Na in the 7th column
            "Mg": row[7] / 100,  # Mg in the 8th column
            "P": row[8] / 100,  # P in the 9th column
            "S": row[9] / 100,  # S in the 10th column
            "Cl": row[10] / 100,  # Cl in the 11th column
            "K": row[11] / 100,  # K in the 12th column
            "Ca": row[12] / 100,  # Ca in the 13th column
            "Fe": row[13] / 100,  # Fe in the 14th column
            "I": row[14] / 100,  # I in the 15th column
        }

        # Extract the density from the 16th column (after I)
        density = None
        try:
            density = float(row[15])  # Convert the value to float
        except ValueError:
            pass  # If there's no valid density value, leave it as None

        # Calculate the sum of atomic percentages
        total_percent = sum(atomic_percentages.values())
        
        # If the total doesn't sum to exactly 1.0, adjust the values proportionally
        if total_percent != 1.0:
            # Scale each component to make the sum equal to 1.0
            scaling_factor = 1.0 / total_percent
            atomic_percentages = {element: percentage * scaling_factor for element, percentage in atomic_percentages.items()}

        # Create a list of components for the material
        components = [{"name": element, "mass_fraction": atomic_percentages[element]} for element in atomic_percentages]

        # Store the material data
        material_data = {
            "name": material_name,
            "density": density,  # Add the density to the material data
            "components": components
        }

        # Append the material data to the list
        materials_data.append(material_data)

    # Path to the output XML file
    output_file = path + "user_defined_materials.xml"

    # Call the function to create the XML
    create_material_xml(materials_data, output_file, with_table)

    print(f"XML file written successfully!")


''' CHANGE THE ORGAN NUMBERS IN THE PHANTOM TO THEIR RESPECTIVE MATERIAL NUMBERS '''
################################################

# Create an array of [organ_number, medium_number] pairs
organmedium_array = np.array(
    [
        [organ_numbers[i], organ_number_vs_medium[i]]
        for i in range(len(organ_numbers))
        if organ_numbers[i] != "n/a" and organ_number_vs_medium[i] != "n/a"
    ], 
    dtype=float
)
#print(organmedium_array)

# Step 1: Read the file line by line
data = []

#with open(folder_path4, 'r') as file:
    #data = file.readlines()[1:]  # Skip first 1 header rows
if child_phantom == True:
    number = 18
else: 
    number = 16

with open(folder_path4, 'r') as file:
    for line in file:
        # Split the line into columns
        columns = list(map(float, line.split()))
        
        # If the row has fewer than 16 columns, pad with zeros
        if len(columns) < number:
            columns.extend([0] * (number - len(columns)))  # Pad with zeros
        
        # Only append rows with exactly 16 columns
        if len(columns) == number:
            data.append(columns)

# Step 2: Convert the list of rows into a NumPy array
phantom_volume = np.array(data)

# Step 3: Check size and reshape
#print("Total values in file:", phantom_volume.size)
expected_size = int(width * height * depth)  # Expected size of the 3D array
#print("Expected values:", expected_size)

# Step 4: Handle the size difference (if any)
difference = phantom_volume.size - expected_size
#print(f"Difference in size: {difference}")

#print("Volume shape (Z, Y, X):", phantom_volume.shape)

# If the size matches, reshape the array
if phantom_volume.size == expected_size:
    if sex_of_phantom == 'Female':
        phantom_volume_3d = phantom_volume.reshape((int(depth), int(height), int(width)))
    else:
        phantom_volume_3d = phantom_volume.reshape((int(depth), int(height), int(width)))
    print("Reshaped array successfully!")
else:
    # If there's a mismatch, trim the excess elements
    phantom_volume = phantom_volume.flatten()[:expected_size]  # Flatten and trim
    if sex_of_phantom == 'Female':
        phantom_volume_3d = phantom_volume.reshape((int(depth), int(height), int(width)))
    else:
        phantom_volume_3d = phantom_volume.reshape((int(depth), int(height), int(width)))
    print("Reshaped array after trimming excess data!")

# Define HU values explicitly
mediumname_air = len(media_names) + 1
mediumname_skin = 27  # New value for 141

# Create a dictionary mapping medium numbers to their HU values
organ_to_medium = {int(organ): medium for organ, medium in organmedium_array}

# Replace each organ number with its HU value in the phantom volume
phantom_volume_final = np.vectorize(lambda x: organ_to_medium.get(int(x), mediumname_air))(phantom_volume_3d)

# After the volume is modified, replace specific values
phantom_volume_final = np.where(phantom_volume_final == 0, mediumname_air, phantom_volume_final)  # Replace 0 with 54
phantom_volume_final = np.where(phantom_volume_final == 141, mediumname_skin, phantom_volume_final)  # Replace 141 with 27


''' ADD A CTDI PHANTOM, RING OR MEASURE IN AIR '''
#####################################################

if measure_in_air == True:
    phantom_volume_final[:] = 54  # Replace all values with 54

if ring == True:
    # Create an empty ring in the middle axial slice
    middle_slice = int(depth / 2)  # Get the middle slice index
    #height, width = phantom_volume_final[1], phantom_volume_final[2]
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # Define the ring parameters
    center = (height // 2, width // 2)
    radius_outer = min(height, width) // 4  # Outer radius
    radius_inner = radius_outer - 1  # Inner radius for 1-pixel thickness

    # Create the ring mask
    distance_from_center = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    ring_mask = (distance_from_center >= radius_inner) & (distance_from_center <= radius_outer)

    # Apply the ring to the middle slice
    phantom_volume_final[middle_slice][ring_mask] = 2

if CTDI_phantom == True:
    # Cylinder parameters
    length_voxels = int(145 / slice_thickness)
    radius_voxels = int((CTDI_diameter/2) / voxel_size_xy[0])
    radius_small = ((13.1 / voxel_size_xy[0])/2)  
    side_distance = radius_small + (3/voxel_size_xy[0])  # in mm
    margin = 50  # Extra space around the cylinder (in voxels)

    # Compute required volume size
    depth = length_voxels + 2 * margin  # Ensure space at both ends
    height = 2 * (radius_voxels + margin)  # Enough space for the cylinder in X
    width = int(width_table/voxel_size_xy[0]) + 2*margin  # Enough space for the cylinder in Y

    # Create the volume
    volume = np.zeros((depth, height, width))
    volume[:] = 54

    # Cylinder center (XY plane)
    center_x, center_y = height // 2, width // 2  

    # Cylinder Z range (keeping space at the ends)
    start_z = margin  # Start after margin
    end_z = start_z + length_voxels  # Ensure correct length

    # Generate the cylinder along the Z-axis
    for z in range(start_z, end_z):  
        for x in range(height):
            for y in range(width):
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius_voxels ** 2:
                    volume[z, x, y] = len(media_numbers) + len(table_densities) + 2  # Set voxel inside the cylinder to 1

    # Generate the small phantom cylinder (inside the large one)
    for z in range(start_z, end_z):  
        for x in range(height):
            for y in range(width):
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius_small ** 2:
                    volume[z, x, y] = len(media_numbers) + len(table_densities) + 3  # Small phantom cylinder (different value for distinction)

    # Add 4 smaller cylinders inside the large cylinder, at the sides
    # Place the center of the small cylinders 8 mm from the edge of the large cylinder
    top_center_y = center_y + (radius_voxels - side_distance)  # 8 mm from the top of the large cylinder
    bottom_center_y = center_y - (radius_voxels - side_distance)  # 8 mm from the bottom of the large cylinder
    left_center_x = center_x - (radius_voxels - side_distance)  # 8 mm from the left of the large cylinder's edge
    right_center_x = center_x + (radius_voxels - side_distance)  # 8 mm from the right of the large cylinder's edge

    # Correct positions for small side cylinders (inside the large cylinder)
    positions = [
        (start_z, center_x, top_center_y),    # Top cylinder inside large cylinder
        (start_z, center_x, bottom_center_y),  # Bottom cylinder inside large cylinder
        (start_z, left_center_x, center_y),    # Left cylinder inside large cylinder
        (start_z, right_center_x, center_y)      # Right cylinder inside large cylinder
    ]

    # Generate the small side cylinders inside the large cylinder
    for pos in positions:
        pos_z, pos_x, pos_y = pos
        
        # Ensure that the Z-index does not go out of bounds
        pos_z = max(margin, min(depth - 1, int(pos_z)))  # Keep within the valid range for Z-axis
        
        # Generate the small side cylinders inside the large cylinder
        for z in range(pos_z, pos_z + length_voxels):
            if z < 0 or z >= depth:
                continue  # Skip if the index is out of bounds
            for x in range(height):
                for y in range(width):
                    if (x - pos_x) ** 2 + (y - pos_y) ** 2 <= radius_small ** 2:
                        if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius_voxels ** 2:  # Make sure it's inside the large cylinder
                            volume[z, x, y] = len(media_numbers) + len(table_densities) + 3  # Small side cylinders (inside the large cylinder)

    phantom_volume_final = volume

''' ADD A TABLE '''
#####################################################

if with_table == True :
    phantom_no_table = phantom_volume_final
    #print(phantom_no_table.shape)    

    widthtable_pixels = width_table / voxel_size_xy[0]
    #print(widthtable_pixels)
    if CTDI_phantom == False:
        difference = int(width-widthtable_pixels)
        #print(difference)
    else: 
        difference = margin
    
    # Create a new array with the correct shape (increase height by thickness_table)
    thickness_table1 = round(thickness_table[0] / voxel_size_xy[0])
    thickness_table2 = round(thickness_table[1] / voxel_size_xy[0]) + 1 #because otherwise it's 0
    thickness_table3 = round(thickness_table[2] / voxel_size_xy[0])
    total_thickness = thickness_table1 + thickness_table2 + thickness_table3
    #print(total_thickness)

    phantom_with_table = np.zeros((int(depth), int(height + total_thickness), int(width)))
    #print(phantom_with_table.shape)

    if table_back == False:
        # Set the first rows to some value (e.g., media_numbers length + 2)
        phantom_with_table[:, :thickness_table1, :] = len(media_numbers) + 2
        phantom_with_table[:, thickness_table1:thickness_table1 + thickness_table2, :] = len(media_numbers) + 3
        phantom_with_table[:, thickness_table1 + thickness_table2:thickness_table1 + thickness_table2 + thickness_table3, :] = len(media_numbers) + 4

        # Set the first and last 15 pixels in each row to 0
        if difference > 0:
            phantom_with_table[:, :total_thickness, :difference] = len(media_numbers) + 1  # First 15 pixels
            phantom_with_table[:, :total_thickness, -difference:] = len(media_numbers) + 1  # Last 15 pixels

        # Copy the original array into the new array starting after table
        phantom_with_table[:, total_thickness:, :] = phantom_no_table

        # Remove rows so table fits perfectly
        phantom_with_table = np.delete(phantom_with_table, np.s_[((int(height)+total_thickness)-fit_end):((int(height)+total_thickness)-fit_start)], axis=1)

    else: 
        # Get the total number of rows in the volume
        num_rows = phantom_with_table.shape[1]

        # Set the outermost layer (thickness 1) at the end
        phantom_with_table[:, num_rows - thickness_table1:, :] = len(media_numbers) + 2
        phantom_with_table[:, num_rows - (thickness_table1 + thickness_table2):num_rows - thickness_table1, :] = len(media_numbers) + 3
        phantom_with_table[:, num_rows - (thickness_table1 + thickness_table2 + thickness_table3):num_rows - (thickness_table1 + thickness_table2), :] = len(media_numbers) + 4

        # Set the first and last 15 pixels in each row to len(media_numbers) + 1
        if difference > 0:
            phantom_with_table[:, num_rows - total_thickness:, :difference] = len(media_numbers) + 1  # First 15 pixels
            phantom_with_table[:, num_rows - total_thickness:, -difference:] = len(media_numbers) + 1  # Last 15 pixels

        # Copy the original array into the new array before the table starts
        phantom_with_table[:, :num_rows - total_thickness, :] = phantom_no_table  

        # Remove rows so table fits perfectly
        #phantom_with_table = np.delete(phantom_with_table, np.s_[fit_start:fit_end], axis=1)      

    #print(phantom_with_table.shape) 
    #print(phantom_no_table.shape)


''' PLOT THE SLICES '''
#####################################################

# Use the new material-mapped volume in the visualization
if with_table == True:
    volume = phantom_with_table
else:
    volume = phantom_volume_final

volume = volume.astype(np.float32)  # Convert to float first
volume = np.round(volume).astype(np.int16)  # Round and convert to int

# Manually define 60 distinct colors
colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", 
    "#bcbd22", "#17becf", "#ff00ff", "#f0f0f0", "#00ff00", "#ff0000", "#0000ff", "#ff6600",
    "#ff3366", "#003366", "#9966cc", "#66ccff", "#ff99ff", "#33cc33", "#ffcc00", "#ccff33", 
    "#ff6600", "#ff3300", "#3399ff", "#66cc33", "#ff3366", "#6699cc", "#ccccff", "#ffcccc", 
    "#ff0066", "#ffcc66", "#cc3333", "#3399cc", "#336699", "#ff9900", "#66ff66", "#ff6699", 
    "#66ccff", "#3399ff", "#cc33cc", "#cc6633", "#ff9933", "#ff3399", "#6699ff", "#ccff66", 
    "#9966ff", "#cccc66", "#ff00cc", "#ff6600", "#ff0033", "#0033ff", "#3366ff",
    
    # 7 new colors
    "#33ffcc", "#9900cc", "#ffcc99", "#663300", "#99ff00", "#cc0099", "#009999"
]

# Create a ListedColormap from these colors
custom_cmap = ListedColormap(colors)

# Define initial slice indices
axial_idx = volume.shape[0] // 2  # Middle slice for axial
coronal_idx = volume.shape[1] // 2  # Middle slice for coronal
sagittal_idx = volume.shape[2] // 2  # Middle slice for sagittal

# Create figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Update slice display with unique colors
axial_img = axes[0].imshow(volume[axial_idx, :, :], cmap=custom_cmap)  
coronal_img = axes[1].imshow(volume[:, coronal_idx, :], cmap=custom_cmap)
sagittal_img = axes[2].imshow(volume[:, :, sagittal_idx], cmap=custom_cmap)

# Titles
axes[0].set_title(f"Axial Slice {axial_idx}", fontsize=16, fontweight='bold')
axes[1].set_title(f"Coronal Slice {coronal_idx}", fontsize=16, fontweight='bold')
axes[2].set_title(f"Sagittal Slice {sagittal_idx}", fontsize=16, fontweight='bold')

axes[0].set_xlabel('X-Axis', fontsize=14)
axes[0].set_ylabel('Y-Axis', fontsize=14)
axes[1].set_xlabel('X-Axis', fontsize=14)
axes[1].set_ylabel('Z-Axis', fontsize=14)
axes[2].set_xlabel('Y-Axis', fontsize=14)
axes[2].set_ylabel('Z-Axis', fontsize=14)

# Remove axis labels
#for ax in axes:
    #ax.axis("off")

# Add sliders to control slice index
axcolor = 'lightgoldenrodyellow'
axial_slider_ax = plt.axes([0.2, 0.02, 0.65, 0.02], facecolor=axcolor)
coronal_slider_ax = plt.axes([0.2, 0.05, 0.65, 0.02], facecolor=axcolor)
sagittal_slider_ax = plt.axes([0.2, 0.08, 0.65, 0.02], facecolor=axcolor)

axial_slider = Slider(axial_slider_ax, 'Axial', 0, volume.shape[0] - 1, valinit=axial_idx, valstep=1)
coronal_slider = Slider(coronal_slider_ax, 'Coronal', 0, volume.shape[1] - 1, valinit=coronal_idx, valstep=1)
sagittal_slider = Slider(sagittal_slider_ax, 'Sagittal', 0, volume.shape[2] - 1, valinit=sagittal_idx, valstep=1)

# Function to update slices
def update(val):
    axial_idx = int(axial_slider.val)
    coronal_idx = int(coronal_slider.val)
    sagittal_idx = int(sagittal_slider.val)

    axial_img.set_data(volume[axial_idx, :, :])
    coronal_img.set_data(volume[:, coronal_idx, :])
    sagittal_img.set_data(volume[:, :, sagittal_idx])

    axes[0].set_title(f"Axial Slice {axial_idx}", fontsize=16, fontweight='bold')
    axes[1].set_title(f"Coronal Slice {coronal_idx}", fontsize=16, fontweight='bold')
    axes[2].set_title(f"Sagittal Slice {sagittal_idx}", fontsize=16, fontweight='bold')

    axes[0].set_xlabel('X-Axis', fontsize=14)
    axes[0].set_ylabel('Y-Axis', fontsize=14)
    axes[1].set_xlabel('X-Axis', fontsize=14)
    axes[1].set_ylabel('Z-Axis', fontsize=14)
    axes[2].set_xlabel('Y-Axis', fontsize=14)
    axes[2].set_ylabel('Z-Axis', fontsize=14)

    fig.canvas.draw_idle()

# Connect sliders to update function
axial_slider.on_changed(update)
coronal_slider.on_changed(update)
sagittal_slider.on_changed(update)

# Show the figure
if show_plot == True:
    plt.show()

''' SAVE AS A DICOM FILE '''
################################################

# Assuming phantom_volume_HU is your final 3D NumPy array with HU values
if with_table == True:
    phantom_volume_final_final = np.nan_to_num(phantom_with_table, nan=0, posinf=0, neginf=0)
else:
    phantom_volume_final_final = np.nan_to_num(phantom_volume_final, nan=0, posinf=0, neginf=0)

volume = phantom_volume_final_final.astype(np.int16)  # Ensure data type matches DICOM standard
#print(volume.shape)

# Define output folder for DICOM files
os.makedirs(output_folder, exist_ok=True)

def create_dicom_slice(slice_data, slice_index, voxel_size_xy, slice_thickness):
    ds = pydicom.Dataset()
    
    # Set required DICOM tags
    ds.PatientName = "Anonymous"
    ds.PatientID = "123456"
    ds.Modality = "CT"
    ds.StudyInstanceUID = "1.2.3.4.5.6.7.8.9"
    ds.SeriesInstanceUID = f"1.2.3.4.5.6.7.8.9"
    ds.SOPInstanceUID = f"1.2.3.4.5.6.7.8.9.{slice_index}"
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    
    ds.Rows, ds.Columns = slice_data.shape
    ds.InstanceNumber = slice_index
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # Signed integers
    ds.RescaleIntercept = 0  # Standard for HU scaling
    ds.RescaleSlope = 1
    ds.ImageOrientationPatient = [1,0,0,0,1,0]

    # Assign pixel data
    ds.PixelData = slice_data.tobytes()

    # Add PixelSpacing and SliceThickness
    ds.PixelSpacing = [str(spacing) for spacing in voxel_size_xy]  # in mm, e.g. [0.5, 0.5] for X and Y
    ds.SliceThickness = str(slice_thickness)  # in mm, e.g. 1.0

    # Create FileMetaDataset and set TransferSyntaxUID
    file_meta = pydicom.FileMetaDataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    
    # Assign the FileMetaDataset to the Dataset
    ds.file_meta = file_meta

    return ds

for i in range(volume.shape[0]):
    slice_data = volume[i, :, :]  # Extract the axial slice
    dicom_slice = create_dicom_slice(slice_data, i, voxel_size_xy, slice_thickness)
    dicom_slice.ImagePositionPatient = [0,0,slice_thickness*i]
    dicom_path = os.path.join(output_folder, f"slice_{i:03d}.dcm")
    
    # Save the DICOM file with the correct transfer syntax
    pydicom.dcmwrite(dicom_path, dicom_slice)

print(f"DICOM files saved in: {output_folder}")

