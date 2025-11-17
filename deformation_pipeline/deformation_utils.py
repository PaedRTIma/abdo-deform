import os
import random
import SimpleITK as sitk
from platipy.imaging.registration.utils import apply_transform
from platipy.imaging.generation.dvf import generate_field_expand
from scipy.stats import truncnorm
import numpy as np
from scipy.interpolate import interp1d
import nibabel as nib
import pandas as pd


def generate_calibration_curve(subject_id, ct, structures_to_deform, bone_mask, expansion, output_calibration):
    """
    Generate the calibration curve for a given expansion vector.
    """
    # Reset CT and structures at every deformation version
    ct_copy = sitk.Image(ct)
    structures_to_deform_copy = {s: sitk.Image(structures_to_deform[s]) for s in structures_to_deform}
    bone_mask_copy = sitk.Image(bone_mask)

    try:
        print(f'\nApply Expansion: {expansion} mm')
        label_deformed, dvf_transform, dvf_field = generate_field_expand(
            structures_to_deform["BowelGas"], bone_mask=bone_mask_copy, expand=expansion, gaussian_smooth=5)

        # Deform specified surrounding structures and CT using the above defined transformation
        deformed_structures = apply_transform_all(
            structures_to_deform_copy, dvf_transform)

        deformed_ct = apply_transform(
            ct_copy, transform=dvf_transform, default_value=-1000, interpolator=sitk.sitkLinear)

        # Save deformed CT 
        print(f'Writing {subject_id}_CT_e{expansion}.nii.gz ...')
        sitk.WriteImage(deformed_ct, os.path.join(
           output_calibration, f"{subject_id}_CT_e{expansion}.nii.gz"))

        # Save deformed Bowel Gas mask
        for structure in deformed_structures:
            print(f'Writing {subject_id}_{structure}_e{expansion}.nii.gz ...')
            sitk.WriteImage(deformed_structures[structure], os.path.join(
                output_calibration, f"{subject_id}_{structure}_e{expansion}.nii.gz"))
           
            # Load the deformed bowel gas mask to calculate the volume
            nii_img = nib.load(os.path.join(
                output_calibration, f"{subject_id}_{structure}_e{expansion}.nii.gz"))
            header_info = nii_img.header
            voxel_dimensions = header_info['pixdim'][1:4]
            nii_data = nii_img.get_fdata()
            volume = calculate_volume_seg(nii_data, voxel_dimensions)
            print(f'=> Gas volume: {volume} ml')

    except Exception as e:
        print(f"An error occurred for expansion {expansion}: {str(e)}")
        #continue

    return volume

def get_random_from_truncated_normal(mean=0, scale=1, bounds=[-5,5], sample_size=1):
    # Choose random number from a normal (Gaussian) distribution within a range
    lower_bound, upper_bound = bounds
    a = (lower_bound - mean) / scale
    b = (upper_bound - mean) / scale
    random_number = truncnorm(a, b, loc=mean, scale=scale)
    return random_number.rvs(size=sample_size)[0]

def get_random_from_uniform(bounds = [-5,5], sample_size=1):
    lower_bound, upper_bound = bounds
    random_number = np.random.uniform(low=lower_bound, high=upper_bound, size=sample_size)
    return random_number[0]



def make_deformation_choice(organ):
    """
    Function that flags whether deformation to a specific structure based on random parameters from literature distributions should be applied or not.

    Returns:
        - x_shift (float): The shift in the x-direction (+/-) (superior/inferior) if deformation is applied.
        - y_shift (float): The shift in the y-direction (+/-) (posterior/anterior) if deformation is applied.
        - z_shift (float): The shift in the z-direction (+/-) (left/right) if deformation is applied.
    """

    apply_deformation_statement = random.choice([True, False])
    print(f'Apply deformation?  {apply_deformation_statement}')

    # Initialize shifts to 0
    x_shift, y_shift, z_shift = 0, 0, 0

    if apply_deformation_statement:
        # Mean and range inter-fraction organ motion (Guirreiro et al.)
        if organ == 'Liver':
            # Organ-specific median and range values for each direction
            median_shifts = {'x': 0.2, 'y': -0.1, 'z': -1.2}
            range_shifts = {'x': [-8.1, 9.0], 'y': [-3.3, 4.8], 'z': [-4.0, 4.3]}

        elif organ == 'Spleen':
            # Organ-specific median and range values for each direction
            median_shifts = {'x': 1, 'y': -0.2, 'z': -0.5}
            range_shifts = {'x': [-9.6, 9.1], 'y': [-4.0, 4.9], 'z': [-3.5, 2.7]}

        elif organ == 'Kidney Right' or organ == 'Kidney Left': # Contralateral kidney???? 
            # Organ-specific median and range values for each direction
            median_shifts = {'x': 0.2, 'y': 0.4, 'z': -0.1}
            range_shifts = {'x': [-6.3, 5.6], 'y': [-1.6, 3.9], 'z': [-3.7, 3.7]}

        #direction = random.choice(['x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'])
        direction = 'xyz'
        print(f'Direction:  {direction}')

        # Set shifts based on the chosen direction
        if 'x' in direction:
            #x_shift = get_random_from_truncated_normal(mean=median_shifts['x'], scale=2, bounds=range_shifts['x'])
            x_shift = get_random_from_uniform(bounds=range_shifts['x'])
        if 'y' in direction:
            #y_shift = get_random_from_truncated_normal(mean=median_shifts['y'], scale=1, bounds=range_shifts['y'])
            y_shift = get_random_from_uniform(bounds=range_shifts['y'])
        if 'z' in direction:
            #z_shift = get_random_from_truncated_normal(mean=median_shifts['z'], scale=1, bounds=range_shifts['z'])
            z_shift = get_random_from_uniform(bounds=range_shifts['z'])

        print(f'Vector shift: x={x_shift:.2f} y={y_shift:.2f} z={z_shift:.2f}')

    # Return the shifts and the apply_deformation flag
    return x_shift, y_shift, z_shift


def apply_transform_all(structures_dictionary, dvf_transform):
    """
    Deform a dictionary of structures using a given transformation.

    Parameters:
    - structures_dictionary (dict): A dictionary containing SimpleITK images representing different anatomical structures.
    - dvf_transform (sitk.DisplacementFieldTransform): The transformation to be applied.

    Returns:
    - dict: A dictionary containing deformed structures.
    """

    deformed_structures = {}

    for struct in structures_dictionary:
        print(f"Deforming: {struct}")
        deformed_structures[struct] = apply_transform(
            structures_dictionary[struct], transform=dvf_transform, default_value=0, interpolator=sitk.sitkNearestNeighbor)

    return deformed_structures


def calculate_volume_seg(nii_data, voxel_dimensions):
    # Count the non-zero values in the binary mask
    total_voxels = np.count_nonzero(nii_data == 1)
    total_volume = total_voxels * np.prod(voxel_dimensions)
    return total_volume * 0.001


def load_and_calculate_volume_seg(path_to_mask):
    """
    Calculate gas volume for a specific deformation version of a patient's structure.
    
    Args:
    - path: Path to the directory containing the structure image
    
    Returns:
    - volume: Calculated volume of the gas in the structure (in mL)
    """
    
    nii_img = nib.load(path_to_mask)
    header_info = nii_img.header
    voxel_dimensions = header_info['pixdim'][1:4]
    nii_data = nii_img.get_fdata()
    
    # Assuming you have a function calculate_volume_seg that calculates volume from segmented data
    volume = calculate_volume_seg(nii_data, voxel_dimensions)
    return volume


def percentage_change(new_value, old_value):
    """
    Calculate the percentage change between two values.

    Parameters:
    - new_value (float): The new value.
    - old_value (float): The old value.

    Returns:
    - float: The percentage change. If the result is positive, it is an increase. If the result is negative, it is a decrease.
    """
    return ((new_value - old_value) / old_value) * 100



def calculate_desired_expansion(expansions_mm, percentage_diffs, desired_expansion_perc):
    """
    Interpolate the desired expansion vector in millimeters based on a desired percentage difference.
    
    Args:
    - expansions_mm: List of expansion values in millimeters
    - percentage_diffs: List of corresponding percentage differences
    - desired_expansion_perc: User input desired percentage difference
    
    Returns:
    - desired_expansion_mm: Interpolated expansion in millimeters from distribution
    """

    # Check if desired percentage is within interpolation range
    min_perc = min(percentage_diffs)
    max_perc = max(percentage_diffs)
    
    # If outside the range, adjust to the minimum or maximum
    if desired_expansion_perc < min_perc:
        desired_expansion_perc = min_perc
        print(f"\nDesired percentage is below minimum ({min_perc:.2f}%), adjusting to {min_perc:.2f}%")
    elif desired_expansion_perc > max_perc:
        desired_expansion_perc = max_perc
        print(f"Desired percentage is above maximum ({max_perc:.2f}%), adjusting to {max_perc:.2f}%")
    elif min_perc <= desired_expansion_perc <= max_perc:
        print(f"Desired percentage is within range ({min_perc:.2f}% - {max_perc:.2f}%)")

        # Interpolate to find the expansion vector for the given percentage difference
        interp_func = interp1d(percentage_diffs, expansions_mm, kind='linear')

        # Interpolated expansion vector
        desired_expansion_mm = interp_func(desired_expansion_perc)

    return desired_expansion_mm


def extract_calibration_data(df_calibration):
    """
    Extract expansion vectors and percentage differences from the calibration DataFrame for easier manipulation.
    """

    expansion_vectors_dict = {}
    percentage_diffs_dict = {}

    for _, row_calibration in df_calibration.iterrows():
        subject_id = row_calibration["SubjectID"]
        expansion_vector = row_calibration["ExpansionVector(mm)"]
        percentage_diffs = row_calibration["PercentageDifference(%)"]

        if subject_id in expansion_vectors_dict:
            expansion_vectors_dict[subject_id].append(expansion_vector)
            percentage_diffs_dict[subject_id].append(percentage_diffs)
        else:
            expansion_vectors_dict[subject_id] = [expansion_vector]
            percentage_diffs_dict[subject_id] = [percentage_diffs]

    return expansion_vectors_dict, percentage_diffs_dict
    

def create_expansions_df(percentage_diffs_dict, num_values=4):
    """
    Create a DataFrame of equally spaced expansion values for each patient based on calibration data.
    
    Args:
    - df_calibration: DataFrame with calibration data
    - num_values: Number of values to generate
    
    Returns:
    - df_expansions: DataFrame with equally spaced expansion values
    """

    equally_spaced_dict = {}
    for key, values in percentage_diffs_dict.items():
        min_val = min(values)
        max_val = max(values)
        equally_spaced = np.linspace(min_val, max_val, num_values)
        equally_spaced_dict[key] = equally_spaced

    df_expansions = pd.DataFrame.from_dict(equally_spaced_dict, orient='index')
    df_expansions = df_expansions.rename_axis('SubjectID').reset_index()
    df_expansions.columns = ['SubjectID', 'Expansion1', 'Expansion2', 'Expansion3', 'Expansion4']

    return df_expansions