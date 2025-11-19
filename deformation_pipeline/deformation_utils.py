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
    Apply a given gas expansion vector (in mm) to a subject, save the deformed CT and structures, and return the gas volume
    (in ml) of the deformed BowelGas mask.

    Parameters:
        - subject_id (str): Identifier of the subject (used for filenames).
        - ct (sitk.Image): Original CT image.
        - structures_to_deform (dict[str, sitk.Image]): Dictionary of structures to deform (e.g. {"BowelGas": <image>, ...}).
            Here: must contain the key "BowelGas" for gas expansion.
        - bone_mask (sitk.Image): Bone mask used to constrain the expansion.
        - expansion (float): Magnitude of deformation vector (in mm) for isotropic expansion. -ve vector -> shrinkage; +ve vector -> expansion.
        - output_calibration (str): Folder where deformed images will be saved.

    Returns:
        - volume (float): Gas volume (ml) of the deformed BowelGas structure.
    """

    # Reset CT and structures at every deformation instance
    ct_copy = sitk.Image(ct)
    structures_to_deform_copy = {s: sitk.Image(structures_to_deform[s]) for s in structures_to_deform}
    bone_mask_copy = sitk.Image(bone_mask)

    try:
        print(f'\nApply Expansion: {expansion} mm')
        # Expand BowelGas structure isotropically
        label_deformed, dvf_transform, dvf_field = generate_field_expand(
            structures_to_deform["BowelGas"], bone_mask=bone_mask_copy, expand=expansion, gaussian_smooth=5)

        # Apply transform to all structures to deform (here: BowelGas only) and CT image itself
        deformed_structures = apply_transform_all(
            structures_to_deform_copy, dvf_transform)

        deformed_ct = apply_transform(
            ct_copy, transform=dvf_transform, default_value=-1000, interpolator=sitk.sitkLinear)

        # Save the deformed CT as <subject>_CT_e<expansion>.nii.gz
        print(f'Writing {subject_id}_CT_e{expansion}.nii.gz ...')
        sitk.WriteImage(deformed_ct, os.path.join(
           output_calibration, f"{subject_id}_CT_e{expansion}.nii.gz"))

        # Save the deformed BowelGas mask as <subject>_BowelGas_e<expansion>.nii.gz
        for structure in deformed_structures:
            print(f'Writing {subject_id}_{structure}_e{expansion}.nii.gz ...')
            sitk.WriteImage(deformed_structures[structure], os.path.join(
                output_calibration, f"{subject_id}_{structure}_e{expansion}.nii.gz"))
           
            # Load the deformed BowelGas mask to calculate the volume
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



def get_random_from_uniform(bounds = [-5,5], sample_size=1):
    """
    Generate a random value from a uniform distribution within given bounds.

    Parameters:
        - bounds (list or tuple): Sequence specifying the lower and upper bounds of the uniform distribution. Default is [-5, 5].
        - sample_size (int): Number of values to sample. Default is 1.

    Returns:
        - random_number (float): A single random value drawn uniformly from the specified interval.
    """
    lower_bound, upper_bound = bounds
    random_number = np.random.uniform(low=lower_bound, high=upper_bound, size=sample_size)
    return random_number[0]



def make_deformation_choice(organ):
    """
    Generate a random deformation vector (x, y, z) for a given organ, based on reported inter-fraction motion ranges [Guerreiro et al. (2018)]. With 50% probability, a deformation is applied; otherwise all shifts are zero.

    Parameters:
        - organ (str): The organ for which deformation should be simulated. Supported options are 'Liver', 'Spleen', 'Kidney Right', and 'Kidney Left'.

    Returns:
        - x_shift (float): The shift in the x-direction (+/-) (superior/inferior) if deformation is applied, otherwise 0.
        - y_shift (float): The shift in the y-direction (+/-) (posterior/anterior) if deformation is applied, otherwise 0.
        - z_shift (float): The shift in the z-direction (+/-) (left/right) if deformation is applied, otherwise 0.
    """

    apply_deformation_statement = random.choice([True, False])
    print(f'Apply deformation?  {apply_deformation_statement}')

    # Initialize shifts to 0
    x_shift, y_shift, z_shift = 0, 0, 0

    if apply_deformation_statement:
        # Organ-specific mean and range inter-fraction motion
        if organ == 'Liver':
            median_shifts = {'x': 0.2, 'y': -0.1, 'z': -1.2}
            range_shifts = {'x': [-8.1, 9.0], 'y': [-3.3, 4.8], 'z': [-4.0, 4.3]}

        elif organ == 'Spleen':
            median_shifts = {'x': 1, 'y': -0.2, 'z': -0.5}
            range_shifts = {'x': [-9.6, 9.1], 'y': [-4.0, 4.9], 'z': [-3.5, 2.7]}

        elif organ == 'Kidney Right' or organ == 'Kidney Left': 
            median_shifts = {'x': 0.2, 'y': 0.4, 'z': -0.1}
            range_shifts = {'x': [-6.3, 5.6], 'y': [-1.6, 3.9], 'z': [-3.7, 3.7]}

        direction = 'xyz' # Apply organ shifts in all 3 directions; otherwise can modify to: random.choice(['x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'])
        print(f'Direction:  {direction}')

        # Set shifts based on the chosen direction
        if 'x' in direction:
            x_shift = get_random_from_uniform(bounds=range_shifts['x'])
        if 'y' in direction:
            y_shift = get_random_from_uniform(bounds=range_shifts['y'])
        if 'z' in direction:
            z_shift = get_random_from_uniform(bounds=range_shifts['z'])

        print(f'Vector shift: x={x_shift:.2f} y={y_shift:.2f} z={z_shift:.2f}')

    return x_shift, y_shift, z_shift



def apply_transform_all(structures_dictionary, dvf_transform):
    """
    Deform a dictionary of structures using a given transformation.

    Parameters:
        - structures_dictionary (dict): A dictionary containing SimpleITK images representing different anatomical structures.
        - dvf_transform (sitk.DisplacementFieldTransform): The transformation to be applied.

    Returns:
        - deformed_structures (dict): A dictionary containing deformed structures.
    """

    deformed_structures = {}

    for struct in structures_dictionary:
        print(f"Deforming: {struct}")
        deformed_structures[struct] = apply_transform(
            structures_dictionary[struct], transform=dvf_transform, default_value=0, interpolator=sitk.sitkNearestNeighbor)

    return deformed_structures



def load_and_calculate_volume_seg(path_to_mask):
    """
    Load a 3D binary segmentation mask from a NIfTI file and calculate the physical volume of the segmented structure.

    Parameters:
        - path_to_mask (str): File path to the NIfTI mask containing a binary segmentation.

    Returns:
        - total_volume (float): The volume of the segmented structure in millilitres (mL).
    """

    # Load the NIfTI image
    nii_img = nib.load(path_to_mask)

    # Extract voxel dimensions (mm)
    voxel_dimensions = nii_img.header['pixdim'][1:4]

    # Load image data
    nii_data = nii_img.get_fdata()

    # Count segmentation voxels (value = 1)
    total_voxels = np.count_nonzero(nii_data == 1)

    # Compute physical volume in mm³
    total_volume_mm3 = total_voxels * np.prod(voxel_dimensions)
    
    return total_volume_mm3 * 0.001 # Convert mm³ → ml.



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



def extract_calibration_data(df_calibration):
    """
    Extract expansion vectors and percentage volume changes from the calibration calibration curve.

    Parameters:
        - df_calibration (pandas.DataFrame): A calibration curve DataFrame containing the following columns:
            - "SubjectID"
            - "ExpansionVector(mm)"
            - "PercentageDifference(%)"

    Returns:
        - tuple of dict: 
            - expansion_vectors_dict (dict): Maps each SubjectID to a list of expansion vectors (mm).
            - percentage_diffs_dict (dict): Maps each SubjectID to a list of percentage volume changes (%).
    """


    expansion_vectors_dict = {}
    percentage_diffs_dict = {}

    # Loop through all calibration entries and group values by subject
    for _, row_calibration in df_calibration.iterrows():
        subject_id = row_calibration["SubjectID"]
        expansion_vector = row_calibration["ExpansionVector(mm)"]
        percentage_diffs = row_calibration["PercentageDifference(%)"]

        # Append data to existing subject or create a new entry
        if subject_id in expansion_vectors_dict:
            expansion_vectors_dict[subject_id].append(expansion_vector)
            percentage_diffs_dict[subject_id].append(percentage_diffs)
        else:
            expansion_vectors_dict[subject_id] = [expansion_vector]
            percentage_diffs_dict[subject_id] = [percentage_diffs]

    return expansion_vectors_dict, percentage_diffs_dict
    


def create_expansions_df(percentage_diffs_dict, num_values=4):
    """
    Generate a DataFrame of equally spaced percentage expansion values for each subject based on their calibration data.
    
    Parameters:
    - percentage_diffs_dict (dict): A dictionary mapping each SubjectID to a list of percentage volume changes (%) obtained during calibration.
    - num_values: Number of equally spaced percentage values to generate for each subject. Default is 4.
    
    Returns:
    - df_expansions (pandas.DataFrame): A DataFrame where each row corresponds to a SubjectID and contains the 4 generated equally spaced percentage volume changes values. Columns are: ['SubjectID', 'Expansion1', ..., 'ExpansionN'].
    """

    equally_spaced_dict = {}
    # Generate equally spaced values for each subject using their min/max calibration percentages
    for key, values in percentage_diffs_dict.items():
        min_val = min(values)
        max_val = max(values)

        # Produce evenly spaced values within the subject’s calibration range
        equally_spaced = np.linspace(min_val, max_val, num_values)
        equally_spaced_dict[key] = equally_spaced

    # Convert dictionary into DataFrame
    df_expansions = pd.DataFrame.from_dict(equally_spaced_dict, orient='index')
    df_expansions = df_expansions.rename_axis('SubjectID').reset_index()

    # Rename columns dynamically based on num_values
    new_columns = ['SubjectID'] + [f'Expansion{i+1}' for i in range(num_values)]
    df_expansions.columns = new_columns

    return df_expansions



def calculate_desired_expansion(expansions_mm, percentage_diffs, desired_expansion_perc):
    """
    Determine an expansion vector (in mm) corresponding to a desired percentage volume change by linearly interpolating between known percentage–expansion pairs.   
    
    Parameters:
        - expansions_mm (array):  Expansion vectors (in mm) corresponding to each percentage in `percentage_diffs`.
        - percentage_diffs (array): Known percentage volume changes used as interpolation points. Must be the same length as `expansions_mm`.
        - desired_expansion_perc (float): The desired percentage change for which an expansion vector is determined.
    
    Returns:
        - desired_expansion_mm (float): Interpolated expansion vector (in mm) corresponding to the requested (or clipped) percentage volume change.
    """

    # Check if desired percentage change is within calibration curve range
    min_perc = min(percentage_diffs)
    max_perc = max(percentage_diffs)
    
    # If outside the range, adjust to the minimum or maximum achievable
    if desired_expansion_perc < min_perc:
        desired_expansion_perc = min_perc
        print(f"\nDesired percentage is below minimum ({min_perc:.2f}%), adjusting to {min_perc:.2f}%")
    elif desired_expansion_perc > max_perc:
        desired_expansion_perc = max_perc
        print(f"Desired percentage is above maximum ({max_perc:.2f}%), adjusting to {max_perc:.2f}%")
    elif min_perc <= desired_expansion_perc <= max_perc:
        print(f"Desired percentage is within range ({min_perc:.2f}% - {max_perc:.2f}%)")

        # Interpolate to find the expansion vector (in mm) for the given percentage difference
        interp_func = interp1d(percentage_diffs, expansions_mm, kind='linear')
        desired_expansion_mm = interp_func(desired_expansion_perc)

    return desired_expansion_mm