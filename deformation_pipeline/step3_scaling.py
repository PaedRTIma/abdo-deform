import os
import numpy as np
import nibabel as nib
import pandas as pd
from deformation_utils import *

# Random seeds: make the deformation choices reproducible
np.random.seed(42)

# --- NumPy 2 compatibility shim for old Platipy code ---
if not hasattr(np, "alen"):
    # np.alen used to be "length of first axis"
    np.alen = lambda a: len(a)


def scale_affine(nifti_file, scale_factor):
    """
    Given a loaded NIfTI image, return a new affine matrix with
    the voxel spacing scaled by a factor in x, y, and z.
    """

    affine_matrix = nifti_file.affine.copy()
    affine_matrix[0, 0] *= scale_factor
    affine_matrix[1, 1] *= scale_factor
    affine_matrix[2, 2] *= scale_factor
    return affine_matrix



# ---------------------------------------------------------------------
# Input folder containing the already deformed CTs and BowelGas masks
# (from step 2)
# ---------------------------------------------------------------------
nifti_folder = "./data/output_deformation"

# Get a list of all files in the folder
all_nifti_files = os.listdir(nifti_folder)


# ---------------------------------------------------------------------
# Select BowelGas masks and CT scans for versions d1–d4 (deformed versions)
#    Example filenames expected:
#       <SubjectID>_BowelGas_d1.nii.gz
#       <SubjectID>_CT_d1.nii.gz
# ---------------------------------------------------------------------
# Filter files for BowelGas and CT scans ending with "_d1", "_d2", "_d3", or "_d4"
gas_masks = np.sort([mask for mask in all_nifti_files if mask.endswith(("_d1.nii.gz", "_d2.nii.gz", "_d3.nii.gz", "_d4.nii.gz")) and 'BowelGas' in mask])
CT_scans = np.sort([scan for scan in all_nifti_files if scan.endswith(("_d1.nii.gz", "_d2.nii.gz", "_d3.nii.gz", "_d4.nii.gz")) and 'CT' in scan])

# Check if the number of BowelGas mask and CT files match
if len(gas_masks) != len(CT_scans):
    raise ValueError("Number of BowelGas and CT files do not match.")


# ---------------------------------------------------------------------
# Loop over BowelGas/CT pairs and apply random uniform scaling
# ---------------------------------------------------------------------

scale_data = []

for gas_mask, CT_scan in zip(gas_masks, CT_scans):
    subject_id = '_'.join(CT_scan.split('_')[0:2])
    count = CT_scan.split('d')[1].split('.')[0]
    print(f"\nProcessing files for patient {subject_id} version {count}:")

    # Load the BowelGas and CT NIfTI files
    gas_mask_nii = nib.load(os.path.join(nifti_folder, gas_mask))
    CT_nii = nib.load(os.path.join(nifti_folder, CT_scan))
    
    # Extract BowelGas mask image info
    gas_mask_header = gas_mask_nii.header
    gas_mask_data = gas_mask_nii.get_fdata()

    # Extract CT image info
    CT_header = CT_nii.header
    CT_data = CT_nii.get_fdata()


    # -----------------------------------------------------------------
    # Random scale factor in [0.955, 1.055]
    #
    # This corresponds to approximately ±4.5% change in size:
    #   0.955 → -4.5%  (downscale)
    #   1.055 → +4.5%  (upscale)
    # -----------------------------------------------------------------
   
    scale_factor = np.random.uniform(0.955, 1.055)
    print(f'Scale factor: {scale_factor}')
    scale_data.append([subject_id, count, scale_factor])

    # Compute scaled affines for BowelGas mask and CT
    gas_mask_affine_scaled = scale_affine(gas_mask_nii, scale_factor)
    CT_scan_affine_scaled = scale_affine(CT_nii, scale_factor)

    # Apply the scaled affines to the image data
    gas_mask_scaled = nib.nifti1.Nifti1Image(dataobj=gas_mask_data.astype(np.uint8), affine=gas_mask_affine_scaled, header=gas_mask_header)
    CT_scaled = nib.nifti1.Nifti1Image(dataobj=CT_data, affine=CT_scan_affine_scaled, header=CT_header)

    # Create output file names and save the new scaled NIfTI files
    basename_gas = os.path.splitext(os.path.basename(gas_mask)[:-4])[0]  # Get filename without extension
    basename_CT = os.path.splitext(os.path.basename(CT_scan)[:-4])[0]  # Get filename without extension

    nib.save(gas_mask_scaled, os.path.join(nifti_folder, f"{basename_gas}_scaled.nii.gz"))
    nib.save(CT_scaled, os.path.join(nifti_folder, f"{basename_CT}_scaled.nii.gz"))

    print(f"Writing {basename_gas}_scaled.nii.gz ...")
    print(f"Writing {basename_CT}_scaled.nii.gz ...")


# ---------------------------------------------------------------------
# Save scale factors in a CSV file
# ---------------------------------------------------------------------

scale_df = pd.DataFrame(scale_data, columns=["SubjectID", "Version", "ScaleFactor"])
scale_csv = os.path.join(nifti_folder, "scaling_data.csv")

# Write the DataFrame to a CSV file
scale_df.to_csv(scale_csv, index=False)
print(f"\nScale data saved to: {scale_csv}")


# ---------------------------------------------------------------------
# Compute Bowel Gas volumes for scaled masks
# ---------------------------------------------------------------------

scaled_gas_masks = np.sort([mask for mask in all_nifti_files if mask.endswith(("_scaled.nii.gz")) and 'BowelGas' in mask])
volume_data = []

print('\nCalculating gas volume for scaled masks...')
for scaled_gas_mask in scaled_gas_masks:
    subject_id = os.path.splitext(os.path.basename(scaled_gas_mask)[:-4])[0] # Get filename without extension
    scaled_volume = load_and_calculate_volume_seg(os.path.join(nifti_folder, scaled_gas_mask))
    volume_data.append([subject_id, scaled_volume]) 

volume_df = pd.DataFrame(volume_data, columns=["SubjectID", "GasVolume(ml)"])
volume_csv = os.path.join(nifti_folder, "scaled_gas_volumes.csv")

# Write the DataFrame to a CSV file
volume_df.to_csv(volume_csv, index=False)
print(f"\nBowel gas volumes of scaled masks saved to: {volume_csv}")