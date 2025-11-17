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


nifti_folder = "../data/output_deformation"

# Get a list of all files in the folder
all_nifti_files = os.listdir(nifti_folder)

# Filter files for BowelGas and CT scans ending with "_d1", "_d2", "_d3", or "_d4"
gas_masks = np.sort([mask for mask in all_nifti_files if mask.endswith(("_d1.nii.gz", "_d2.nii.gz", "_d3.nii.gz", "_d4.nii.gz")) and 'BowelGas' in mask])
CT_scans = np.sort([scan for scan in all_nifti_files if scan.endswith(("_d1.nii.gz", "_d2.nii.gz", "_d3.nii.gz", "_d4.nii.gz")) and 'CT' in scan])


# Check if the number of BowelGas and CT files match
if len(gas_masks) != len(CT_scans):
    raise ValueError("Number of BowelGas and CT files do not match.")


def scale_affine(nifti_file, scale_factor):
      # Scale the affine matrix of the BowelGas NIfTI file
    affine_matrix = nifti_file.affine.copy()
    affine_matrix[0, 0] *= scale_factor
    affine_matrix[1, 1] *= scale_factor
    affine_matrix[2, 2] *= scale_factor
    return affine_matrix


scale_data = []

# Iterate through the files and apply the same random scale factor
for gas_mask, CT_scan in zip(gas_masks, CT_scans):
    id = '_'.join(CT_scan.split('_')[0:2])
    count = CT_scan.split('d')[1].split('.')[0]
    print(f"\nProcessing files for patient {id} version {count}:")

    # Load the BowelGas and CT NIfTI files
    gas_mask_nii = nib.load(os.path.join(nifti_folder, gas_mask))
    CT_nii = nib.load(os.path.join(nifti_folder, CT_scan))
    
    # Extract image info
    gas_mask_header = gas_mask_nii.header
    gas_mask_data = gas_mask_nii.get_fdata()
    CT_header = CT_nii.header
    CT_data = CT_nii.get_fdata()

    # Define the range for the random scale factor (-0.045 to 0.045 inclusive)
    scale_factor = np.random.uniform(0.955, 1.055)
    print(f'Scale factor: {scale_factor}')
    scale_data.append([id, count, scale_factor])

    gas_mask_affine_scaled = scale_affine(gas_mask_nii, scale_factor)
    CT_scan_affine_scaled = scale_affine(CT_nii, scale_factor)

    # Apply the scaling to the image data
    gas_mask_scaled = nib.nifti1.Nifti1Image(dataobj=gas_mask_data.astype(np.uint8), affine=gas_mask_affine_scaled, header=gas_mask_header)
    CT_scaled = nib.nifti1.Nifti1Image(dataobj=CT_data, affine=CT_scan_affine_scaled, header=CT_header)

    # Save the new scaled NIfTI files
    basename_gas = os.path.splitext(os.path.basename(gas_mask)[:-4])[0]  # Get filename without extension
    basename_CT = os.path.splitext(os.path.basename(CT_scan)[:-4])[0]  # Get filename without extension

    nib.save(gas_mask_scaled, os.path.join(nifti_folder, f"{basename_gas}_scaled.nii.gz"))
    nib.save(CT_scaled, os.path.join(nifti_folder, f"{basename_CT}_scaled.nii.gz"))

    print(f"Writing {basename_gas}_scaled.nii.gz ...")
    print(f"Writing {basename_CT}_scaled.nii.gz ...")


scale_df = pd.DataFrame(scale_data, columns=["PatientID", "Version", "ScaleFactor"])
scale_csv = os.path.join(nifti_folder, "scale_data.csv")
# Write the DataFrame to a CSV file
scale_df.to_csv(scale_csv, index=False)
print(f"\nScale data saved to: {scale_csv}")


scaled_gas_masks = np.sort([mask for mask in all_nifti_files if mask.endswith(("_scaled.nii.gz")) and 'BowelGas' in mask])
volume_data = []

print('\nCalculating gas volume for scaled masks...')
for scaled_gas_mask in scaled_gas_masks:
    id = os.path.splitext(os.path.basename(scaled_gas_mask)[:-4])[0] # Get filename without extension
    scaled_volume = load_and_calculate_volume_seg(os.path.join(nifti_folder, scaled_gas_mask))
    volume_data.append([id, scaled_volume]) 


volume_df = pd.DataFrame(volume_data, columns=["PatientID", "GasVolume(ml)"])
volume_csv = os.path.join(nifti_folder, "scaled_gas_labels.csv")
# Write the DataFrame to a CSV file
volume_df.to_csv(volume_csv, index=False)
print(f"Volume data saved to: {volume_csv}")