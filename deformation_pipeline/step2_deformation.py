import os
import SimpleITK as sitk
from platipy.imaging.registration.utils import apply_transform
from platipy.imaging.generation.dvf import (generate_field_shift, generate_field_expand)
from platipy.imaging.generation.mask import get_bone_mask
from deformation_utils import *
import pandas as pd

# Random seeds: make the deformation choices reproducible
np.random.seed(42)
random.seed(42)

# --- NumPy 2 compatibility shim for old Platipy code ---
if not hasattr(np, "alen"):
    # np.alen used to be "length of first axis"
    np.alen = lambda a: len(a)



# ---------------------------------------------------------------------
# Paths and basic configuration
# ---------------------------------------------------------------------

# Base data directory
path = "./data"

# Folders for CT and structure segmentations
input_CT_folder = os.path.join(path, "input_CT")
input_SEG_folder = os.path.join(path, "input_SEG")

# Structures to deform (here: Liver, Spleen, Kidneys - xyz shift; BowelGas - isotropic shrink/expansion)
structure_names_to_deform = ["Liver", "Spleen", "Kidney Left", "Kidney Right", "BowelGas"]

# Structures for which deformed segmentations will be saved (here: only BowelGas)
structure_names_to_be_deformed = ["BowelGas"]

# Output folder for all deformed CTs and BowelGas segmentations
output_deform_folder = os.path.join(path, "output_deformation")
os.makedirs(output_deform_folder, exist_ok=True)



# ---------------------------------------------------------------------
# Calibration data handling for BowelGas expansion
# ---------------------------------------------------------------------

# Load calibration curve from step 1 (expansion vs percentage change in volume)
df_calibration = pd.read_csv(os.path.join(path, "output_calibration","calibration_curve.csv"), delimiter=",")
print(df_calibration)

# Extract per-subject expansion vectors and percentage volume changes from the calibration curve
expansion_vectors_dict, percentage_diffs_dict = extract_calibration_data(df_calibration)
# Generate 4 target percentage volume changes for each subject. These represent the target volume changes achievable when deforming the BowelGas and will be used to interpolate the actual expansion (in mm) from the calibration curve.
df_expansions = create_expansions_df(percentage_diffs_dict)
print(df_expansions)



# ---------------------------------------------------------------------
# Arrays to collect results for CSV export
# ---------------------------------------------------------------------

save_gas_volumes = [] # Deformed bowel gas volumes
save_subject_ids = [] # Subject IDs
deformation_data = [] # Deformation parameters (translations / expansions)



# ---------------------------------------------------------------------
# Main loop: apply deformation pipeline to each subject
# ---------------------------------------------------------------------

# Each row corresponds to one subject and their 4 planned Bowel Gas expansions
for _, row_expansions in df_expansions.iloc[:].iterrows():
    subject_id = row_expansions["SubjectID"]
    print(f'Processing subject {subject_id}...')

    # Load CT and generate bone mask
    ct = sitk.ReadImage(os.path.join(input_CT_folder, f'{subject_id}.nii.gz'))
    bone_mask = get_bone_mask(ct)

    # Load segmentations for all organs
    try:
        # Structures to deform 
        structures_to_deform = {s: sitk.ReadImage(os.path.join(
            input_SEG_folder, f"{subject_id}_{s}.nii.gz")) for s in structure_names_to_deform}
        
        # Structures for which segmentations will be saved after deformation
        structures_to_be_deformed = {s: sitk.ReadImage(os.path.join(
            input_SEG_folder, f"{subject_id}_{s}.nii.gz")) for s in structure_names_to_be_deformed}
    except RuntimeError as e:
        print(f"Error reading {subject_id} structures: {e}")
        continue



    # --------------------------------------------------------------
    # Save original (undeformed) CT and BowelGas label as version d0
    # --------------------------------------------------------------

    print(f'Writing {subject_id}_CT_d0.nii.gz ...')
    sitk.WriteImage(ct, os.path.join(output_deform_folder, f"{subject_id}_CT_d0.nii.gz"))
    
    # (Optional) Save all deformed structures
    # for structure in structures_to_deform:
    #     print(f'Writing {patient_id}_{structure}_d0.nii.gz ...')
    #     sitk.WriteImage(structures_to_deform[structure], os.path.join(output_deform_folder, f"{patient_id}_{structure}_d0.nii.gz"))

    for structure in structures_to_be_deformed:
        print(f'Writing {subject_id}_{structure}_d0.nii.gz ...')
        sitk.WriteImage(structures_to_be_deformed[structure], os.path.join(output_deform_folder, f"{subject_id}_{structure}_d0.nii.gz"))

    # Compute initial BowelGas volume for this subject (baseline before deformation)
    volume_init = load_and_calculate_volume_seg(os.path.join(output_deform_folder, f"{subject_id}_{structure}_d0.nii.gz"))
    save_gas_volumes.append(volume_init) 
    save_subject_ids.append(f"{subject_id}_d0")
        
    
    
    # -----------------------------------------------------------------
    # Generate 4 different deformation versions of the same subject
    # -----------------------------------------------------------------
    for i in range(1, 5):
        all_organ_transforms = []
        print(f'\nVersion {i}')

        # Reset CT and structures at every deformation version
        ct_copy = sitk.Image(ct)
        bone_mask_copy = sitk.Image(bone_mask)
        structures_to_deform_copy = {s: sitk.Image(structures_to_deform[s]) for s in structures_to_deform}
        structures_to_be_deformed_copy = {s: sitk.Image(structures_to_be_deformed[s]) for s in structures_to_be_deformed}


        print(f'\nGenerate transform:')
        for structure in structures_to_deform_copy:
            print(f'::::: {structure}')



            # ----------------------------------------------------------
            # 1) Bowel Gas expansion
            # ----------------------------------------------------------

            if structure == "BowelGas": 
                # Use the calibration curve to determine how much to expannd BowelGas (in mm) to reach the target percentage volume change.
                desired_expansion_perc = row_expansions[f"Expansion{i}"]

                # Subject-specific calibration information from step 1
                expansions_mm = expansion_vectors_dict[subject_id]
                percentage_diffs = percentage_diffs_dict[subject_id]

                # Interpolate to find the expansion in mm that yields the desired percentage volume change
                desired_expansion_mm = calculate_desired_expansion(expansions_mm, percentage_diffs, desired_expansion_perc)

                print(f"Desired Percentage Difference: {desired_expansion_perc:.2f}%")
                print(f"Interpolated Expansion Vector: {desired_expansion_mm:.2f} mm")

                # Generate a deformation vector field that expands the gas region
                label_deformed, dvf_transform, dvf_field = generate_field_expand(
                    structures_to_deform_copy[structure], bone_mask=bone_mask_copy, expand=desired_expansion_mm, gaussian_smooth=5)
                
                deformation_data.append([subject_id, i, structure, desired_expansion_mm, desired_expansion_mm, desired_expansion_mm])
                


            # ----------------------------------------------------------
            # 2) Organ motion (Liver/ Spleen/ Kidney Left/ Kidney Right)
            # ----------------------------------------------------------

            else: 
                # Select a displacement vector for this organ 
                x_shift, y_shift, z_shift = make_deformation_choice(structure)

                # Generate a shift DVF for the organ structure
                label_deformed, dvf_transform, dvf_field = generate_field_shift(
                    structures_to_deform_copy[structure],
                    vector_shift=(x_shift, y_shift, z_shift),
                    gaussian_smooth=5)
                
                deformation_data.append([subject_id, i, structure, x_shift, y_shift, z_shift])
                
            all_organ_transforms.append(dvf_transform)
        

        # Combine all individual organ transforms into a single composite transform
        print(f'\n===> Generate Composite Transform...')
        dvf_transform = sitk.CompositeTransform(all_organ_transforms)

        # (Optional) Save the composite transform
        # sitk.WriteTransform(dvf_transform, os.path.join(output_deform_folder, f"{patient_id}_transform_d{i}.tfm"))
        # print(f"Composite transform saved to {patient_id}_transform_d{i}.tfm")
        


        # --------------------------------------------------------------
        # Apply the composite transform to CT and structures
        # --------------------------------------------------------------

        # Deform specified structures and CT using the above defined transformation
        deformed_structures = apply_transform_all(
            structures_to_deform_copy, dvf_transform)
        
        deformed_adjacent_structures = apply_transform_all(
            structures_to_be_deformed_copy, dvf_transform)

        deformed_ct = apply_transform(
            ct_copy, transform=dvf_transform, default_value=-1000, interpolator=sitk.sitkLinear)



        # --------------------------------------------------------------
        # Save deformed CT and structures for this version
        # --------------------------------------------------------------
        print(f'Writing {subject_id}_CT_d{i}.nii.gz ...')
        sitk.WriteImage(deformed_ct, os.path.join(output_deform_folder, f"{subject_id}_CT_d{i}.nii.gz"))
        
        # (Optional) Save all deformed structures
        # for structure in deformed_structures:
        #     print(f'Writing {patient_id}_{structure}_d{i}.nii.gz ...')
        #     sitk.WriteImage(deformed_structures[structure], os.path.join(output_deform_folder, f"{patient_id}_{structure}_d{i}.nii.gz"))

        for structure in deformed_adjacent_structures:
            print(f'Writing {subject_id}_{structure}_d{i}.nii.gz ...')
            sitk.WriteImage(deformed_adjacent_structures[structure], os.path.join(output_deform_folder, f"{subject_id}_{structure}_d{i}.nii.gz"))



        # --------------------------------------------------------------
        # Compute and store BowelGas volume for this deformation version
        # --------------------------------------------------------------
        
        volume_deformed = load_and_calculate_volume_seg(os.path.join(output_deform_folder, f"{subject_id}_{structure}_d{i}.nii.gz"))
        save_gas_volumes.append(volume_deformed)
        save_subject_ids.append(f"{subject_id}_d{i}")

        
results_df = pd.DataFrame({
    'SubjectID': save_subject_ids,
    'GasVolume(ml)': save_gas_volumes,
})

results_csv = os.path.join(output_deform_folder, "gas_volumes.csv")
results_df.to_csv(results_csv, index=False)
print(f"\nBowelGas volumes of deformed masks saved to: {results_csv}")


# ---------------------------------------------------------------------
# Save deformation parameters (BowelGas expansions & Organ shifts)
# ---------------------------------------------------------------------
deformation_df = pd.DataFrame(deformation_data, columns=["SubjectID", "Version", "StructureName", "X", "Y", "Z"])
deformation_csv = os.path.join(output_deform_folder, "deformation_data.csv")
deformation_df.to_csv(deformation_csv, index=False)
print(f"\nDeformation data saved to: {deformation_csv}")