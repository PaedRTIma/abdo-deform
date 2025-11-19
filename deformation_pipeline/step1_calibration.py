import os
import SimpleITK as sitk
from platipy.imaging.registration.utils import apply_transform
from platipy.imaging.generation.dvf import generate_field_expand
from platipy.imaging.generation.mask import get_bone_mask
from deformation_utils import *
import nibabel as nib
import pandas as pd

# --- NumPy 2 compatibility shim for old Platipy code ---
if not hasattr(np, "alen"):
    # np.alen used to be "length of first axis"
    np.alen = lambda a: len(a)



def run_calibration_branch(expansions, subject_id, ct, structures_to_deform, bone_mask, output_folder, *, initial_prev, stop_condition):
    """
    Run one branch (either shrink or expand) until the monotonicity condition breaks (i.e. platipy reaches maximum shrinkage/ expansion achievable for the subject).

    Parameters
        - expansions (iterable of float): Expansion values to try (e.g. negative or positive).
        - initial_prev (float): Initial previous volume value (large for shrink, small for expand).
        - stop_condition (callable): Function taking (prev_volume, current_volume) and returning True if we should stop.

    Returns
        - gas_volumes (list[float): Monotonic gas volumes.
        - subject_ids (list[str]): Corresponding subject IDs.
        - used_expansions (list[float]): Expansion values that were actually used.
    """

    break_index = None
    prev_volume = initial_prev
    gas_volumes = []
    subject_ids = []
    used_expansions = []

    for idx, expansion in enumerate(expansions):
        volume = generate_calibration_curve(subject_id, ct, structures_to_deform, bone_mask, expansion, output_folder)

        # Decide whether to stop based on monotonicity condition
        if stop_condition(prev_volume, volume):
            break_index = idx
            break

        gas_volumes.append(volume)
        prev_volume = volume
        subject_ids.append(subject_id)
        used_expansions.append(expansion)

    # If broken early, trim lists up to the break index
    if break_index is not None:
        gas_volumes = gas_volumes[:break_index]
        subject_ids = subject_ids[:break_index]
        used_expansions = used_expansions[:break_index]

    # Keep expansions and volumes aligned while sorting
    paired = sorted(zip(used_expansions, gas_volumes), key=lambda x: x[0])
    if paired:
        used_expansions, gas_volumes = zip(*paired)
        used_expansions = list(used_expansions)
        gas_volumes = list(gas_volumes)

    return gas_volumes, subject_ids, used_expansions



# ---------------------------------------------------------------------
# Paths and basic configuration
# ---------------------------------------------------------------------

# Base data directory
path = "./data"

# Folders for CT and segmentations
input_CT_folder = os.path.join(path, "input_CT")
input_SEG_folder = os.path.join(path, "input_SEG")

# Structures to deform (here: only BowelGas)
structure_names_to_deform = ["BowelGas"]

# Output folder for the deformed images used for calibration
output_calib_folder = os.path.join(path, "output_calibration")
os.makedirs(output_calib_folder, exist_ok=True)

# Get all CT files (.nii.gz) and sort them
subject_file_list = [f for f in os.listdir(input_CT_folder) if f.endswith(".nii.gz")]
subject_file_list = sorted(subject_file_list)



# ---------------------------------------------------------------------
# Expansion settings
# ---------------------------------------------------------------------

# Negative expansions (shrinks): 0, -2, -4, ..., -10
expansions_negative_base = range(0, -11, -2)

# Positive expansions (expand): 2, 4, ..., 18
expansions_positive_base = range(2, 20, 2)

# Containers to collect results across all subjects
subject_ids_all = []
expansion_vectors_all = []
gas_volumes_all = []
percentage_diffs_all = []



# ---------------------------------------------------------------------
# Main loop: apply deformation pipeline to each subject
# ---------------------------------------------------------------------

for subject in subject_file_list[:]:
    # Extract subject ID from filename (strip extension)
    subject_id = (os.path.basename(subject)).split('.')[0]
    print(f'Processing subject {subject_id}...')

    # Load CT and generate bone mask 
    try: 
        ct = sitk.ReadImage(os.path.join(input_CT_folder, subject))
        bone_mask = get_bone_mask(ct)
    except RuntimeError as e:
        print(f"Error reading {subject_id} CT: {e}")
        continue
    
    # Load segmentation to deform (e.g. <subjectID>_<structure>.nii.gz)
    try:
        structures_to_deform = {s: sitk.ReadImage(os.path.join(input_SEG_folder, f"{subject_id}_{s}.nii.gz")) for s in structure_names_to_deform}
    except RuntimeError as e:
        print(f"Error reading {subject_id} structures: {e}")
        continue
    
    # Make local copies of expansion ranges for this subject
    expansions_negative = list(expansions_negative_base)  
    expansions_positive = list(expansions_positive_base) 
    gas_volumes =[]


    # -----------------------------------------------------------------
    # 1) Shrink branch: negative expansions (0, -2, -4, ...)
    #    Stop when the BowelGas volume starts increasing again (current >= previous)
    # -----------------------------------------------------------------

    gas_volumes_shrink, patient_ids_shrink, expansions_negative_all = run_calibration_branch(expansions_negative, subject_id, ct, structures_to_deform, bone_mask, output_calib_folder,
        initial_prev=1e4, stop_condition=lambda prev, cur: cur >= prev)


    # -----------------------------------------------------------------
    # 2) Expansion branch: positive expansions (2, 4, 6, ...)
    #    Stop when the BowelGas volume starts decreasing again (current <= previous)
    # -----------------------------------------------------------------

    gas_volumes_expand, patient_ids_expand, expansions_positive_all = run_calibration_branch(expansions_positive, subject_id, ct, structures_to_deform, bone_mask, output_calib_folder, initial_prev=-1e4, stop_condition=lambda prev, cur: cur <= prev)


    # -----------------------------------------------------------------
    # 3) Combine shrink + expand results for current subject
    # -----------------------------------------------------------------

    # All gas volumes (negative + positive expansions)
    gas_volumes = gas_volumes_shrink + gas_volumes_expand
    gas_volumes_all.append(gas_volumes)

    # All expansion values (mm) in corresponding order
    expansion_vectors = expansions_negative_all + expansions_positive_all
    expansion_vectors_all.append(expansion_vectors)

    # Corresponding subject IDs (same subject repeated)
    subject_ids = patient_ids_shrink + patient_ids_expand
    subject_ids_all.append(subject_ids)

    # Compute percentage change relative to volume at 0 mm expansion
    percentage_diffs = [percentage_change(volume, gas_volumes[expansions_negative_all.index(0.0)]) for volume in gas_volumes]
    percentage_diffs_all.append(percentage_diffs)



# ---------------------------------------------------------------------
# After processing all subjects: aggregate into a single DataFrame
# ---------------------------------------------------------------------

results_df = pd.DataFrame({
    'SubjectID': np.concatenate(subject_ids_all),
    'ExpansionVector(mm)': np.concatenate(expansion_vectors_all),
    'GasVolume(ml)': np.concatenate(gas_volumes_all),
    'PercentageDifference(%)': np.concatenate(percentage_diffs_all)})

results_csv = os.path.join(output_calib_folder, "calibration_curve.csv")
results_df.to_csv(results_csv, index=False)
print(f"\nResults saved to: {results_csv}")