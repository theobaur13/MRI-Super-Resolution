import os
import re
from datetime import datetime

def get_matching_adni_scan(path):
    image = os.path.basename(path)
    adni_dir = os.path.dirname(os.path.dirname(path))

    # Search the 1.5T and 3T directories for the given image_id
    pattern = re.compile(r'ADNI_(\d+_S_\d+)_MR_([A-Za-z0-9_]+)_br_raw_(\d+)_([0-9]+)_S[0-9]+_(I\d+)')
    match = pattern.match(image)
    
    if match:
        # Extract patient ID, description, and timestamp from the filename
        patient_id = match.group(1)
        query_description = match.group(2)
        timestamp_raw = match.group(3)
        datestamp = datetime.strptime(timestamp_raw, '%Y%m%d%H%M%S%f').date()
        datestamp = str(datestamp).replace("-", "")

        # Check if the image is in the 1.5T or 3T directory
        if query_description == "Axial_PD_T2_FSE_":
            target_strengh = "3T"
            target_description = "Double_TSE"
        elif query_description == "Double_TSE":
            target_strengh = "1.5T"
            target_description = "Axial_PD_T2_FSE_"

        # If image is in 1.5T, search in 3T and vice versa
        target_dir = os.path.join(adni_dir, target_strengh)

        # Iterate through the files names in the target directory
        target = None
        for file in os.listdir(target_dir):
            search_pattern = re.compile(rf'ADNI_{patient_id}_MR_{target_description}_br_raw_{datestamp}.*')

            if search_pattern.match(file):
                target = os.path.join(target_dir, file)
                break

        # Return the paths of the images with the given image_id in order of 1.5T and 3T
        if target_strengh == "1.5T":
            return target, path
        elif target_strengh == "3T":
            return path, target
        else:
            print(f"No matching image found in {target_strengh} directory for {image}")
            return None