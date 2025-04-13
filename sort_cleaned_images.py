import os
import csv
import shutil

# Source folders for each class.
healthy_src = "healthy_images_cropped"
wssv_src = "wssv_images_cropped"

# Destination folders.
healthy_dest = "healthy_images_clean"
wssv_dest = "wssv_images_clean"

# Create destination folders if they don't exist.
os.makedirs(healthy_dest, exist_ok=True)
os.makedirs(wssv_dest, exist_ok=True)

# CSV file containing the classifications.
csv_file = "image_classifications.csv"

with open(csv_file, mode="r", newline="") as f:
    reader = csv.DictReader(f)  # Expect headers "image_name" and "label"
    for row in reader:
        image_name = row["image_name"]
        label = row["label"].strip().lower()
        if label == "healthy":
            source_path = os.path.join(healthy_src, image_name)
            dest_path = os.path.join(healthy_dest, image_name)
        elif label == "wssv":
            source_path = os.path.join(wssv_src, image_name)
            dest_path = os.path.join(wssv_dest, image_name)
        else:
            print(f"Unknown label '{label}' for {image_name}; skipping.")
            continue
        
        if not os.path.exists(source_path):
            print(f"Source file {source_path} does not exist; skipping.")
            continue
        
        shutil.copy(source_path, dest_path)
        print(f"Copied {image_name} to {dest_path}")

print("Copying complete.")
