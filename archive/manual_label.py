import cv2
import os
import random
import csv

# Folders with your cropped images.
healthy_folder = "healthy_images_cropped"
wssv_folder = "wssv_images_cropped"

# List image files (supports jpg, jpeg, png).
healthy_files = [os.path.join(healthy_folder, f) for f in os.listdir(healthy_folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
wssv_files = [os.path.join(wssv_folder, f) for f in os.listdir(wssv_folder)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Combine and randomly shuffle the lists.
all_files = healthy_files + wssv_files
random.shuffle(all_files)

# CSV file to log classifications.
csv_filename = "image_classifications.csv"
with open(csv_filename, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_name", "label"])

    # Process images one by one.
    for file_path in all_files:
        img = cv2.imread(file_path)
        if img is None:
            print(f"Could not load {file_path}. Skipping.")
            continue

        # Show the image.
        cv2.imshow("Image", img)
        print(f"Displaying image: {file_path}")
        key = cv2.waitKey(0) & 0xFF

        # Check key press: 0 for healthy, 1 for wssv, q to quit.
        if key == ord('0'):
            label = "healthy"
        elif key == ord('1'):
            label = "wssv"
        elif key == ord('q'):
            print("Quitting classification.")
            break
        else:
            print("Invalid key pressed; skipping image.")
            cv2.destroyAllWindows()
            continue

        # Log the image filename and label.
        writer.writerow([os.path.basename(file_path), label])
        cv2.destroyAllWindows()

cv2.destroyAllWindows()
print("Classification complete! Results saved in", csv_filename)
