import cv2
import os

# Set directories; update these paths as needed.
input_dir = "ShrimpDiseaseImageBD An Image Dataset for Computer Vision-Based Detection of Shrimp Diseases in Bangladesh/Root/Raw Images/1. Healthy"  # Folder with your original images
output_dir = "healthy_images_cropped"  # Folder to save cropped images

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Global variables to store the current top-left coordinates of the crop box.
crop_x, crop_y = 0, 0
box_size = 224

# Mouse callback function: updates crop coordinates when you move the mouse.
def mouse_callback(event, x, y, flags, param):
    global crop_x, crop_y, img
    if event == cv2.EVENT_MOUSEMOVE:
        height, width, _ = img.shape
        crop_x = min(max(x, 0), width - box_size)
        crop_y = min(max(y, 0), height - box_size)

# List image files in the input directory (supports jpg, jpeg, png).
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Process each image.
for image_name in image_files:
    image_path = os.path.join(input_dir, image_name)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load {image_path}. Skipping.")
        continue

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        display_img = img.copy()
        cv2.rectangle(display_img, (crop_x, crop_y), (crop_x + box_size, crop_y + box_size), (0, 255, 0), 2)
        cv2.imshow("Image", display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            crop_img = img[crop_y:crop_y + box_size, crop_x:crop_x + box_size]
            base_name, ext = os.path.splitext(image_name)
            new_filename = f"{base_name}_{crop_x}_{crop_y}{ext}"
            output_path = os.path.join(output_dir, new_filename)
            cv2.imwrite(output_path, crop_img)
            print(f"Saved crop to {output_path}")
            break
        elif key == ord('n'):
            print(f"Skipping {image_name}")
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit(0)

    cv2.destroyWindow("Image")

cv2.destroyAllWindows()
