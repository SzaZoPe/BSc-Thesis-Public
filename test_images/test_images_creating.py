import cv2
import os
import shutil
import time

# Input folders
input_folder = "./images/base_pgms/cover"
temp_folder = "./images/normal_jpegs"
output_folder = "./images/double_compressed_jpegs"

# Quality levels
qualities = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

# Create output folders
os.makedirs(output_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)

# Start timer
start_time = time.time()

# Process images
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".pgm"):  # Only PGM images
        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)  # Load PGM image

        if image is not None:
            base_name = os.path.splitext(filename)[0]  # Filename without extension

            # Save with each quality level
            for q in qualities:
                # Save to temp folder with quality q
                temp_output_path = os.path.join(temp_folder, f"{base_name}_q{q}.jpg")
                cv2.imwrite(temp_output_path, image, [cv2.IMWRITE_JPEG_QUALITY, q])


for filename in os.listdir(temp_folder):
    if filename.lower().endswith(".jpg"):  # Only JPG images
        input_path = os.path.join(temp_folder, filename)

        # Load image
        image = cv2.imread(input_path)

        if image is not None:
            # Extract original quality from filename
            base_name, ext = os.path.splitext(filename)
            original_quality = int(base_name.split('_q')[-1])  # Extract q{quality}

            # Save with all other quality levels
            for q in qualities:
                if q != original_quality:  # If not the same quality, save
                    output_subfolder = os.path.join(output_folder, f"{original_quality}_{q}")
                    os.makedirs(output_subfolder, exist_ok=True)

                    output_path = os.path.join(output_subfolder, f"{base_name}_q{q}.jpg")
                    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, q])

# Stop timer and print elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print("Images saved with different qualities.")
print(f"Execution time: {elapsed_time:.2f} seconds.")
