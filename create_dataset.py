import os
from PIL import Image, ImageFilter

# --- 1. SETTINGS ---
# --- IMPORTANT: Change this to the path of your source image ---
SOURCE_IMAGE_PATH = "C:\\Users\\hp\\Downloads\\secondTestImage.jpg" 

# This is where the new blurred images will be saved
OUTPUT_DIR = "blurred_dataset" 

# These are the blur levels from your project plan 
BLUR_RADII = [0, 2, 5, 10, 15, 20]
# ---------------------


def create_blurred_dataset():
    
    # --- 2. Check if the output folder exists, if not, create it ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # --- 3. Try to open the source image ---
    try:
        img = Image.open(SOURCE_IMAGE_PATH)
        # Get the base filename (e.g., "my_source_image")
        base_name = os.path.splitext(os.path.basename(SOURCE_IMAGE_PATH))[0]
    except FileNotFoundError:
        print(f"Error: Source image not found at '{SOURCE_IMAGE_PATH}'")
        print("Please update the SOURCE_IMAGE_PATH variable in this script.")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print(f"Loading source image: {SOURCE_IMAGE_PATH}")

    # --- 4. Loop through each blur radius, apply filter, and save ---
    for radius in BLUR_RADII:
        print(f"Applying blur radius: {radius}")
        
        # Apply the Gaussian Blur filter
        # If radius is 0, just use the original image [cite: 30-31]
        if radius == 0:
            blurred_img = img
        else:
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            
        # Create the new filename
        # e.g., "my_source_image_blur_0.jpg"
        output_filename = f"{base_name}_blur_{radius}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Save the new blurred image
        blurred_img.save(output_path)
        print(f"Saved: {output_path}")

    print("\nDataset generation complete!")

if __name__ == "__main__":
    create_blurred_dataset()