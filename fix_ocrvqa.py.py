import os
from PIL import Image

def convert_to_jpg(old_path, jpg_path):
    """
    Open a .gif/.png file using PIL and save it as .jpg.
    """
    if os.path.exists(jpg_path):
        return True
    try:
        img = Image.open(old_path)
        # This ensures the file is converted in RGB mode, not P mode
        img = img.convert('RGB')
        img.save(jpg_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    return True


def main():
    # Directory containing the .gif images
    directory = "./playground/data/ocr_vqa/images/"

    # Go through each file in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(".gif") or filename.lower().endswith(".png"):
            # Form the full file paths
            old_path = os.path.join(directory, filename)
            jpg_path = os.path.join(directory, os.path.splitext(filename)[0] + ".jpg")

            # Convert the .gif to .jpg
            if convert_to_jpg(old_path, jpg_path):
                print(f"Converted {old_path} to {jpg_path}")
            else:
                print(f"Failed to convert {old_path}")

            # Optionally, if you want to remove the original .gif files, uncomment the line below:
            # os.remove(old_path)
            
if __name__ == "__main__":
    main()