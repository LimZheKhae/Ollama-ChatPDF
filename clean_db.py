from app import db_path
import os
import shutil

def clear_directory(directory_path: str) -> None:
    """
    Remove all files and folders within the specified directory.

    Args:
        directory_path (str): The path to the directory to be cleared.
    """
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    # Iterate over all files and folders in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        # Remove files or directories
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # Remove file or symbolic link
            print(f"Deleted file: {item_path}")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove directory and its contents
            print(f"Deleted directory: {item_path}")

    print(f"All files and folders in {directory_path} have been removed.")

try:
    clear_directory(db_path)
except Exception as e:
    print(f"Ensure to close your app.py before running python clean_db.py")