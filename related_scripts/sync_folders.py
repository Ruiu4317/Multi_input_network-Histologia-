import os
import shutil

def sync_folders_by_name(folder_a, folder_b, dry_run=True):
    """
    Synchronizes folder A with folder B by removing files from A that are not present in B.
    
    This function compares filenames (without full path) and deletes extra files in folder A
    if they don't have a counterpart in folder B.

    Args:
        folder_a (str): Path to the folder from which files will be removed (source to clean up).
        folder_b (str): Path to the reference folder (source of truth).
        dry_run (bool): If True, only shows what would be deleted without performing deletion.
    """
    # Get sets of filenames (without paths) in both folders
    files_a = set(os.listdir(folder_a))
    files_b = set(os.listdir(folder_b))

    # Find files present in A but missing in B
    files_to_delete = files_a - files_b

    if not files_to_delete:
        print(f"✅ No extra files found in '{folder_a}'.")
        return

    print(f"📁 Found {len(files_to_delete)} file(s) to remove from '{folder_a}':")
    for fname in sorted(files_to_delete):
        print(f"  🔻 {fname}")

    if dry_run:
        print("\nℹ️ Dry run mode: no files will be deleted. Review the list above.")
    else:
        print(f"\n🗑 Removing files from '{folder_a}'...")
        for fname in files_to_delete:
            file_path = os.path.join(folder_a, fname)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"✅ Deleted file: {fname}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"✅ Deleted directory: {fname}")
            except Exception as e:
                print(f"❌ Error deleting {fname}: {e}")


# ————————————————————————
# Usage example
# ————————————————————————

# Define paths
folder_b = '.../high_mag_512'  # Reference folder
folder_a = '.../high_mag_2048'  # Folder to clean

# Step 1: Preview what will be deleted (recommended)
# sync_folders_by_name(folder_a, folder_b, dry_run=True)

# Step 2: After reviewing, run with dry_run=False to perform actual deletion
sync_folders_by_name(folder_a, folder_b, dry_run=False)
