import os
import re

# Regular expression to match checkpoint files
pattern = re.compile(r'(\d+)(backbone|header)\.pth$')

# Walk through the directory tree
def clean_folder(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Check if the current folder is a leaf (no subdirectories)
        if not dirnames:
            checkpoints = {}

            # Identify checkpoint files and group them by numeric prefix
            for filename in filenames:
                match = pattern.match(filename)
                if match:
                    num, ctype = match.groups()
                    if num not in checkpoints:
                        checkpoints[num] = set()
                    checkpoints[num].add(ctype)

            # Find the highest numeric prefix with both backbone and header
            valid_pairs = [int(num) for num, types in checkpoints.items() if len(types) == 2]

            if valid_pairs:
                max_num = str(max(valid_pairs))

                # Define the files to keep
                keep_files = {f'{max_num}backbone.pth', f'{max_num}header.pth', 'training.log'}

                # Delete all other files
                for filename in filenames:
                    if filename not in keep_files:
                        file_path = os.path.join(dirpath, filename)
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")

if __name__ == "__main__":
    root_folder = input("Enter the root folder path: ").strip()
    clean_folder(root_folder)
    print("Cleanup completed.")
