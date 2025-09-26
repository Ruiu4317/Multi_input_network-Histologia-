import os
import pandas as pd
import random

# ————————————————————————
# 1. Paths and parameters
# ————————————————————————

folder_path = '.../high_mag_1024'
output_csv = '.../labels.csv'

# ————————————————————————
# 2. Label mapping and classes to ignore
# ————————————————————————

class_to_label = {
    'normal': 0,
    'pap': 1, 'fol': 1, 'sol': 1, 'pap+fol': 1, 'pap+sol': 1,
    'pap+fol+sol': 1, 'fol+sol': 1, 'skler': 1
}

# Classes that should be excluded from the dataset
ignore_classes = {'embol', 'psamoma', 'back', 'background'}

# ————————————————————————
# 3. Collect files by class
# ————————————————————————

print(f"Scanning folder: {folder_path}")

if not os.path.exists(folder_path):
    raise FileNotFoundError(f"Folder not found: {folder_path}")

# Dictionary: label → list of filenames (without extension)
class_files = {}
for filename in os.listdir(folder_path):
    if not filename.lower().endswith('.png'):
        continue  # Skip non-PNG files

    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    if len(parts) < 2:
        print(f"⚠ Skipping file (name too short): {filename}")
        continue

    # Assume the last part is the class label
    class_name = parts[-1].lower()

    if class_name in ignore_classes:
        print(f"🗑️  Ignored: {filename} (class={class_name})")
        continue

    if class_name not in class_to_label:
        print(f"❓ Unknown class: {filename} → '{class_name}'")
        continue

    label = class_to_label[class_name]

    if label not in class_files:
        class_files[label] = []
    class_files[label].append(name_without_ext)

# ————————————————————————
# 4. Balancing: select equal number of samples per class
# ————————————————————————

# Option A: use the size of the smallest class (strict balance)
max_per_class = min(len(files) for files in class_files.values())

# Option B (alternative): cap at a fixed number (e.g., 500), but not more than available
# max_per_class = min(500, min(len(files) for files in class_files.values()))

balanced_data = []

for label, files in class_files.items():
    selected = random.sample(files, min(max_per_class, len(files)))
    for base_name in selected:
        balanced_data.append({'filename': base_name, 'label': label})

# Shuffle the final dataset to mix classes
random.shuffle(balanced_data)

# ————————————————————————
# 5. Save to CSV
# ————————————————————————

if not balanced_data:
    print("❌ No data to save.")
else:
    df = pd.DataFrame(balanced_data)
    df.to_csv(output_csv, index=False)
    print(f"✅ labels_2.csv successfully created: {output_csv}")
    print(f"📊 Total entries: {len(df)}")
    print(f"🔢 Label distribution:\n{df['label'].value_counts().sort_index()}")

    # Additional: print balance summary
    print("\n📈 Class balance:")
    for label, count in df['label'].value_counts().items():
        print(f"  Class {label}: {count} files")
