import os
import random
import sys
from pathlib import Path

def split_dataset(ratios: dict):
    dataset_path = Path('dataset/')
    source_images_dir = dataset_path / 'images'
    source_labels_dir = dataset_path / 'labels'
    
    print(f"Starting dataset split with ratio: "
          f"Train={int(ratios['train']*100)}%, "
          f"Validation={int(ratios['val']*100)}%, "
          f"Test={int(ratios['test']*100)}%")

    try:
        all_image_files = [f for f in os.listdir(source_images_dir) if os.path.isfile(source_images_dir / f)]
        if not all_image_files:
            print(f"Error: No image files found in '{source_images_dir}'.")
            return
    except FileNotFoundError:
        print(f"Error: The directory '{source_images_dir}' does not exist.")
        return

    random.shuffle(all_image_files)
    total_files = len(all_image_files)

    train_end = int(total_files * ratios['train'])
    val_end = train_end + int(total_files * ratios['val'])

    train_files = all_image_files[:train_end]
    val_files = all_image_files[train_end:val_end]
    test_files = all_image_files[val_end:]

    sets = {'train': train_files, 'val': val_files, 'test': test_files}
    for set_name in sets.keys():
        os.makedirs(source_images_dir / set_name, exist_ok=True)
        os.makedirs(source_labels_dir / set_name, exist_ok=True)

    def move_files(file_list, set_name):
        moved_count = 0
        img_dest_dir = source_images_dir / set_name
        lbl_dest_dir = source_labels_dir / set_name
        
        for image_name in file_list:
            base_name = Path(image_name).stem
            label_name = f"{base_name}.txt"

            image_source = source_images_dir / image_name
            label_source = source_labels_dir / label_name
            
            if image_source.exists() and label_source.exists():
                os.rename(image_source, img_dest_dir / image_name)
                os.rename(label_source, lbl_dest_dir / label_name)
                moved_count += 1
            else:
                print(f"Warning: Missing image or label for '{base_name}'. Skipping.")
        return moved_count

    for set_name, file_list in sets.items():
        count = move_files(file_list, set_name)
        print(f"Moved {count} image-label pairs to '{set_name}'.")

    print("\nDataset splitting complete!")

if __name__ == '__main__':
    # Default percentages
    percentages = {'train': 70, 'val': 20, 'test': 10}

    try:
        cli_args = dict(arg.split('=', 1) for arg in sys.argv[1:])
        for key in cli_args:
            if key in percentages:
                percentages[key] = int(cli_args[key])
            else:
                print(f"⚠️ Warning: Ignoring unknown argument '{key}'")
    except (ValueError, TypeError):
        print("Error: Invalid argument format. Use key=value (e.g., train=70).")
        sys.exit(1)

    if sum(percentages.values()) != 100:
        print(f"Error: The sum of percentages must be 100. Current sum is {sum(percentages.values())}.")
        sys.exit(1)

    ratios = {key: val / 100.0 for key, val in percentages.items()}
    
    split_dataset(ratios)