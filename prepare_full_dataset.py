#!/usr/bin/env python3
"""
Use ALL dorsal images for training - no compromises!
Like your friend's approach.
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

def load_all_dorsal_data(csv_path="HandInfo.csv"):
    """Load ALL dorsal hand images"""
    all_data = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'dorsal' in row['aspectOfHand']:
                person_id = row['id'].lstrip('0') or '0'
                all_data.append({
                    'person_id': person_id,
                    'image': row['imageName'],
                    'age': int(row['age']),
                    'aspect': row['aspectOfHand'],
                    'gender': row['gender'],
                    'skinColor': row['skinColor']
                })

    return all_data

def split_full_dataset(all_data, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Simple random split of all data - maximize training data"""
    random.seed(seed)

    # Shuffle all data
    random.shuffle(all_data)

    n_total = len(all_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train + n_val]
    test_data = all_data[n_train + n_val:]

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

def print_full_statistics(splits):
    """Print statistics for full dataset"""

    print(f"\n{'='*60}")
    print("FULL DATASET STATISTICS (ALL DORSAL IMAGES)")
    print(f"{'='*60}")

    for split_name, data in splits.items():
        print(f"\n{split_name.upper()} SET:")
        print(f"  Total images: {len(data)}")

        # Age distribution
        ages = [d['age'] for d in data]
        if ages:
            print(f"  Age range: {min(ages)} - {max(ages)}")
            print(f"  Mean age: {sum(ages)/len(ages):.1f}")

            # Age counts
            age_counts = defaultdict(int)
            for age in ages:
                age_counts[age] += 1

            print("  Age distribution (showing key ages):")
            key_ages = [18, 19, 20, 21, 22, 23, 25, 27, 30, 36, 43, 54, 70, 75]
            for age in key_ages:
                if age in age_counts:
                    count = age_counts[age]
                    pct = (count / len(ages)) * 100
                    print(f"    Age {age:2d}: {count:4d} images ({pct:4.1f}%)")

def save_full_splits(splits, output_dir="full_dataset_splits"):
    """Save full dataset splits"""
    Path(output_dir).mkdir(exist_ok=True)

    for split_name, data in splits.items():
        output_path = Path(output_dir) / f"{split_name}.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {split_name}: {len(data)} images to {output_path}")

def main():
    print("Creating FULL dataset (like your friend's approach)...")
    print("Using ALL dorsal images for maximum training data!")

    # Load ALL dorsal data
    all_data = load_all_dorsal_data()
    print(f"Loaded {len(all_data)} total dorsal images")

    # Simple split - maximize training data (80/10/10)
    splits = split_full_dataset(all_data)

    # Print statistics
    print_full_statistics(splits)

    # Save splits
    save_full_splits(splits)

    print(f"\n{'='*60}")
    print("Full dataset created!")
    print("This should match your friend's approach")
    print("Use 'full_dataset_splits/' for training")

    # Compare to previous approaches
    print(f"\n{'='*60}")
    print("COMPARISON:")
    print(f"  Original biased: 3,724 training images (threw away older ages)")
    print(f"  Balanced: 231 training images (too small)")
    print(f"  Compromise: 1,824 training images (still too small)")
    print(f"  FULL: {len(splits['train'])} training images (ALL DATA!)")

if __name__ == "__main__":
    main()