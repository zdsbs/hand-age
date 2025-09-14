#!/usr/bin/env python3
"""
Prepare dorsal hand images dataset for training.
Splits by person ID to avoid data leakage.
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

def load_dorsal_data(csv_path="HandInfo.csv"):
    """Load only dorsal hand images grouped by person"""
    persons = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'dorsal' in row['aspectOfHand']:
                person_id = row['id'].lstrip('0') or '0'
                persons[person_id].append({
                    'image': row['imageName'],
                    'age': int(row['age']),
                    'aspect': row['aspectOfHand'],
                    'gender': row['gender'],
                    'skinColor': row['skinColor']
                })

    return dict(persons)

def split_dataset(persons, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split persons into train/val/test sets"""
    random.seed(seed)

    person_ids = list(persons.keys())
    random.shuffle(person_ids)

    n_total = len(person_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = person_ids[:n_train]
    val_ids = person_ids[n_train:n_train + n_val]
    test_ids = person_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids

def create_splits(persons, train_ids, val_ids, test_ids):
    """Create dataset splits with all images from selected persons"""
    splits = {
        'train': [],
        'val': [],
        'test': []
    }

    for person_id in train_ids:
        for img_data in persons[person_id]:
            splits['train'].append({
                'person_id': person_id,
                'image': img_data['image'],
                'age': img_data['age'],
                'aspect': img_data['aspect'],
                'gender': img_data['gender'],
                'skinColor': img_data['skinColor']
            })

    for person_id in val_ids:
        for img_data in persons[person_id]:
            splits['val'].append({
                'person_id': person_id,
                'image': img_data['image'],
                'age': img_data['age'],
                'aspect': img_data['aspect'],
                'gender': img_data['gender'],
                'skinColor': img_data['skinColor']
            })

    for person_id in test_ids:
        for img_data in persons[person_id]:
            splits['test'].append({
                'person_id': person_id,
                'image': img_data['image'],
                'age': img_data['age'],
                'aspect': img_data['aspect'],
                'gender': img_data['gender'],
                'skinColor': img_data['skinColor']
            })

    return splits

def save_splits(splits, output_dir="dataset_splits"):
    """Save splits to JSON files"""
    Path(output_dir).mkdir(exist_ok=True)

    for split_name, data in splits.items():
        output_path = Path(output_dir) / f"{split_name}.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {split_name}: {len(data)} images")

    # Save test set separately for verification
    test_verification = random.sample(splits['test'], min(20, len(splits['test'])))
    with open(Path(output_dir) / "test_verification_samples.json", 'w') as f:
        json.dump(test_verification, f, indent=2)
    print(f"Saved {len(test_verification)} test samples for manual verification")

def print_statistics(splits):
    """Print dataset statistics"""
    print("\n" + "="*60)
    print("Dataset Statistics (Dorsal Images Only)")
    print("="*60)

    for split_name, data in splits.items():
        print(f"\n{split_name.upper()} SET:")
        print(f"  Total images: {len(data)}")
        print(f"  Unique persons: {len(set(d['person_id'] for d in data))}")

        # Age statistics
        ages = [d['age'] for d in data]
        print(f"  Age range: {min(ages)} - {max(ages)}")
        print(f"  Mean age: {sum(ages)/len(ages):.1f}")

        # Age distribution
        age_bins = defaultdict(int)
        for age in ages:
            if age < 20:
                age_bins['<20'] += 1
            elif age < 30:
                age_bins['20-29'] += 1
            elif age < 40:
                age_bins['30-39'] += 1
            elif age < 50:
                age_bins['40-49'] += 1
            elif age < 60:
                age_bins['50-59'] += 1
            else:
                age_bins['60+'] += 1

        print("  Age distribution:")
        for bin_name in ['<20', '20-29', '30-39', '40-49', '50-59', '60+']:
            if bin_name in age_bins:
                count = age_bins[bin_name]
                pct = (count / len(ages)) * 100
                print(f"    {bin_name}: {count} ({pct:.1f}%)")

        # Aspect distribution
        aspects = defaultdict(int)
        for d in data:
            aspects[d['aspect']] += 1
        print("  Hand aspects:")
        for aspect, count in aspects.items():
            print(f"    {aspect}: {count}")

def main():
    print("Loading dorsal hand data...")
    persons = load_dorsal_data()
    print(f"Loaded {len(persons)} unique persons with dorsal hand images")

    # Split by person ID
    train_ids, val_ids, test_ids = split_dataset(persons)
    print(f"\nSplit: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test persons")

    # Create splits with all images
    splits = create_splits(persons, train_ids, val_ids, test_ids)

    # Print statistics
    print_statistics(splits)

    # Save splits
    save_splits(splits)

    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("Splits saved to dataset_splits/")
    print("Test verification samples saved for manual checking")

if __name__ == "__main__":
    main()