#!/usr/bin/env python3
"""
Create a compromise dataset - more age balance than original,
but more data than the extreme balanced version.
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

def load_dorsal_data_by_age(csv_path="HandInfo.csv"):
    """Load dorsal images grouped by person and age"""
    persons_by_age = defaultdict(lambda: defaultdict(list))

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'dorsal' in row['aspectOfHand']:
                person_id = row['id'].lstrip('0') or '0'
                age = int(row['age'])

                persons_by_age[age][person_id].append({
                    'image': row['imageName'],
                    'age': age,
                    'aspect': row['aspectOfHand'],
                    'gender': row['gender'],
                    'skinColor': row['skinColor']
                })

    return persons_by_age

def create_compromise_dataset(persons_by_age):
    """Create a compromise dataset with reasonable age distribution"""

    compromise_data = []

    # Strategy: Take more from young adults but limit the extreme bias
    # Include all available older adults to improve age range coverage

    age_sampling_strategy = {
        # Young adults (18-25): Heavily represented but capped
        18: {'max_persons': 1},   # Take all (only 1 available)
        19: {'max_persons': 2},   # Take all (only 2 available)
        20: {'max_persons': 15},  # Sample from 39 available (reduce bias)
        21: {'max_persons': 25},  # Sample from 80 available (reduce bias)
        22: {'max_persons': 15},  # Sample from 36 available
        23: {'max_persons': 10},  # Sample from 17 available
        24: {'max_persons': 2},   # Take all (only 2 available)
        25: {'max_persons': 2},   # Take all (only 2 available)

        # Adults (26-35): Take all available
        26: {'max_persons': 2},   # Take all available
        27: {'max_persons': 1},   # Take all available
        28: {'max_persons': 1},   # Take all available
        29: {'max_persons': 1},   # Take all available
        30: {'max_persons': 1},   # Take all available

        # Middle age and older: Take all available
        36: {'max_persons': 1},   # Take all available
        43: {'max_persons': 1},   # Take all available
        54: {'max_persons': 1},   # Take all available
        70: {'max_persons': 1},   # Take all available
        75: {'max_persons': 1},   # Take all available
    }

    print("Compromise sampling strategy:")
    total_persons = 0
    total_images = 0

    for age in sorted(age_sampling_strategy.keys()):
        if age not in persons_by_age:
            continue

        available_persons = len(persons_by_age[age])
        max_persons = age_sampling_strategy[age]['max_persons']
        selected_persons = min(available_persons, max_persons)

        # Sample persons for this age
        person_ids = list(persons_by_age[age].keys())
        if selected_persons < len(person_ids):
            selected_person_ids = random.sample(person_ids, selected_persons)
        else:
            selected_person_ids = person_ids

        # Add all images from selected persons
        age_images = 0
        for person_id in selected_person_ids:
            for img_data in persons_by_age[age][person_id]:
                compromise_data.append({
                    'person_id': person_id,
                    'image': img_data['image'],
                    'age': img_data['age'],
                    'aspect': img_data['aspect'],
                    'gender': img_data['gender'],
                    'skinColor': img_data['skinColor']
                })
                age_images += 1

        print(f"  Age {age:2d}: {selected_persons:2d}/{available_persons:2d} persons, {age_images:3d} images")
        total_persons += selected_persons
        total_images += age_images

    print(f"\nTotal: {total_persons} persons, {total_images} images")
    return compromise_data

def split_compromise_dataset(compromise_data, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split compromise data by persons"""
    random.seed(seed)

    # Group by person ID
    persons_data = defaultdict(list)
    for item in compromise_data:
        persons_data[item['person_id']].append(item)

    person_ids = list(persons_data.keys())
    random.shuffle(person_ids)

    n_total = len(person_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_persons = person_ids[:n_train]
    val_persons = person_ids[n_train:n_train + n_val]
    test_persons = person_ids[n_train + n_val:]

    # Create splits
    splits = {'train': [], 'val': [], 'test': []}

    for person_id in train_persons:
        splits['train'].extend(persons_data[person_id])

    for person_id in val_persons:
        splits['val'].extend(persons_data[person_id])

    for person_id in test_persons:
        splits['test'].extend(persons_data[person_id])

    return splits

def print_compromise_statistics(splits):
    """Print statistics for compromise dataset"""

    print(f"\n{'='*60}")
    print("COMPROMISE DATASET STATISTICS")
    print(f"{'='*60}")

    for split_name, data in splits.items():
        print(f"\n{split_name.upper()} SET:")
        print(f"  Total images: {len(data)}")
        print(f"  Unique persons: {len(set(d['person_id'] for d in data))}")

        # Age distribution
        ages = [d['age'] for d in data]
        if ages:
            print(f"  Age range: {min(ages)} - {max(ages)}")
            print(f"  Mean age: {sum(ages)/len(ages):.1f}")

            # Detailed age distribution
            age_counts = defaultdict(int)
            for age in ages:
                age_counts[age] += 1

            print("  Age distribution:")
            for age in sorted(age_counts.keys()):
                count = age_counts[age]
                pct = (count / len(ages)) * 100
                print(f"    Age {age:2d}: {count:3d} images ({pct:4.1f}%)")

def save_compromise_splits(splits, output_dir="compromise_dataset_splits"):
    """Save compromise splits"""
    Path(output_dir).mkdir(exist_ok=True)

    for split_name, data in splits.items():
        output_path = Path(output_dir) / f"{split_name}.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {split_name}: {len(data)} images to {output_path}")

def main():
    print("Creating compromise dataset...")
    print("Strategy: Reduce young adult bias but keep enough data for training")

    # Load data
    persons_by_age = load_dorsal_data_by_age()

    # Create compromise dataset
    compromise_data = create_compromise_dataset(persons_by_age)

    if not compromise_data:
        print("No compromise data created")
        return

    # Split dataset
    splits = split_compromise_dataset(compromise_data)

    # Print statistics
    print_compromise_statistics(splits)

    # Save splits
    save_compromise_splits(splits)

    print(f"\n{'='*60}")
    print("Compromise dataset created!")
    print("Less biased than original, more data than extreme balanced")
    print("Use 'compromise_dataset_splits/' for training")

if __name__ == "__main__":
    main()