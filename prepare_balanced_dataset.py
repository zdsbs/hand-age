#!/usr/bin/env python3
"""
Create a balanced dataset by age groups to avoid age bias.
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

def create_balanced_age_groups(persons_by_age):
    """Create balanced age groups"""

    # Define age groups to balance
    age_groups = {
        'young_adult': list(range(18, 26)),      # 18-25
        'adult': list(range(26, 36)),           # 26-35
        'middle_age': list(range(36, 55)),      # 36-54
        'older_adult': list(range(55, 80))      # 55-79
    }

    print("Available ages and person counts:")
    available_persons_by_group = {}

    for group_name, ages in age_groups.items():
        persons_in_group = []
        for age in ages:
            if age in persons_by_age:
                persons_in_group.extend(list(persons_by_age[age].keys()))
                print(f"  Age {age}: {len(persons_by_age[age])} persons")

        available_persons_by_group[group_name] = len(set(persons_in_group))
        print(f"Group {group_name}: {available_persons_by_group[group_name]} unique persons")

    return age_groups, available_persons_by_group

def sample_balanced_dataset(persons_by_age, age_groups, max_persons_per_group=None):
    """Sample balanced dataset across age groups"""

    # Determine how many persons to sample per group
    group_person_counts = {}
    for group_name, ages in age_groups.items():
        all_persons = set()
        for age in ages:
            if age in persons_by_age:
                all_persons.update(persons_by_age[age].keys())
        group_person_counts[group_name] = len(all_persons)

    # Find the limiting group (smallest available)
    min_available = min(count for count in group_person_counts.values() if count > 0)

    if max_persons_per_group:
        target_per_group = min(min_available, max_persons_per_group)
    else:
        target_per_group = min_available

    print(f"\nBalancing strategy:")
    print(f"Target persons per group: {target_per_group}")

    balanced_data = []

    for group_name, ages in age_groups.items():
        print(f"\nProcessing {group_name} (ages {min(ages)}-{max(ages)}):")

        # Collect all persons in this age group
        group_persons = {}
        for age in ages:
            if age in persons_by_age:
                group_persons.update(persons_by_age[age])

        if len(group_persons) == 0:
            print(f"  No data available - skipping")
            continue

        # Sample persons for this group
        available_persons = list(group_persons.keys())
        if len(available_persons) < target_per_group:
            print(f"  Warning: Only {len(available_persons)} persons available, using all")
            selected_persons = available_persons
        else:
            selected_persons = random.sample(available_persons, target_per_group)

        # Add all images from selected persons
        group_images = 0
        for person_id in selected_persons:
            person_age = None
            for age in ages:
                if age in persons_by_age and person_id in persons_by_age[age]:
                    person_age = age
                    for img_data in persons_by_age[age][person_id]:
                        balanced_data.append({
                            'person_id': person_id,
                            'image': img_data['image'],
                            'age': img_data['age'],
                            'aspect': img_data['aspect'],
                            'gender': img_data['gender'],
                            'skinColor': img_data['skinColor'],
                            'age_group': group_name
                        })
                        group_images += 1
                    break

        print(f"  Selected {len(selected_persons)} persons with {group_images} images")

    return balanced_data

def split_balanced_dataset(balanced_data, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split balanced data by persons to avoid leakage"""
    random.seed(seed)

    # Group by person ID to maintain person-level splits
    persons_data = defaultdict(list)
    for item in balanced_data:
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

def print_balanced_statistics(splits):
    """Print statistics for balanced dataset"""

    print(f"\n{'='*60}")
    print("BALANCED DATASET STATISTICS")
    print(f"{'='*60}")

    for split_name, data in splits.items():
        print(f"\n{split_name.upper()} SET:")
        print(f"  Total images: {len(data)}")
        print(f"  Unique persons: {len(set(d['person_id'] for d in data))}")

        # Age distribution
        ages = [d['age'] for d in data]
        print(f"  Age range: {min(ages)} - {max(ages)}")
        print(f"  Mean age: {sum(ages)/len(ages):.1f}")

        # Age group distribution
        age_groups = defaultdict(int)
        for d in data:
            age_groups[d.get('age_group', 'unknown')] += 1

        print("  Age group distribution:")
        for group, count in age_groups.items():
            pct = (count / len(data)) * 100
            print(f"    {group}: {count} ({pct:.1f}%)")

def save_balanced_splits(splits, output_dir="balanced_dataset_splits"):
    """Save balanced splits"""
    Path(output_dir).mkdir(exist_ok=True)

    for split_name, data in splits.items():
        output_path = Path(output_dir) / f"{split_name}.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {split_name}: {len(data)} images to {output_path}")

def main():
    print("Creating balanced age distribution dataset...")

    # Load data grouped by age
    persons_by_age = load_dorsal_data_by_age()

    # Analyze age groups
    age_groups, available_counts = create_balanced_age_groups(persons_by_age)

    # Sample balanced dataset
    # Limit to reasonable size per group to avoid huge dataset
    balanced_data = sample_balanced_dataset(persons_by_age, age_groups, max_persons_per_group=50)

    if not balanced_data:
        print("No balanced data created - insufficient age diversity")
        return

    # Split dataset
    splits = split_balanced_dataset(balanced_data)

    # Print statistics
    print_balanced_statistics(splits)

    # Save splits
    save_balanced_splits(splits)

    print(f"\n{'='*60}")
    print("Balanced dataset created!")
    print("Use 'balanced_dataset_splits/' for training")
    print("This should give much better age prediction across all ages")

if __name__ == "__main__":
    main()