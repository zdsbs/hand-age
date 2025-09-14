#!/usr/bin/env python3
import csv
import json
import random
from collections import defaultdict

def load_hand_data(csv_path="HandInfo.csv"):
    """Load and organize hand data by user ID"""
    users = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = row['id'].lstrip('0') or '0'
            users[user_id].append({
                'age': int(row['age']),
                'image': row['imageName'],
                'aspect': row['aspectOfHand']
            })

    return users

def create_finetune_examples(users, selected_ids, base_url="https://uhhhhhh"):
    """Create fine-tuning examples in OpenAI format"""
    examples = []
    used_images = []  # Track which images we're using

    for user_id in selected_ids:
        if user_id not in users:
            print(f"Warning: User ID {user_id} not found")
            continue

        user_data = users[user_id]
        age = user_data[0]['age']  # Age is consistent for all images of same person

        # Sample up to 5 images per user for variety
        sampled_images = random.sample(user_data, min(5, len(user_data)))

        for image_data in sampled_images:
            used_images.append(image_data['image'])  # Track this image
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an assistant that identifies age based on hands."
                    },
                    {
                        "role": "user",
                        "content": "how old is this person?"
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"{base_url}/{image_data['image']}"
                                }
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": str(age)
                    }
                ]
            }
            examples.append(example)

    return examples, used_images

def save_dataset(examples, output_path="finetune_dataset.jsonl"):
    """Save examples to JSONL format"""
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"Saved {len(examples)} examples to {output_path}")

def save_csv_subset(used_images, csv_path="HandInfo.csv", output_path="finetune_subset.csv"):
    """Save CSV rows for specific images used in fine-tuning"""
    rows_to_save = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            if row['imageName'] in used_images:
                rows_to_save.append(row)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_save)

    print(f"Saved {len(rows_to_save)} CSV rows for used images to {output_path}")

def main():
    # Load data
    users = load_hand_data()
    print(f"Loaded data for {len(users)} users")

    # Get list of all user IDs and show some stats
    all_user_ids = list(users.keys())

    print("\nFirst 10 user IDs and their ages:")
    for i, user_id in enumerate(sorted(all_user_ids)[:10]):
        age = users[user_id][0]['age']
        num_images = len(users[user_id])
        print(f"  {user_id}: {age} years old, {num_images} images")

    # For initial validation, use just 2 user IDs
    print("\n" + "="*60)
    print("Creating validation dataset with 2 users...")

    # Select 2 diverse ages for validation
    selected_ids = ['0', '1']  # Adjust based on actual IDs in your data

    # Let's first check what IDs actually exist
    print("\nActual user IDs available (first 20):")
    for user_id in sorted(all_user_ids)[:20]:
        print(f"  '{user_id}'")

    # Now select 2 that actually exist
    selected_ids = sorted(all_user_ids)[:2]
    print(f"\nSelected user IDs for validation: {selected_ids}")

    # Create examples
    examples, used_images = create_finetune_examples(users, selected_ids)

    # Save to file
    save_dataset(examples, "finetune_validation.jsonl")

    # Save CSV subset for the specific images used
    save_csv_subset(used_images, output_path="finetune_validation_subset.csv")

    # Print which images were used
    print(f"\nUsed {len(used_images)} images:")
    for img in used_images[:10]:  # Show first 10
        print(f"  {img}")

    # Display first example for verification
    if examples:
        print("\nFirst example (formatted):")
        print(json.dumps(examples[0], indent=2))

    print("\n" + "="*60)
    print("To create full dataset with 10 users, run:")
    print("  python create_finetune_dataset.py --full")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Full dataset mode
        users = load_hand_data()
        all_user_ids = list(users.keys())

        # Select 10 random users
        selected_ids = random.sample(all_user_ids, min(10, len(all_user_ids)))
        print(f"Creating full dataset with users: {selected_ids}")

        examples, used_images = create_finetune_examples(users, selected_ids)
        save_dataset(examples, "finetune_full.jsonl")

        # Save CSV subset for the specific images used
        save_csv_subset(used_images, output_path="finetune_full_subset.csv")

        print(f"\nDataset statistics:")
        print(f"  Users: {len(selected_ids)}")
        print(f"  Total examples: {len(examples)}")

        # Show age distribution
        age_counts = defaultdict(int)
        for ex in examples:
            age = ex['messages'][3]['content']
            age_counts[age] += 1

        print("\nAge distribution in dataset:")
        for age, count in sorted(age_counts.items(), key=lambda x: int(x[0])):
            print(f"  Age {age}: {count} examples")
    else:
        main()