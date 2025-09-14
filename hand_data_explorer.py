#!/usr/bin/env python3
import csv
from collections import defaultdict
from typing import Dict, List, Any

class HandDataExplorer:
    def __init__(self, csv_path: str = "HandInfo.csv"):
        self.data = []
        self.persons = {}
        self._load_data(csv_path)
        self.persons = self._aggregate_by_person()

    def _load_data(self, csv_path: str):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['id'] = row['id'].lstrip('0') or '0'  # Remove leading zeros, keep at least '0' if all zeros
                row['age'] = int(row['age'])
                row['accessories'] = int(row['accessories'])
                row['nailPolish'] = int(row['nailPolish'])
                row['irregularities'] = int(row['irregularities'])
                self.data.append(row)

    def _aggregate_by_person(self) -> Dict[str, Dict[str, Any]]:
        persons = defaultdict(lambda: {
            'age': None,
            'gender': None,
            'skinColor': None,
            'accessories': set(),
            'nailPolish': set(),
            'aspectOfHand': set(),
            'images': [],
            'irregularities': set(),
            'num_images': 0
        })

        for row in self.data:
            person_id = row['id']
            person = persons[person_id]

            person['age'] = row['age']
            person['gender'] = row['gender']
            person['skinColor'] = row['skinColor']
            person['accessories'].add(row['accessories'])
            person['nailPolish'].add(row['nailPolish'])
            person['aspectOfHand'].add(row['aspectOfHand'])
            person['images'].append(row['imageName'])
            person['irregularities'].add(row['irregularities'])
            person['num_images'] += 1

        for person_id in persons:
            persons[person_id]['accessories'] = list(persons[person_id]['accessories'])
            persons[person_id]['nailPolish'] = list(persons[person_id]['nailPolish'])
            persons[person_id]['aspectOfHand'] = list(persons[person_id]['aspectOfHand'])
            persons[person_id]['irregularities'] = list(persons[person_id]['irregularities'])

        return dict(persons)

    def get_person(self, person_id: str) -> Dict[str, Any]:
        return self.persons.get(person_id)

    def list_all_persons(self) -> List[str]:
        return list(self.persons.keys())

    def get_stats(self) -> Dict[str, Any]:
        total_persons = len(self.persons)
        total_images = len(self.data)

        age_distribution = defaultdict(int)
        gender_distribution = defaultdict(int)
        skin_color_distribution = defaultdict(int)

        for row in self.data:
            age_distribution[row['age']] += 1
            gender_distribution[row['gender']] += 1
            skin_color_distribution[row['skinColor']] += 1

        return {
            'total_persons': total_persons,
            'total_images': total_images,
            'avg_images_per_person': total_images / total_persons if total_persons > 0 else 0,
            'age_distribution': dict(age_distribution),
            'gender_distribution': dict(gender_distribution),
            'skin_color_distribution': dict(skin_color_distribution)
        }

    def search_by_criteria(self, age=None, gender=None, skin_color=None):
        matching_persons = []

        for person_id, data in self.persons.items():
            if age and data['age'] != age:
                continue
            if gender and data['gender'] != gender:
                continue
            if skin_color and data['skinColor'] != skin_color:
                continue
            matching_persons.append(person_id)

        return matching_persons


def main():
    explorer = HandDataExplorer()

    print("=" * 60)
    print("Hand Data Explorer")
    print("=" * 60)

    while True:
        print("\nOptions:")
        print("1. Look up person by ID")
        print("2. List all person IDs")
        print("3. Show dataset statistics")
        print("4. Search by criteria")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == '1':
            person_id = input("Enter person ID: ").strip()
            person_data = explorer.get_person(person_id)

            if person_data:
                print(f"\n--- Person {person_id} ---")
                print(f"Age: {person_data['age']}")
                print(f"Gender: {person_data['gender']}")
                print(f"Skin Color: {person_data['skinColor']}")
                print(f"Accessories: {person_data['accessories']}")
                print(f"Nail Polish: {person_data['nailPolish']}")
                print(f"Hand Aspects: {person_data['aspectOfHand']}")
                print(f"Irregularities: {person_data['irregularities']}")
                print(f"Number of Images: {person_data['num_images']}")

                show_images = input("\nShow image names? (y/n): ").strip().lower()
                if show_images == 'y':
                    print("\nImages:")
                    for img in person_data['images']:
                        print(f"  - {img}")
            else:
                print(f"Person ID '{person_id}' not found.")

        elif choice == '2':
            all_persons = explorer.list_all_persons()
            print(f"\nTotal persons: {len(all_persons)}")
            show_all = input("Show all IDs? (y/n): ").strip().lower()

            if show_all == 'y':
                for person_id in sorted(all_persons):
                    person = explorer.get_person(person_id)
                    print(f"  {person_id}: {person['age']}y {person['gender']}, {person['num_images']} images")

        elif choice == '3':
            stats = explorer.get_stats()
            print("\n--- Dataset Statistics ---")
            print(f"Total Persons: {stats['total_persons']}")
            print(f"Total Images: {stats['total_images']}")
            print(f"Avg Images per Person: {stats['avg_images_per_person']:.2f}")

            print("\nAge Distribution:")
            for age, count in sorted(stats['age_distribution'].items()):
                print(f"  Age {age}: {count} images")

            print("\nGender Distribution:")
            for gender, count in stats['gender_distribution'].items():
                print(f"  {gender}: {count} images")

            print("\nSkin Color Distribution:")
            for color, count in stats['skin_color_distribution'].items():
                print(f"  {color}: {count} images")

        elif choice == '4':
            print("\nSearch Criteria (press Enter to skip):")
            age_input = input("Age: ").strip()
            gender_input = input("Gender: ").strip()
            skin_input = input("Skin Color: ").strip()

            age = int(age_input) if age_input else None
            gender = gender_input if gender_input else None
            skin_color = skin_input if skin_input else None

            results = explorer.search_by_criteria(age, gender, skin_color)

            if results:
                print(f"\nFound {len(results)} matching persons:")
                for person_id in results[:10]:
                    person = explorer.get_person(person_id)
                    print(f"  {person_id}: {person['age']}y {person['gender']} {person['skinColor']}, {person['num_images']} images")

                if len(results) > 10:
                    print(f"  ... and {len(results) - 10} more")
            else:
                print("No matching persons found.")

        elif choice == '5':
            #Z: Change this to 'q'
            print("\nGoodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
