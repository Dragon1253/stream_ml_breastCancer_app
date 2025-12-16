import pandas as pd
import random
from datetime import datetime, timedelta

# Constants for attributes
MENOPAUS_VALUES = [0, 1, 9]
AGEGRP_VALUES = list(range(1, 11))
DENSITY_VALUES = [1, 2, 3, 4, 9]
RACE_VALUES = [1, 2, 3, 4, 5, 9]
HISPANIC_VALUES = [0, 1, 9]
BMI_VALUES = [1, 2, 3, 4, 9]
AGEFIRST_VALUES = [0, 1, 2, 9]
NRELBC_VALUES = [0, 1, 2, 9]
BRSTPROC_VALUES = [0, 1, 9]
LASTMAMM_VALUES = [0, 1, 9]
SURGMENO_VALUES = [0, 1, 9]
HRT_VALUES = [0, 1, 9]

# Image files found in patient_images directory
IMAGE_FILES = [
    "about.png", "ahmed.png", "Ali.png", 
    "comment-author-01.jpg", "testi1.jpg", 
    "testi2.jpg", "testi3.jpg"
]

# Dummy names
FIRST_NAMES = [
    "Marie", "Sophie", "Julie", "Camille", "Lea", "Manon", "Chloe", "Laura", 
    "Sarah", "Emma", "Alice", "Thomas", "Nicolas", "Julien", "Pierre", 
    "Lucas", "Maxime", "Antoine", "Alexandre", "Paul"
]
LAST_NAMES = [
    "Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", 
    "Durand", "Leroy", "Moreau", "Simon", "Laurent", "Lefebvre", "Michel", 
    "Garcia", "David", "Bertrand", "Roux", "Vincent", "Fournier"
]

def generate_random_date(start_year=2023, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randrange(delta.days)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")

def generate_data(num_records=100):
    data = []
    for _ in range(num_records):
        record = {
            "first_name": random.choice(FIRST_NAMES),
            "last_name": random.choice(LAST_NAMES),
            "exam_date": generate_random_date(),
            "patient_image": random.choice(IMAGE_FILES),
            "menopaus": random.choice(MENOPAUS_VALUES),
            "agegrp": random.choice(AGEGRP_VALUES),
            "density": random.choice(DENSITY_VALUES),
            "race": random.choice(RACE_VALUES),
            "Hispanic": random.choice(HISPANIC_VALUES),
            "bmi": random.choice(BMI_VALUES),
            "agefirst": random.choice(AGEFIRST_VALUES),
            "nrelbc": random.choice(NRELBC_VALUES),
            "brstproc": random.choice(BRSTPROC_VALUES),
            "lastmamm": random.choice(LASTMAMM_VALUES),
            "surgmeno": random.choice(SURGMENO_VALUES),
            "hrt": random.choice(HRT_VALUES)
        }
        data.append(record)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_data(100)
    output_file = "patients_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} records in {output_file}")
