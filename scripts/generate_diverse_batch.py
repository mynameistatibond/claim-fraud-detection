
import pandas as pd
import numpy as np
import random
from pathlib import Path

def generate_diverse_data(n=1000):
    # Options derived from preprocessing_metadata.json and experience
    collision_types = ['Side Collision', 'Rear Collision', 'Front Collision', '?']
    severities = ['Minor Damage', 'Total Loss', 'Major Damage', 'Trivial Damage']
    authorities = ['Police', 'Fire', 'Ambulance', 'None', 'Other']
    states = ['NY', 'SC', 'WV', 'NC', 'VA', 'PA', 'OH']
    makes = ['Subaru', 'Dodge', 'Saab', 'Nissan', 'Chevrolet', 'Ford', 'BMW', 'Toyota', 'Audi', 'Volkswagen', 'Mercedes', 'Honda', 'Jeep', 'Accura']
    
    data = {
        "Claim ID": [f"TEST_{i:04d}" for i in range(n)],
        "months_as_customer": np.random.randint(0, 500, n),
        "age": np.random.randint(18, 90, n),
        "policy_state": [random.choice(states) for _ in range(n)],
        "policy_csl": [random.choice(["100/300", "250/500", "500/1000"]) for _ in range(n)],
        "policy_deductable": [random.choice([500, 1000, 2000]) for _ in range(n)],
        "policy_annual_premium": np.random.uniform(400, 2500, n).round(2),
        "umbrella_limit": [random.choice([0, 2000000, 3000000, 4000000, 5000000, 6000000]) for _ in range(n)],
        "insured_sex": [random.choice(["MALE", "FEMALE"]) for _ in range(n)],
        "insured_education_level": [random.choice(["High School", "College", "Masters", "JD", "MD", "PhD", "Associate"]) for _ in range(n)],
        "insured_occupation": [random.choice(["sales", "tech-support", "handlers-cleaners", "prof-specialty", "exec-managerial", "craft-repair", "transport-moving"]) for _ in range(n)],
        "insured_hobbies": ["reading" for _ in range(n)], # Default/Ignored
        "insured_relationship": [random.choice(["husband", "wife", "own-child", "unmarried", "other-relative", "not-in-family"]) for _ in range(n)],
        "capital-gains": np.random.choice([0, 5000, 25000, 50000, 80000], n),
        "capital-loss": np.random.choice([0, -20000, -40000, -60000], n),
        "incident_type": [random.choice(["Multi-vehicle Collision", "Single Vehicle Collision", "Vehicle Theft", "Parked Car"]) for _ in range(n)],
        "collision_type": [random.choice(collision_types) for _ in range(n)],
        "incident_severity": [random.choice(severities) for _ in range(n)],
        "authorities_contacted": [random.choice(authorities) for _ in range(n)],
        "incident_state": [random.choice(states) for _ in range(n)],
        "incident_city": ["City" for _ in range(n)],
        "incident_hour_of_the_day": np.random.randint(0, 23, n),
        "number_of_vehicles_involved": np.random.randint(1, 4, n),
        "property_damage": [random.choice(["YES", "NO", "?"]) for _ in range(n)],
        "bodily_injuries": np.random.randint(0, 3, n),
        "witnesses": np.random.randint(0, 4, n),
        "police_report_available": [random.choice(["YES", "NO", "?"]) for _ in range(n)],
        "total_claim_amount": np.random.uniform(2000, 100000, n).round(2),
        "injury_claim": np.random.uniform(0, 20000, n).round(0),
        "property_claim": np.random.uniform(0, 20000, n).round(0),
        "vehicle_claim": np.random.uniform(0, 60000, n).round(0),
        "auto_make": [random.choice(makes) for _ in range(n)],
        "auto_model": ["Model" for _ in range(n)],
        "auto_year": np.random.randint(1995, 2024, n),
        "fraud_reported": ["?" for _ in range(n)]
    }
    
    # Enforce some correlations for realism/diversity of scores
    # E.g. Major Damage + High Claim should be riskier in some models, but let's keep it random for maximum coverage.
    
    df = pd.DataFrame(data)
    
    # Create derived columns expected by batch_ingest mapping friendly to internal if needed
    # But we are using the "internal" names (mostly) or standard CSV names.
    # The `batch_ingest` Friendly Map expects "Claim Value", etc if using friendly.
    # Let's verify batch_ingest FRIENDLY_TO_INTERNAL. 
    # Actually, batch_ingest checks keys. 
    # Let's assume standard snake_case keys work (except "Claim ID").
    
    return df

if __name__ == "__main__":
    df = generate_diverse_data(1000)
    out_path = Path(__file__).resolve().parent.parent / "csv examples/test_batch_1000_diverse.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {out_path} with 1000 unique rows.")
