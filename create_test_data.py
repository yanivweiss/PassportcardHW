import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Force stdout to flush after each print for better logging
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def create_test_claims_data(n_claims=1000, n_members=100, output_file='claims_data_clean.csv'):
    """Create synthetic claims data for testing"""
    print_flush(f"Generating {n_claims} test claims for {n_members} members...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate member IDs
    member_ids = np.random.randint(1, n_members + 1, size=n_claims)
    
    # Generate policy IDs (each member belongs to one policy)
    policy_map = {i: 100 + i//2 for i in range(1, n_members + 1)}
    policy_ids = np.array([policy_map[m_id] for m_id in member_ids])
    
    # Generate service dates over the past 3 years
    base_date = datetime(2020, 1, 1)
    days_range = np.random.randint(0, 365*3, size=n_claims).tolist()  # Convert to Python ints
    service_dates = [base_date + timedelta(days=d) for d in days_range]
    print_flush(f"Generated {len(service_dates)} service dates")
    
    # Generate payment dates (a few days after service)
    pay_days = np.random.randint(5, 30, size=n_claims).tolist()  # Convert to Python ints
    pay_dates = [sd + timedelta(days=d) for sd, d in zip(service_dates, pay_days)]
    
    # Generate service groups and types
    service_groups = np.random.choice(['Outpatient', 'Inpatient', 'Emergency', 'Preventive', 'Dental'], size=n_claims)
    service_types = np.random.choice(['Consultation', 'Surgery', 'Lab', 'Imaging', 'Medication', 'Therapy'], size=n_claims)
    
    # Generate payment amounts based on service type
    base_amounts = {
        'Consultation': 100,
        'Surgery': 5000,
        'Lab': 200,
        'Imaging': 800,
        'Medication': 150,
        'Therapy': 300
    }
    
    # Add some randomness to payment amounts
    payments = np.array([base_amounts[st] * (0.5 + np.random.random()) for st in service_types])
    
    # Create claim numbers
    claim_numbers = np.arange(1, n_claims + 1)
    
    # Generate gender
    genders = np.random.choice(['M', 'F'], size=n_claims)
    
    # Create DataFrame
    print_flush("Creating claims DataFrame...")
    claims_df = pd.DataFrame({
        'ClaimNumber': claim_numbers,
        'TotPaymentUSD': payments,
        'ServiceDate': service_dates,
        'PayDate': pay_dates,
        'ServiceGroup': service_groups,
        'ServiceType': service_types,
        'Member_ID': member_ids,
        'PolicyID': policy_ids,
        'Sex': genders
    })
    
    # Save to CSV
    print_flush(f"Saving claims data to {output_file}...")
    claims_df.to_csv(output_file, index=False)
    print_flush(f"Created claims data with {len(claims_df)} rows, saved to {output_file}")
    
    return claims_df

def create_test_members_data(n_members=100, output_file='members_data_clean.csv'):
    """Create synthetic member data for testing"""
    print_flush(f"Generating {n_members} test members...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate member IDs
    member_ids = np.arange(1, n_members + 1)
    
    # Generate policy IDs (each member belongs to one policy)
    policy_map = {i: 100 + i//2 for i in range(1, n_members + 1)}
    policy_ids = np.array([policy_map[m_id] for m_id in member_ids])
    
    # Generate dates
    base_date = datetime(2020, 1, 1)
    
    # Policy start dates (1-2 years ago)
    policy_start_days = np.random.randint(-365*2, -365, size=n_members).tolist()  # Convert to Python ints
    policy_start_dates = [base_date + timedelta(days=d) for d in policy_start_days]
    
    # Policy end dates (0-2 years in future)
    policy_end_days = np.random.randint(365, 365*3, size=n_members).tolist()  # Convert to Python ints
    policy_end_dates = [base_date + timedelta(days=d) for d in policy_end_days]
    
    # Birth dates (20-80 years ago)
    birth_days = np.random.randint(-365*80, -365*20, size=n_members).tolist()  # Convert to Python ints
    birth_dates = [base_date + timedelta(days=d) for d in birth_days]
    
    # Generate demographics
    countries_origin = np.random.choice(['USA', 'UK', 'Canada', 'Germany', 'France', 'Australia', 'Japan'], size=n_members)
    countries_dest = np.random.choice(['USA', 'UK', 'Canada', 'Germany', 'France', 'Australia', 'Japan'], size=n_members)
    genders = np.random.choice([True, False], size=n_members)  # True is male
    
    # Generate BMI
    bmis = np.random.normal(25, 4, size=n_members)
    
    # Generate questionnaire responses (binary)
    questionnaire_fields = [
        'cancer', 'smoke', 'heart', 'diabetes', 'respiratory', 
        'thyroid', 'liver', 'immune', 'tumor', 'relatives',
        'alcoholism', 'drink'
    ]
    
    # Create DataFrame
    print_flush("Creating members DataFrame...")
    members_df = pd.DataFrame({
        'Member_ID': member_ids,
        'PolicyID': policy_ids,
        'PolicyStartDate': policy_start_dates,
        'PolicyEndDate': policy_end_dates,
        'DateOfBirth': birth_dates,
        'CountryOfOrigin': countries_origin,
        'CountryOfDestination': countries_dest,
        'Gender': genders,
        'BMI': bmis
    })
    
    # Add questionnaire fields (with low probability of positive response)
    for field in questionnaire_fields:
        prob_yes = 0.05 if field in ['cancer', 'tumor', 'heart'] else 0.15
        members_df[f'Questionnaire_{field}'] = np.random.choice([0, 1], size=n_members, p=[1-prob_yes, prob_yes])
    
    # Add some synthetic risk metrics
    members_df['uw_pct'] = np.random.uniform(0.5, 2.0, size=n_members)
    members_df['average_us'] = np.random.uniform(0.1, 0.9, size=n_members)
    members_df['average_ov'] = np.random.uniform(0.1, 0.9, size=n_members)
    members_df['average_ob'] = np.random.uniform(0.1, 0.9, size=n_members)
    
    # Save to CSV
    print_flush(f"Saving members data to {output_file}...")
    members_df.to_csv(output_file, index=False)
    print_flush(f"Created members data with {len(members_df)} rows, saved to {output_file}")
    
    return members_df

if __name__ == "__main__":
    print_flush("Generating test data for PassportCard Claims Analysis...")
    claims_df = create_test_claims_data(n_claims=1000, n_members=100)
    members_df = create_test_members_data(n_members=100)
    print_flush("Test data generation complete!") 