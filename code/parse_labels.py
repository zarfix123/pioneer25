#!/usr/bin/env python3
"""
Data Preprocessing Utilities for East Asian Musical Influence Classification

Handles label parsing, feature ordering, and dataset preparation.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

def parse_influence_labels(raw_labels):
    """Parse influence labels from various formats to clean 0/1 labels."""
    clean_labels = []
    
    for label in raw_labels:
        # Convert to string and extract first digit
        label_str = str(label).strip()
        match = re.search(r'^\s*([01])', label_str)
        
        if match:
            clean_labels.append(int(match.group(1)))
        else:
            # Handle edge cases
            label_lower = label_str.lower()
            if 'western' in label_lower or 'non' in label_lower or label_str == '0':
                clean_labels.append(0)
            elif 'influenced' in label_lower or 'asian' in label_lower or label_str == '1':
                clean_labels.append(1)
            else:
                clean_labels.append(np.nan)  # Mark as missing
    
    return pd.Series(clean_labels)

def ensure_feature_order(df, required_features=None):
    """Ensure features are in the correct order."""
    if required_features is None:
        required_features = [
            'pentatonicism', 'parallel_motion', 'density', 'rhythm_reg', 
            'syncopation', 'melodic_intervals', 'register_usage', 
            'articulation', 'dynamics'
        ]
    
    # Check if all required features exist
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Reorder columns
    other_columns = [col for col in df.columns if col not in required_features]
    ordered_columns = required_features + other_columns
    
    return df[ordered_columns]

def group_by_piece(df):
    """Group segments by musical piece."""
    if 'piece' not in df.columns:
        raise ValueError("Dataset must contain 'piece' column for grouping")
    
    piece_groups = df.groupby('piece')
    
    print("=== Piece Distribution ===")
    for piece, group in piece_groups:
        influence_counts = group['influence'].value_counts().sort_index()
        print(f"{piece}: {len(group)} segments", end="")
        if 0 in influence_counts.index and 1 in influence_counts.index:
            print(f" ({influence_counts[0]} non-inf, {influence_counts[1]} inf)")
        else:
            print(f" (class {influence_counts.index[0]})")
    
    return piece_groups

def validate_dataset(df):
    """Validate dataset for ML training."""
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values: {missing[missing > 0].to_dict()}")
    
    # Check label distribution
    if 'influence' in df.columns:
        label_counts = df['influence'].value_counts()
        if len(label_counts) != 2:
            issues.append(f"Invalid labels: should be 0/1, found {label_counts.index.tolist()}")
        
        balance_ratio = min(label_counts) / max(label_counts)
        if balance_ratio < 0.2:
            issues.append(f"Severe class imbalance: {balance_ratio:.3f}")
    
    # Check feature ranges
    feature_cols = ['pentatonicism', 'parallel_motion', 'density', 'rhythm_reg', 
                   'syncopation', 'melodic_intervals', 'register_usage', 
                   'articulation', 'dynamics']
    
    for feature in feature_cols:
        if feature in df.columns:
            values = df[feature].dropna()
            if not values.empty:
                min_val, max_val = values.min(), values.max()
                if min_val < 0 or max_val > 2:
                    issues.append(f"{feature} out of range [0,2]: [{min_val}, {max_val}]")
    
    return issues

def clean_dataset(western_path, influenced_path, output_path=None):
    """Clean and combine datasets."""
    print("=== Dataset Preprocessing ===")
    
    # Load datasets
    western_data = pd.read_csv(western_path)
    influenced_data = pd.read_csv(influenced_path)
    
    print(f"Western data: {len(western_data)} rows")
    print(f"Influenced data: {len(influenced_data)} rows")
    
    # Combine datasets
    combined_data = pd.concat([western_data, influenced_data], ignore_index=True)
    
    # Parse labels
    if 'influence' in combined_data.columns:
        combined_data['influence'] = parse_influence_labels(combined_data['influence'])
        
        # Remove rows with invalid labels
        before_count = len(combined_data)
        combined_data = combined_data.dropna(subset=['influence'])
        after_count = len(combined_data)
        
        if before_count != after_count:
            print(f"Removed {before_count - after_count} rows with invalid labels")
    
    # Ensure feature ordering
    try:
        combined_data = ensure_feature_order(combined_data)
        print("Features ordered correctly")
    except ValueError as e:
        print(f"Feature ordering issue: {e}")
    
    # Validate dataset
    issues = validate_dataset(combined_data)
    if issues:
        print("Dataset issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Dataset validation passed")
    
    # Group analysis
    try:
        piece_groups = group_by_piece(combined_data)
    except ValueError as e:
        print(f"Grouping issue: {e}")
    
    # Save cleaned dataset
    if output_path:
        combined_data.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved: {output_path}")
    
    print(f"Final dataset: {len(combined_data)} rows")
    print()
    
    return combined_data

def create_feature_summary(df):
    """Create summary of feature distributions."""
    feature_cols = ['pentatonicism', 'parallel_motion', 'density', 'rhythm_reg', 
                   'syncopation', 'melodic_intervals', 'register_usage', 
                   'articulation', 'dynamics']
    
    print("=== Feature Summary ===")
    
    if 'influence' in df.columns:
        non_influenced = df[df['influence'] == 0]
        influenced = df[df['influence'] == 1]
        
        for feature in feature_cols:
            if feature in df.columns:
                non_inf_mean = non_influenced[feature].mean()
                inf_mean = influenced[feature].mean()
                difference = inf_mean - non_inf_mean
                
                print(f"{feature}:")
                print(f"  Non-influenced: {non_inf_mean:.3f}")
                print(f"  Influenced: {inf_mean:.3f}")
                print(f"  Difference: {difference:+.3f}")
    else:
        for feature in feature_cols:
            if feature in df.columns:
                print(f"{feature}: {df[feature].mean():.3f} ± {df[feature].std():.3f}")

def main():
    """Main preprocessing pipeline."""
    # Paths (adjust as needed)
    western_path = '/home/dennis/Projects/research/data/western data.csv'
    influenced_path = '/home/dennis/Projects/research/data/influenced data.csv'
    output_path = '/home/dennis/Projects/research/data/cleaned_dataset.csv'
    
    try:
        # Clean dataset
        cleaned_data = clean_dataset(western_path, influenced_path, output_path)
        
        # Create feature summary
        create_feature_summary(cleaned_data)
        
        print("Preprocessing complete!")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check data file paths")
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    main()