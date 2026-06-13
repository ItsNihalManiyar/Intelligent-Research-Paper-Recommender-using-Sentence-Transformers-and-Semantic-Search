import os
import pandas as pd
from load_data import load_dataset

def validate_dataset(input_path: str, report_output_path: str) -> None:
    """
    Validates schema and content of the cleaned dataset.
    """
    df = load_dataset(input_path)
    
    validation_passed = True
    issues = []
    
    # 1. Schema Validation
    required_fields = ['paper_id', 'title', 'abstract', 'authors', 'year', 'category']
    missing_fields = [field for field in required_fields if field not in df.columns]
    
    if missing_fields:
        validation_passed = False
        issues.append(f"Missing required fields: {', '.join(missing_fields)}")
    else:
        issues.append("Schema check passed: all required fields are present.")
        
    # 2. Missing values validation (especially title and abstract)
    null_titles = df['title'].isnull().sum()
    null_abstracts = df['abstract'].isnull().sum()
    
    if null_titles > 0 or null_abstracts > 0:
        validation_passed = False
        issues.append(f"Data contains null values: {null_titles} missing titles, {null_abstracts} missing abstracts.")
    else:
        issues.append("Missing values check passed: no missing titles or abstracts.")
        
    # 3. Year validity check
    invalid_years = 0
    if 'year' in df.columns:
        # Check if year is numeric and within a reasonable range (e.g. 1900 to 2026)
        valid_year_mask = df['year'].astype(float).between(1900, 2026)
        invalid_years = len(df) - valid_year_mask.sum()
        if invalid_years > 0:
            validation_passed = False
            issues.append(f"Contains {invalid_years} invalid/unreasonable years.")
        else:
            issues.append("Year validity check passed: all years are within the valid range (1900-2026).")
            
    # 4. Category validity check
    invalid_categories = 0
    if 'category' in df.columns:
        invalid_categories = df['category'].isnull().sum() + (df['category'] == '').sum() + (df['category'] == 'unknown').sum()
        if invalid_categories > 0:
            validation_passed = False
            issues.append(f"Contains {invalid_categories} missing/unknown/empty categories.")
        else:
            issues.append("Category validity check passed: all categories are populated.")
            
    # Generate markdown validation report
    markdown_content = f"""# Data Validation Report

## Overall Status: {"PASSED" if validation_passed else "FAILED"}

## Validation Details
"""
    for issue in issues:
        if "passed" in issue.lower():
            markdown_content += f"* **[PASS]** {issue}\n"
        else:
            markdown_content += f"* **[FAIL]** {issue}\n"
            
    markdown_content += f"""
## Dataset Dimensions
* **Total Rows**: {len(df)}
* **Total Columns**: {len(df.columns)}
"""
    
    # Save validation report
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
        
    print(f"Data validation report saved to {report_output_path}")

if __name__ == "__main__":
    processed_path = "data/processed/cleaned_papers.csv"
    report_path = "data/reports/data_validation_report.md"
    if os.path.exists(processed_path):
        validate_dataset(processed_path, report_path)
    else:
        print(f"Cleaned dataset not found at {processed_path}. Cannot validate.")
