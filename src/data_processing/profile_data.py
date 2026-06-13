import os
import pandas as pd
from load_data import load_dataset

def profile_dataset(input_path: str, report_output_path: str) -> None:
    """
    Profiles the dataset and generates a markdown report.
    """
    df = load_dataset(input_path)
    
    total_papers = len(df)
    missing_values = df.isnull().sum().to_dict()
    duplicate_count = df.duplicated(subset=['paper_id']).sum()
    
    # Category distribution (top 15)
    category_counts = df['category'].value_counts()
    category_dist_str = "\n".join([f"| {cat} | {count} |" for cat, count in category_counts.items()])
    
    # Year distribution
    year_counts = df['year'].value_counts().sort_index()
    year_dist_list = []
    for year, count in year_counts.items():
        if year is not None and year == year:
            try:
                year_dist_list.append(f"| {int(float(str(year)))} | {count} |")
            except (ValueError, TypeError):
                pass
    year_dist_str = "\n".join(year_dist_list)
    
    # Average abstract length
    avg_abstract_len = df['abstract'].fillna('').apply(len).mean()
    
    # Generate markdown content
    markdown_content = f"""# Data Profile Report

## Summary Statistics
* **Total Papers**: {total_papers}
* **Duplicate Papers (by paper_id)**: {duplicate_count}
* **Average Abstract Length (characters)**: {avg_abstract_len:.2f}

## Missing Values
| Field | Missing Count |
| --- | --- |
"""
    for field, count in missing_values.items():
        markdown_content += f"| {field} | {count} |\n"
        
    markdown_content += f"""
## Category Distribution
| Category | Paper Count |
| --- | --- |
{category_dist_str}

## Year Distribution
| Year | Paper Count |
| --- | --- |
{year_dist_str}
"""
    
    # Write report
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
        
    print(f"Data profile report saved to {report_output_path}")

if __name__ == "__main__":
    raw_path = "data/raw/arxiv_raw.csv"
    report_path = "data/reports/data_profile_report.md"
    if os.path.exists(raw_path):
        profile_dataset(raw_path, report_path)
    else:
        print(f"Raw dataset not found at {raw_path}. Cannot profile.")
