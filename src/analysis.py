import pandas as pd

def load_processed_data():
    """Load the processed data."""
    data_path = 'data/processed/student_processed.csv'
    
    df = pd.read_csv(data_path)
    return df

def print_descriptive_statistics(df):
    """Print comprehensive descriptive statistics."""
    key_features = ['age', 'Medu', 'Fedu', 'studytime', 'failures', 
                    'absences', 'G3', 'avg_alcohol', 'passed']
    
    print("\nKey Features Summary:")
    print(df[key_features].describe())
    
    print("\nPass/Fail Distribution:")
    passed_count = df['passed'].sum()
    failed_count = len(df) - passed_count
    pass_rate = (passed_count / len(df)) * 100
    
    print(f"Passed (G3 >= 10): {passed_count} students ({pass_rate:.1f}%)")
    print(f"Failed (G3 < 10):  {failed_count} students ({100-pass_rate:.1f}%)")
    
    print("\nGrade Statistics:")
    print(f"Mean Grade (G3):   {df['G3'].mean():.2f}")
    print(f"Median Grade (G3): {df['G3'].median():.2f}")
    print(f"Std Dev Grade:     {df['G3'].std():.2f}")
    print(f"Min Grade:         {df['G3'].min():.2f}")
    print(f"Max Grade:         {df['G3'].max():.2f}")

def analyze_correlations(df):
    """Analyze and print key correlations with target variable."""
    print("\nCorrelation Analysis:")
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    correlations = df[numeric_cols].corr()['G3'].sort_values(ascending=False)
    
    print("\nTop 10 Features Positively Correlated with Final Grade (G3):")
    count = 0
    for feature, corr in correlations.items():
        if feature != 'G3' and count < 10:
            print(f"{count+1:2d}. {feature:35s} {corr:6.3f}")
            count += 1
    
    print("\nTop 10 Features Negatively Correlated with Final Grade (G3):")
    negative_corrs = correlations.tail(10)
    for i, (feature, corr) in enumerate(negative_corrs.items(), 1):
        print(f"{i:2d}. {feature:35s} {corr:6.3f}")

def generate_insights(df):
    """Generate and print key insights from the data."""
    print("\nKey Insights:")
    
    # Study time impact
    high_study = df[df['studytime'] >= 3]['G3'].mean()
    low_study = df[df['studytime'] <= 2]['G3'].mean()
    print(f"\n1. Study Time Impact:")
    print(f"   Students who study more (studytime >= 3) have a mean grade of {high_study:.2f}")
    print(f"   Students who study less (studytime <= 2) have a mean grade of {low_study:.2f}")
    print(f"   The difference in mean grades is {high_study - low_study:.2f} points")
    
    # Mother's education impact
    high_medu = df[df['Medu'] >= 3]['G3'].mean()
    low_medu = df[df['Medu'] < 3]['G3'].mean()
    print(f"\n2. Mother's Education Impact:")
    print(f"   Students with high mother education (Medu >= 3) have a mean grade of {high_medu:.2f}")
    print(f"   Students with low mother education (Medu < 3) have a mean grade of {low_medu:.2f}")
    print(f"   The difference in mean grades is {high_medu - low_medu:.2f} points")
    
    # Previous failures impact
    no_failures = df[df['failures'] == 0]['G3'].mean()
    has_failures = df[df['failures'] > 0]['G3'].mean()
    print(f"\n3. Previous Failures Impact:")
    print(f"   Students with no previous failures have a mean grade of {no_failures:.2f}")
    print(f"   Students with previous failures have a mean grade of {has_failures:.2f}")
    print(f"   The difference in mean grades is {no_failures - has_failures:.2f} points")
    
    # Absences impact
    low_absences = df[df['absences'] <= df['absences'].median()]['G3'].mean()
    high_absences = df[df['absences'] > df['absences'].median()]['G3'].mean()
    print(f"\n4. Absences Impact:")
    print(f"   Students with low absences (≤ median) have a mean grade of {low_absences:.2f}")
    print(f"   Students with high absences (> median) have a mean grade of {high_absences:.2f}")
    print(f"   The difference in mean grades is {low_absences - high_absences:.2f} points")
    
    # Alcohol consumption impact
    low_alcohol = df[df['avg_alcohol'] <= 1.5]['G3'].mean()
    high_alcohol = df[df['avg_alcohol'] > 1.5]['G3'].mean()
    print(f"\n5. Alcohol Consumption Impact:")
    print(f"   Students with low alcohol consumption (≤ 1.5) have a mean grade of {low_alcohol:.2f}")
    print(f"   Students with high alcohol consumption (> 1.5) have a mean grade of {high_alcohol:.2f}")
    print(f"   The difference in mean grades is {low_alcohol - high_alcohol:.2f} points")

def analyze_distributions(df):
    """Analyze distributions of key features."""
    print("\nDistribution Analysis:")
    
    # Age distribution
    print("\nAge Distribution:")
    print(df['age'].value_counts().sort_index())
    
    # Study time distribution
    print("\nStudy Time Distribution:")
    print(df['studytime'].value_counts().sort_index())
    
    # Failures distribution
    print("\nPrevious Failures Distribution:")
    print(df['failures'].value_counts().sort_index())

def main():
    df = load_processed_data()
    
    print_descriptive_statistics(df)
    analyze_correlations(df)
    generate_insights(df)
    analyze_distributions(df)


if __name__ == "__main__":
    main()
