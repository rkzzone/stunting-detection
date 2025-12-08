"""
Script untuk generate sample data jika data_balita.csv tidak tersedia
Hanya untuk testing/demo purposes
"""

import pandas as pd
import numpy as np
from datetime import datetime

def generate_sample_data(n_samples=1000):
    """
    Generate sample data untuk testing
    Berdasarkan distribusi WHO growth standards
    """
    
    np.random.seed(42)
    
    print("🔄 Generating sample data...")
    
    data = []
    
    for i in range(n_samples):
        # Random age (0-60 bulan)
        age = np.random.randint(0, 61)
        
        # Random gender
        gender = np.random.choice(['Laki-laki', 'Perempuan'])
        
        # WHO reference heights (approximate median)
        if gender == 'Laki-laki':
            base_heights = {
                0: 50, 6: 67, 12: 76, 18: 82, 24: 88,
                30: 92, 36: 96, 42: 100, 48: 103, 54: 107, 60: 110
            }
        else:
            base_heights = {
                0: 49, 6: 66, 12: 74, 18: 81, 24: 86,
                30: 91, 36: 96, 42: 100, 48: 103, 54: 107, 60: 110
            }
        
        # Interpolate base height
        ages_keys = sorted(base_heights.keys())
        if age in base_heights:
            base_height = base_heights[age]
        else:
            # Linear interpolation
            lower_age = max([a for a in ages_keys if a <= age])
            upper_age = min([a for a in ages_keys if a >= age])
            if lower_age == upper_age:
                base_height = base_heights[lower_age]
            else:
                ratio = (age - lower_age) / (upper_age - lower_age)
                base_height = base_heights[lower_age] + ratio * (base_heights[upper_age] - base_heights[lower_age])
        
        # Add variation based on nutritional status
        status_prob = np.random.random()
        
        if status_prob < 0.05:  # 5% severely stunted
            z_score = np.random.uniform(-4.5, -3)
            status = 'severely stunted'
        elif status_prob < 0.20:  # 15% stunted
            z_score = np.random.uniform(-3, -2)
            status = 'stunted'
        elif status_prob < 0.95:  # 75% normal
            z_score = np.random.uniform(-2, 3)
            status = 'normal'
        else:  # 5% tall
            z_score = np.random.uniform(3, 4)
            status = 'tall'
        
        # Calculate height based on z-score
        # Approximate: height = base_height + (z_score * SD)
        # SD approximately 3-5% of base_height
        sd = base_height * 0.04
        height = base_height + (z_score * sd)
        height = round(height, 1)
        
        # Ensure reasonable bounds
        height = max(40, min(150, height))
        
        data.append({
            'Umur (bulan)': age,
            'Jenis Kelamin': gender,
            'Tinggi Badan (cm)': height,
            'Status Gizi': status
        })
    
    df = pd.DataFrame(data)
    
    return df

def main():
    """Main function to generate and save sample data"""
    
    print("=" * 80)
    print("📊 SAMPLE DATA GENERATOR FOR STUNTING DETECTION")
    print("=" * 80)
    print("\n⚠️ WARNING: This is sample data for TESTING/DEMO only!")
    print("For real deployment, use actual clinical data.\n")
    
    # Generate data
    n_samples = 2000
    df = generate_sample_data(n_samples)
    
    # Show statistics
    print(f"\n✅ Generated {len(df)} samples")
    print(f"\n📊 Status Distribution:")
    print(df['Status Gizi'].value_counts())
    print("\nPercentage:")
    print(df['Status Gizi'].value_counts(normalize=True) * 100)
    
    print(f"\n👥 Gender Distribution:")
    print(df['Jenis Kelamin'].value_counts())
    
    print(f"\n📏 Height Statistics:")
    print(df['Tinggi Badan (cm)'].describe())
    
    print(f"\n⏰ Age Statistics:")
    print(df['Umur (bulan)'].describe())
    
    # Save to CSV
    output_path = r"C:\Users\muham\Project\Stunting\data_balita.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n💾 Data saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("✅ SAMPLE DATA GENERATION COMPLETED")
    print("=" * 80)
    
    print("\n📝 Next steps:")
    print("1. Verify the data: pandas.read_csv('data_balita.csv')")
    print("2. Train the model: python knn_model_trainer.py")
    print("3. Run the app: streamlit run app.py")
    
    return df

if __name__ == "__main__":
    df = main()
    
    # Display sample
    print("\n🔍 Sample data (first 10 rows):")
    print(df.head(10))
    
    print("\n🔍 Sample data (random 5 rows per status):")
    for status in df['Status Gizi'].unique():
        print(f"\n{status.upper()}:")
        print(df[df['Status Gizi'] == status].sample(min(5, len(df[df['Status Gizi'] == status]))))