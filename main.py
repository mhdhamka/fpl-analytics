from src.ingest import save_raw_data
from src.clean import clean_data
from src.visualize import plot_top_goalscorers, plot_xg_vs_actual
from src.model import train_points_model

def run_pipeline():
    """Runs the complete Premier League analytics and ML pipeline end-to-end."""
    print("=== Step 1: Ingesting Data from FPL API ===")
    save_raw_data()
    
    print("\n=== Step 2: Cleaning and Processing Data ===")
    clean_data()
    
    print("\n=== Step 3: Generating Visualizations ===")
    plot_top_goalscorers()
    plot_xg_vs_actual()
    
    print("\n=== Step 4: Training Machine Learning Model ===")
    train_points_model()
    
    print("\n Pipeline execution completed successfully!")

if __name__ == "__main__":
    run_pipeline()