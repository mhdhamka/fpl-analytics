import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import (
    PLAYERS_CLEANED_PATH, 
    TOP_SCORERS_FIG_PATH, 
    XG_VS_ACTUAL_FIG_PATH, 
    FIGURES_DIR
)

def plot_top_goalscorers(df_path=PLAYERS_CLEANED_PATH):
    """Generates a styled bar chart of the top 10 goalscorers and saves it."""
    if not os.path.exists(df_path):
        print("Processed data not found! Please run src/clean.py first.")
        return

    df = pd.read_csv(df_path)
    
    # Sort by goals scored and get top 10
    top_scorers = df.sort_values(by="goals_scored", ascending=False).head(10)
    
    # Create full player name column for better chart labels
    top_scorers["full_name"] = top_scorers["first_name"].fillna('') + " " + top_scorers["second_name"]

    # Setup Seaborn theme & figure size
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 6))
    
    # Create horizontal bar plot
    ax = sns.barplot(
        x="goals_scored", 
        y="full_name", 
        data=top_scorers, 
        hue="full_name",
        palette="Blues_r",
        legend=False
    )
    
    # Styling and Labels
    plt.title("Top 10 Premier League Goalscorers (26/27 Season)", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Goals Scored", fontsize=12, fontweight="bold")
    plt.ylabel("Player Name", fontsize=12, fontweight="bold")
    
    # Annotate numeric values on the bars
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(f"{int(width)}",
                        (width, p.get_y() + p.get_height() / 2.),
                        ha="left", va="center",
                        xytext=(6, 0),
                        textcoords="offset points",
                        fontsize=10, fontweight="bold")

    plt.tight_layout()
    
    # Save figure to outputs folder using config path
    os.makedirs(FIGURES_DIR, exist_ok=True)
    output_file = TOP_SCORERS_FIG_PATH
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Visualization successfully saved to {output_file}!")

def plot_xg_vs_actual(df_path=PLAYERS_CLEANED_PATH):
    """Generates a comparative scatter/regression plot of Expected Goals (xG) vs Actual Goals."""
    if not os.path.exists(df_path):
        print("Processed data not found! Please run src/clean.py first.")
        return

    df = pd.read_csv(df_path)
    
    # Filter for players who have taken shots/scored to keep the chart meaningful
    active_players = df[(df["goals_scored"] > 2) | (df["expected_goals"] > 2)].copy()
    active_players["full_name"] = active_players["first_name"].fillna('') + " " + active_players["second_name"]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot comparing xG to Goals Scored
    ax = sns.scatterplot(
        x="expected_goals",
        y="goals_scored",
        data=active_players,
        hue="position",
        palette="deep",
        s=80,
        alpha=0.8
    )
    
    # Add a reference line for perfect performance (xG = Actual Goals)
    max_val = max(active_players["expected_goals"].max(), active_players["goals_scored"].max())
    plt.plot([0, max_val], [0, max_val], color="gray", linestyle="--", label="Fair Value (xG = Goals)")

    # Styling and Labels
    plt.title("Expected Goals (xG) vs. Actual Goals Scored (26/27 Season)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Expected Goals (xG)", fontsize=12, fontweight="bold")
    plt.ylabel("Actual Goals Scored", fontsize=12, fontweight="bold")
    plt.legend(title="Position", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    
    # Save figure to outputs folder using config path
    os.makedirs(FIGURES_DIR, exist_ok=True)
    output_file = XG_VS_ACTUAL_FIG_PATH
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"xG Visualization successfully saved to {output_file}!")

if __name__ == "__main__":
    plot_top_goalscorers()
    plot_xg_vs_actual()