import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    FEATURE_IMPORTANCE_FIG_PATH,
    FIGURES_DIR,
    PLAYERS_CLEANED_PATH,
    TOP_SCORERS_FIG_PATH,
    VALUE_PICKS_FIG_PATH,
    XG_VS_ACTUAL_FIG_PATH,
)
from src.logging_config import get_logger

logger = get_logger(__name__)


def _ensure_full_name(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["full_name"] = df["first_name"].fillna("") + " " + df["second_name"]
    return df


def plot_top_goalscorers(df_path: str = PLAYERS_CLEANED_PATH):
    """Bar chart of the top 10 goalscorers."""
    if not os.path.exists(df_path):
        logger.error("Processed data not found. Run `python -m src.clean` first.")
        return

    df = pd.read_csv(df_path)
    top_scorers = _ensure_full_name(df).sort_values("goals_scored", ascending=False).head(10)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        x="goals_scored", y="full_name", data=top_scorers,
        hue="full_name", palette="Blues_r", legend=False,
    )
    plt.title("Top 10 Premier League Goalscorers", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Goals Scored", fontsize=12, fontweight="bold")
    plt.ylabel("Player Name", fontsize=12, fontweight="bold")

    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(
                f"{int(width)}", (width, p.get_y() + p.get_height() / 2.0),
                ha="left", va="center", xytext=(6, 0), textcoords="offset points",
                fontsize=10, fontweight="bold",
            )

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(TOP_SCORERS_FIG_PATH, dpi=300)
    plt.close()
    logger.info("Saved -> %s", TOP_SCORERS_FIG_PATH)


def plot_xg_vs_actual(df_path: str = PLAYERS_CLEANED_PATH):
    """Scatter of expected vs actual goals, colored by position."""
    if not os.path.exists(df_path):
        logger.error("Processed data not found. Run `python -m src.clean` first.")
        return

    df = pd.read_csv(df_path)
    active = df[(df["goals_scored"] > 2) | (df["expected_goals"] > 2)].copy()
    if active.empty:
        logger.warning("No players cleared the activity threshold for xG plot; skipping.")
        return
    active = _ensure_full_name(active)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="expected_goals", y="goals_scored", data=active,
        hue="position", palette="deep", s=80, alpha=0.8,
    )
    max_val = max(active["expected_goals"].max(), active["goals_scored"].max())
    plt.plot([0, max_val], [0, max_val], color="gray", linestyle="--", label="Fair Value (xG = Goals)")

    plt.title("Expected Goals (xG) vs. Actual Goals Scored", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Expected Goals (xG)", fontsize=12, fontweight="bold")
    plt.ylabel("Actual Goals Scored", fontsize=12, fontweight="bold")
    plt.legend(title="Position", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(XG_VS_ACTUAL_FIG_PATH, dpi=300)
    plt.close()
    logger.info("Saved -> %s", XG_VS_ACTUAL_FIG_PATH)


def plot_feature_importance(importances_df: pd.DataFrame):
    """Bar chart of model feature importances (expects columns: feature, importance)."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))
    top = importances_df.head(15)
    sns.barplot(x="importance", y="feature", data=top, hue="feature", palette="mako", legend=False)
    plt.title("Model Feature Importances", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Importance", fontsize=12, fontweight="bold")
    plt.ylabel("")
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(FEATURE_IMPORTANCE_FIG_PATH, dpi=300)
    plt.close()
    logger.info("Saved -> %s", FEATURE_IMPORTANCE_FIG_PATH)


def plot_value_picks(df_path: str = PLAYERS_CLEANED_PATH, top_n: int = 15):
    """Top points-per-million players — the 'best value' chart FPL managers
    actually care about, which the original pipeline never surfaced.
    """
    if not os.path.exists(df_path):
        logger.error("Processed data not found. Run `python -m src.clean` first.")
        return

    df = pd.read_csv(df_path)
    df = df[df["minutes"] >= 450]  # roughly 5 full games, to avoid one-cap flukes
    top_value = _ensure_full_name(df).sort_values("points_per_million", ascending=False).head(top_n)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))
    sns.barplot(
        x="points_per_million", y="full_name", data=top_value,
        hue="position", dodge=False, palette="viridis",
    )
    plt.title(f"Top {top_n} Value Picks (Points per £1m)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Points per £Million", fontsize=12, fontweight="bold")
    plt.ylabel("")
    plt.legend(title="Position", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(VALUE_PICKS_FIG_PATH, dpi=300)
    plt.close()
    logger.info("Saved -> %s", VALUE_PICKS_FIG_PATH)


if __name__ == "__main__":
    plot_top_goalscorers()
    plot_xg_vs_actual()
    plot_value_picks()