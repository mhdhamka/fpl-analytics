import os
from src.config import BASE_DIR, FPL_API_URL, PLAYERS_CLEANED_PATH

def test_base_directory():
    """Ensure the base directory exists."""
    assert os.path.exists(BASE_DIR)

def test_api_url():
    """Ensure the FPL API URL is correctly formatted."""
    assert isinstance(FPL_API_URL, str)
    assert "fantasy.premierleague.com" in FPL_API_URL

def test_paths_defined():
    """Ensure core file paths are strings pointing inside the project."""
    assert isinstance(PLAYERS_CLEANED_PATH, str)
    assert "players_cleaned.csv" in PLAYERS_CLEANED_PATH