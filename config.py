"""
config.py — single source of truth for paths, env vars, and API settings.
No other module may hardcode a path or a secret. See DESIGN.md section 4.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root, if present

BASE_DIR = Path(os.getenv("THESIS_PROJECT_DIR", Path.home() / "thesis_project"))

RAW_DIR = BASE_DIR / "data" / "raw"
CORDIS_HORIZON_DIR = RAW_DIR / "cordis_horizon"
COMPLEXITY_RAW_DIR = RAW_DIR / "complexity"     # PENDING source, see docs/DATA_MANIFEST.md
MAPPING_RAW_DIR = RAW_DIR / "mapping"           # PENDING source, see docs/DATA_MANIFEST.md
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

CORDIS_CANDIDATE_POOL_CSV = PROCESSED_DIR / "projects_clean.csv"

PILOT_SAMPLE_CSV = PROCESSED_DIR / "pilot_sample_20.csv"

CORDIS_POLICY_FILE = CORDIS_HORIZON_DIR / "policyPriorities.csv"  # TODO: confirm actual filename

COMPLEXITY_INPUT = COMPLEXITY_RAW_DIR / "trade.csv"   # actual filename as downloaded
MAPPING_CSV = MAPPING_RAW_DIR / "mappingHs22toNACE.csv"

BENCHMARK_CASES_CSV = PROCESSED_DIR / "benchmark_cases.csv"
COMPLEXITY_RESULTS_CSV = PROCESSED_DIR / "complexity_results.csv"
PROXIMITY_RESULTS_CSV = PROCESSED_DIR / "proximity_results.csv"
THRESHOLD_CALIBRATION_JSON = RESULTS_DIR / "threshold_calibration.json"

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "")

API_KEY_ENV = "OPEN_AI_API_KEY"
OPENAI_API_KEY = os.getenv(API_KEY_ENV, "")

PROMPT_VARIANTS = ["content", "ec_feasible", "capability", "policy"]
CONSENSUS_N_RUNS = 3
CONSENSUS_MIN_VOTES = 2
TOP_N_DIVERSIFICATION_PATHS = 10
MAX_FUNDING_RESULTS = 10

LOGSUPERMODULARITY_SAMPLES = 85000


def ensure_dirs() -> None:
    """Create PROCESSED_DIR and RESULTS_DIR if missing (the two
    disposable/regeneratable folders per PROJECT_STRUCTURE.md). Does NOT
    create anything under data/raw/ — that's scripts/setup_data_folder.py's
    job, run once, deliberately separate from the pipeline's own writes.
    Call once at program start; nothing else should call Path.mkdir()."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)