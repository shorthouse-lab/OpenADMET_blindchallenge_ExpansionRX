# env.py
import time, torch, os, logging, warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning, UndefinedMetricWarning

# Environment
DEFAULT_FP_SIZE = 1024
_FORCE_CPU = os.getenv('CHEMGRID_FORCE_CPU', '0') == '1'
if _FORCE_CPU:
	DEVICE = 'cpu'
else:
	DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"[env] DEVICE={DEVICE} (force_cpu={_FORCE_CPU})")

# Study parameters
N_TESTS = int(os.getenv('N_TESTS', '10'))
N_FOLDS = int(os.getenv('N_FOLDS', '5'))
N_TRIALS = int(os.getenv('N_TRIALS', '15'))
TEST_SIZE = float(os.getenv('N_TEST_SIZE', '0.2'))

# Control sklearn thread usage (helps when also parallelizing across seeds)
# Defaults to 1 to avoid oversubscription; can be overridden via env var.
N_JOBS_SKLEARN = int(os.getenv('N_JOBS_SKLEARN', '1'))

# Verbosity / logging
VERBOSE = os.getenv('CHEMGRID_VERBOSE', '1') == '1'
PROGRESS = os.getenv('CHEMGRID_PROGRESS', '1') == '1'  # lightweight progress prints
SUPPRESS_WARNINGS = os.getenv('CHEMGRID_SUPPRESS_WARNINGS', '1') == '1'  # default on to reduce noise
LOG_LEVEL = logging.INFO if VERBOSE else logging.WARNING
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=LOG_LEVEL)
logger = logging.getLogger('ChemGridML')

# Warning filters
if SUPPRESS_WARNINGS:
	# Always suppress frequent convergence warnings from sklearn (e.g., ElasticNet)
	warnings.filterwarnings('ignore', category=ConvergenceWarning)
	warnings.filterwarnings('ignore', category=FitFailedWarning)
	warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
	# Common numerical runtime warnings that aren't actionable during HPO
	warnings.filterwarnings('ignore', message='.*overflow encountered.*')
	warnings.filterwarnings('ignore', message='.*invalid value encountered.*')
	warnings.filterwarnings('ignore', message='.*Objective function might be poorly conditioned.*')
else:
	# If not globally suppressing, still quiet things when not verbose
	if not VERBOSE:
		warnings.filterwarnings('ignore', category=ConvergenceWarning)
		warnings.filterwarnings('ignore', message='.*overflow encountered.*')
		warnings.filterwarnings('ignore', message='.*invalid value encountered.*')
		warnings.filterwarnings('ignore', message='.*Objective function might be poorly conditioned.*')

def set_verbose(is_verbose: bool):
	"""Dynamically update verbosity and warning filters at runtime."""
	global VERBOSE
	VERBOSE = bool(is_verbose)
	logger.setLevel(logging.INFO if VERBOSE else logging.WARNING)
	# When turning verbosity off, ensure the most common warnings are hidden
	if not VERBOSE or SUPPRESS_WARNINGS:
		warnings.filterwarnings('ignore', category=ConvergenceWarning)
		warnings.filterwarnings('ignore', message='.*overflow encountered.*')
		warnings.filterwarnings('ignore', message='.*invalid value encountered.*')
		warnings.filterwarnings('ignore', message='.*Objective function might be poorly conditioned.*')

