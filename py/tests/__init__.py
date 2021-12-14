from pathlib import Path
import sys

path_package = Path(__file__).parents[1]
sys.path.insert(0, path_package)
