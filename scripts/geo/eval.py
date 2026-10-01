import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

# The evaluation lives in the package, so that it is also available from an
# installation that is not a checkout of the repository (command `mbm-eval`).
from massbalancemachine.cli.evaluate import main

if __name__ == "__main__":
    main()
