"""
Convenience wrapper to run dataset download module.
"""
import sys
from dataset.download_dataset.run import main

if __name__ == "__main__":
    main(sys.argv[1:])
