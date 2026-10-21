import sys
import json
from generate_report import create_inspection_report

if __name__ == "__main__":
    if len(sys.argv) > 1:
        kwargs = json.loads(sys.argv[1])
        create_inspection_report(**kwargs)
