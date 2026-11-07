"""공통 실행기의 MovieLens 1M 경로."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_full_recsys_pipeline import main
if __name__ == '__main__':
    sys.argv.extend(['--dataset','1m'])
    main()
