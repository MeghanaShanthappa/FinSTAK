from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
OLD = Path('/var/tmp/work-mshanthappa/finstak_recovered/finstak-100x10-v1')
sys.path.insert(0, str(OLD / 'source_snapshot'))
spec = importlib.util.spec_from_file_location('fresh_check', OLD / 'delivery_scripts/verify_scale_study.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT = HERE / 'run_view'
module.main()
