"""Replay the gather-scatter scenario through the same engine as the API."""
import json
from pathlib import Path
from finguard.data import demo
from finguard.service import Engine

if __name__=='__main__':
    path=Path('finguard.db')
    if path.exists():
        raise SystemExit('finguard.db already exists. Use another FINGUARD_DB via the API or move your existing demo DB before replay.')
    engine=Engine(path)
    result=engine.ingest(demo().to_dict('records'),actor='scenario-demo')
    Path('reports/demo_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'transfers':len(result),'hybrid_alerts':sum(r['flagged'] for r in result), 'model_alerts':sum('model_threshold' in r['triggers'] for r in result),'last_transfer':result[-1]},indent=2))
