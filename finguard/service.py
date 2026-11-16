"""Atomic, bounded SQLite transaction ingestion for a local research prototype."""
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, DMatrix
from .features import build, evidence
from .data import validate

SCHEMA='''
CREATE TABLE IF NOT EXISTS transactions (
 tx_id TEXT PRIMARY KEY, step INTEGER NOT NULL CHECK(step>=0), sender TEXT NOT NULL,
 receiver TEXT NOT NULL, amount REAL NOT NULL CHECK(amount>=0), score REAL NOT NULL,
 flagged INTEGER NOT NULL, reasons TEXT NOT NULL, features TEXT NOT NULL, model_version TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS tx_step ON transactions(step);
CREATE INDEX IF NOT EXISTS tx_sender ON transactions(sender,step);
CREATE INDEX IF NOT EXISTS tx_receiver ON transactions(receiver,step);
CREATE TABLE IF NOT EXISTS audit (
 id INTEGER PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 action TEXT NOT NULL, actor TEXT NOT NULL, batch_sha256 TEXT NOT NULL, row_count INTEGER NOT NULL);
'''

class Conflict(ValueError): pass

class Engine:
    def __init__(self,db='finguard.db',model_dir='models'):
        self.db=str(db); model_dir=Path(model_dir)
        self.meta=json.loads((model_dir/'metadata.json').read_text())
        self.model=XGBClassifier(); self.model.load_model(model_dir/'xgb_full.json')
        self.fingerprint=hashlib.sha256((model_dir/'xgb_full.json').read_bytes()).hexdigest()
        with sqlite3.connect(self.db) as con: con.executescript(SCHEMA)
    def ingest(self,records,actor='ingest'):
        d=validate(pd.DataFrame(records))
        if not 1<=len(d)<=1000: raise ValueError('Batch size must be 1–1000')
        with sqlite3.connect(self.db,timeout=15) as con:
            con.execute('BEGIN IMMEDIATE')
            prior=pd.read_sql_query('SELECT tx_id,step,sender,receiver,amount FROM transactions ORDER BY step,tx_id',con)
            if len(prior)+len(d)>10000: raise ValueError('Demo database limit is 10,000 transactions')
            if len(prior) and (d.step.min()<=prior.step.max() or d.tx_id.isin(prior.tx_id).any()):
                raise Conflict('Submit a later step and unique transaction IDs; same-step transfers belong in one batch')
            merged=pd.concat([prior,d],ignore_index=True) if len(prior) else d
            ordered,x=build(merged,self.meta['window'])
            mask=ordered.tx_id.isin(d.tx_id); new=ordered.loc[mask]; xx=x.loc[mask]
            matrix=xx[self.meta['features']]
            scores=self.model.predict_proba(matrix)[:,1]
            contributions=self.model.get_booster().predict(DMatrix(matrix),pred_contribs=True)
            output=[]
            for (_,r),(_,f),score,contrib in zip(new.iterrows(),xx.iterrows(),scores,contributions):
                facts=evidence(f)
                top=np.argsort(np.abs(contrib[:-1]))[-3:][::-1]
                facts.extend(f'Model contribution: {self.meta["features"][i]} = {contrib[i]:+.3f} log-odds' for i in top)
                model_flagged=bool(score>=self.meta['threshold'])
                rule_flagged=bool(f.src_in_degree>=20 and f.src_flow_ratio>=0.5)
                flagged=model_flagged or rule_flagged
                triggers=(["model_threshold"] if model_flagged else [])+(["gather_scatter_rule"] if rule_flagged else [])
                if triggers: facts.insert(0, 'Alert policy triggered: '+', '.join(triggers))
                result={**r.to_dict(),'score':float(score),'flagged':flagged,'reasons':facts,
                    'model_version':self.fingerprint,'threshold':self.meta['threshold'], 'triggers':triggers, 'contributions':dict(zip(self.meta['features']+['base_value'],map(float,contrib)))}
                output.append(result)
                con.execute('INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (r.tx_id,int(r.step),r.sender,r.receiver,float(r.amount),float(score),int(flagged),json.dumps(facts),json.dumps(f.to_dict()),self.fingerprint))
            digest=hashlib.sha256(d.to_json(orient='records').encode()).hexdigest()
            con.execute('INSERT INTO audit(action,actor,batch_sha256,row_count) VALUES (?,?,?,?)',('ingest',actor,digest,len(d)))
            return output
    def alerts(self,limit=100):
        with sqlite3.connect(self.db) as con:
            con.row_factory=sqlite3.Row
            return [{**dict(r),'reasons':json.loads(r['reasons']),'features':json.loads(r['features'])} for r in con.execute(
                'SELECT * FROM transactions WHERE flagged=1 ORDER BY score DESC LIMIT ?', (min(max(limit,1),1000),))]
