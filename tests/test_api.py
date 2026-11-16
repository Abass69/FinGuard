from pathlib import Path
import sqlite3
import pytest
from fastapi.testclient import TestClient
from finguard.api import app,engine
from finguard.data import demo

KEY='test-only-ingestion-key-123456789'
READ='test-only-investigator-key-987654321'
@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('FINGUARD_DB',str(tmp_path/'test.db'))
    monkeypatch.setenv('FINGUARD_MODELS',str(Path(__file__).parents[1]/'models'))
    monkeypatch.setenv('FINGUARD_INGEST_KEY',KEY)
    monkeypatch.setenv('FINGUARD_READ_KEY',READ)
    engine.cache_clear()
    with TestClient(app) as c: yield c
    engine.cache_clear()

def payload(): return {'transactions':demo().to_dict('records')}

def test_auth_and_roles(client):
    assert client.post('/transactions',json=payload()).status_code==401
    assert client.post('/transactions',json=payload(),headers={'X-API-Key':READ}).status_code==401
    assert client.get('/alerts',headers={'X-API-Key':KEY}).status_code==401

def test_ingestion_alerts_and_duplicate_atomicity(client):
    r=client.post('/transactions',json=payload(),headers={'X-API-Key':KEY}); assert r.status_code==200
    rows=r.json()['transactions']; last=next(x for x in rows if x['tx_id']=='out-04')
    assert last['flagged'] and 'gather_scatter_rule' in last['triggers']
    import math
    assert abs(1/(1+math.exp(-sum(last['contributions'].values())))-last['score'])<1e-6
    assert any('20 distinct' in s for s in last['reasons'])
    assert client.post('/transactions',json=payload(),headers={'X-API-Key':KEY}).status_code==409
    with sqlite3.connect(engine().db) as db:
        assert db.execute('SELECT count(*) FROM transactions').fetchone()[0]==25
        assert db.execute('SELECT count(*) FROM audit').fetchone()[0]==1
    assert client.get('/alerts',headers={'X-API-Key':READ}).status_code==200

def test_replay_equals_incremental(client):
    from finguard.service import Engine
    import tempfile
    records=demo()
    direct=engine().ingest(records.to_dict('records'))
    with tempfile.TemporaryDirectory() as td:
        other=Engine(Path(td)/'second.db',Path(__file__).parents[1]/'models'); incremental=[]
        for _,batch in records.groupby('step'): incremental.extend(other.ingest(batch.to_dict('records')))
        assert [(x['score'],x['flagged']) for x in direct]==[(x['score'],x['flagged']) for x in incremental]

def test_label_injection_rejected(client):
    p=payload(); p['transactions'][0]['label']=1
    assert client.post('/transactions',json=p,headers={'X-API-Key':KEY}).status_code==422

def test_fail_closed_without_config(client,monkeypatch):
    monkeypatch.delenv('FINGUARD_INGEST_KEY')
    assert client.post('/transactions',json=payload(),headers={'X-API-Key':KEY}).status_code==503
