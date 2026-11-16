"""Strict ingestion and provenance. Labels are never model inputs."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

COMMIT = '7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca'
SAMPLE = '20K_fanin200cycle200'
URL = f'https://raw.githubusercontent.com/IBM/AMLSim/{COMMIT}/sample/{SAMPLE}.tgz'
COLUMNS = ['tx_id', 'step', 'sender', 'receiver', 'amount']

def validate(df, labelled=False):
    required = COLUMNS + (['label'] if labelled else [])
    if not set(required).issubset(df.columns):
        raise ValueError(f'Required columns: {required}')
    d = df[required].copy()
    if d.isna().any().any():
        raise ValueError('Missing required values: reject rather than invent transactions')
    for c in ['tx_id', 'sender', 'receiver']:
        d[c] = d[c].astype(str)
        if d[c].str.strip().eq('').any():
            raise ValueError('Empty identifiers')
    if d.tx_id.duplicated().any():
        raise ValueError('Duplicate transaction IDs')
    for c in ['step', 'amount']:
        d[c] = pd.to_numeric(d[c], errors='raise')
        if not np.isfinite(d[c]).all() or (d[c] < 0).any():
            raise ValueError('Step and amount must be finite and nonnegative')
    if (d.step % 1 != 0).any():
        raise ValueError('Step must be an integer')
    d['step'] = d.step.astype(int)
    if labelled and not d.label.isin([0, 1]).all():
        raise ValueError('Labels must be 0 or 1')
    return d.sort_values(['step', 'tx_id'], kind='stable').reset_index(drop=True)

def fetch(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    archive = root / (SAMPLE + '.tgz')
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as r:
            archive.write_bytes(r.read())
    expected = 'f187bc5784223cdee137ac603164f7183fd58f114090f39e2a12543b6bf48319'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError('AMLSim archive checksum mismatch')
    with tarfile.open(archive) as tar:
        for name in ['transactions.csv', 'nodes.csv', 'metadata.txt']:
            member = tar.extractfile(f'{SAMPLE}/{name}')
            if member is None:
                raise ValueError('Missing archive member')
            dest = root / SAMPLE / name; dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(member.read())
    return root / SAMPLE

def load_ibm(folder):
    folder = Path(folder)
    tx = pd.read_csv(folder / 'transactions.csv')
    nodes = pd.read_csv(folder / 'nodes.csv')
    if nodes.nodeid.duplicated().any() or not nodes.isFraud.isin([0,1]).all():
        raise ValueError('Invalid node labels')
    known = set(nodes.nodeid)
    if not (set(tx.sourceNodeId) | set(tx.targetNodeId)) <= known:
        raise ValueError('Transaction references unknown node')
    bad = set(nodes.loc[nodes.isFraud.eq(1), 'nodeid'])
    d = tx.rename(columns={'sourceNodeId':'sender','targetNodeId':'receiver','value':'amount','time':'step'})
    d['tx_id'] = [f'ibm-{i:07}' for i in range(len(d))]
    d['label'] = (d.sender.isin(bad) | d.receiver.isin(bad)).astype(int)
    d = validate(d, labelled=True)
    manifest = {'source':URL, 'commit':COMMIT, 'label_definition':'Either endpoint is an isFraud=1 account; NOT transaction-level laundering ground truth',
        'rows':len(d), 'accounts':len(nodes), 'positive_rate':float(d.label.mean()),
        'time_unit':'simulation step, not assumed seconds or days', 'missing_required':0,
        'files_sha256':{n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ['transactions.csv','nodes.csv']}}
    return d, manifest

def load_raw_amlsim(path):
    """Current Java simulator raw log; true isSAR labels. Explicit adapter only."""
    cols=['step','type','amount','sender','orig_before','orig_after','receiver','dest_before','dest_after','label','alert_id']
    d=pd.read_csv(path, header=None, names=cols)
    d['tx_id']=[f'raw-{i:09}' for i in range(len(d))]
    return validate(d, labelled=True)

def demo():
    rows=[]
    for i in range(20):
        rows.append({'tx_id':f'in-{i:02}', 'step':i+1,'sender':f'person-{i:02}','receiver':'hub','amount':500.0})
    for i in range(5):
        rows.append({'tx_id':f'out-{i:02}', 'step':21+i,'sender':'hub','receiver':f'exit-{i}','amount':1900.0})
    return validate(pd.DataFrame(rows))
