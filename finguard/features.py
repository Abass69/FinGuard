"""Strict-past rolling graph features. Same-step transfers cannot see one another."""
from collections import defaultdict, deque, Counter
import numpy as np
import pandas as pd
from .data import validate

TX = ['log_amount']
BEHAVIOR = ['src_count','dst_count','src_amount_ratio','dst_amount_ratio']
GRAPH = ['src_in_degree','src_out_degree','dst_in_degree','dst_out_degree','reverse_edge','closes_triangle','src_flow_ratio']
ALL = TX + BEHAVIOR + GRAPH

class History:
    def __init__(self, window=30):
        self.window=window; self.queue=deque()
        self.ins=defaultdict(Counter); self.outs=defaultdict(Counter)
        self.in_sum=defaultdict(float); self.out_sum=defaultdict(float)
        self.in_n=Counter(); self.out_n=Counter()
    def expire(self, step):
        while self.queue and self.queue[0][0] < step-self.window:
            _,s,r,a=self.queue.popleft()
            for book,key,other in [(self.outs,s,r),(self.ins,r,s)]:
                book[key][other]-=1
                if book[key][other]==0: del book[key][other]
            self.out_sum[s]-=a; self.in_sum[r]-=a
            self.out_n[s]-=1; self.in_n[r]-=1
    def features(self, row):
        s,r,a=row.sender,row.receiver,row.amount
        src_n=self.out_n[s]; dst_n=self.in_n[r]
        return [np.log1p(a),src_n,dst_n,
            a/(self.out_sum[s]/src_n+1) if src_n else 1.,
            a/(self.in_sum[r]/dst_n+1) if dst_n else 1.,
            len(self.ins[s]),len(self.outs[s]),len(self.ins[r]),len(self.outs[r]),
            int(s in self.outs[r]),
            int(bool(self.outs[r].keys() & self.ins[s].keys())),
            min(100.,(self.out_sum[s]+a)/(self.in_sum[s]+1)) if self.in_n[s] else 0.]
    def add(self,row):
        s,r,a=row.sender,row.receiver,row.amount
        self.queue.append((row.step,s,r,a)); self.outs[s][r]+=1; self.ins[r][s]+=1
        self.out_sum[s]+=a; self.in_sum[r]+=a; self.out_n[s]+=1; self.in_n[r]+=1

def build(df, window=30):
    d=validate(df, labelled='label' in df)
    history=History(window); result=[]
    for step,batch in d.groupby('step',sort=True):
        history.expire(step)
        result.extend(history.features(row) for row in batch.itertuples())
        for row in batch.itertuples(): history.add(row)
    return d,pd.DataFrame(result,columns=ALL)

def graph_score(x):
    return np.maximum(x.src_in_degree,x.dst_in_degree).to_numpy()+5*x.reverse_edge.to_numpy()+5*x.closes_triangle.to_numpy()

def rule_score(x):
    return ((x.src_in_degree>=10)&(x.src_flow_ratio>=0.5)).astype(float).to_numpy()+ (x.dst_in_degree>=10).astype(float).to_numpy()+x.closes_triangle.to_numpy()

def evidence(row):
    facts=[]
    if row.src_in_degree>=10:
        facts.append(f'Sender received from {int(row.src_in_degree)} distinct counterparties in the previous 30 steps')
    if row.dst_in_degree>=10:
        facts.append(f'Receiver already had {int(row.dst_in_degree)} distinct incoming counterparties in the previous 30 steps')
    if row.src_in_degree>=10 and row.src_flow_ratio>=0.5:
        facts.append(f'Outgoing total including this transfer is {row.src_flow_ratio:.0%} of recent incoming value')
    if row.reverse_edge: facts.append('A reverse-direction transfer exists in the prior window')
    if row.closes_triangle: facts.append('This transfer closes a directed three-account cycle in the prior window')
    return facts or ['Model score reflects the combined amount, activity and network features; no configured motif explanation fired']
