import numpy as np
import pandas as pd
import pytest
from finguard.data import demo, validate, load_raw_amlsim
from finguard.features import build, ALL
from finguard.experiment import threshold

def test_future_cannot_change_past():
    d=demo(); before,x=build(d[d.step<=15]); after,y=build(d)
    np.testing.assert_array_equal(x,y.iloc[:len(x)])

def test_labels_and_ids_are_not_features():
    d=demo(); d['label']=0; _,x=build(d); d['label']=1; _,y=build(d)
    pd.testing.assert_frame_equal(x,y)
    assert not {'label','sender','receiver','tx_id','fraudStep','isFraud'} & set(ALL)

def test_same_step_isolation():
    d=demo().iloc[:2].copy(); d['step']=1
    _,x=build(d)
    assert x.dst_in_degree.tolist()==[0,0]
    _,y=build(d.iloc[::-1]); pd.testing.assert_frame_equal(x,y)

def test_window_expiration():
    d=demo().iloc[[0,1]].copy(); d.loc[d.index[1],'step']=32
    _,x=build(d); assert x.iloc[-1].dst_in_degree==0

def test_gather_scatter():
    _,x=build(demo()); assert x.iloc[-1].src_in_degree==20
    assert .94 < x.iloc[-1].src_flow_ratio < .96

def test_directed_triangle():
    d=pd.DataFrame([['1',1,'a','b',10],['2',2,'b','c',10],['3',3,'c','a',10]],columns=['tx_id','step','sender','receiver','amount'])
    _,x=build(d); assert x.closes_triangle.tolist()==[0,0,1]

@pytest.mark.parametrize('column,value',[('amount',float('nan')),('amount',-1),('step',1.5),('receiver',''),('amount',float('inf'))])
def test_invalid_records(column,value):
    d=demo(); d.loc[0,column]=value
    with pytest.raises(ValueError): validate(d)

def test_duplicate_rejected():
    d=demo(); d.loc[1,'tx_id']=d.loc[0,'tx_id']
    with pytest.raises(ValueError): validate(d)

def test_validation_threshold_ties():
    y=np.array([0,0,1,1]); s=np.array([.2,.5,.5,.8]); t=threshold(y,s,.0)
    assert t>.5 and ((s[y==0]>=t).mean()==0)

def test_raw_amlsim_adapter(tmp_path):
    p=tmp_path/'raw.csv'; p.write_text('1,TRANSFER,50,a,100,50,b,0,50,1,42\n')
    d=load_raw_amlsim(p); assert d.label.tolist()==[1]
    assert 'alert_id' not in d
