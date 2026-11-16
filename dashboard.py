"""Local investigator dashboard. Read-only SQL; synthetic data only."""
import json
import os
import sqlite3
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title='FinGuard | Investigator workspace',page_icon='🛡️',layout='wide')
st.title('FinGuard')
st.caption('FINANCIAL NETWORK INTELLIGENCE · RESEARCH PROTOTYPE')
st.info('Synthetic data only. A flag is a review lead, not proof of fraud. Scores are uncalibrated ranking signals.')
root=Path(__file__).parent
mode=st.sidebar.radio('Workspace',['Investigation','Research evaluation'])
if mode=='Research evaluation':
    p=root/'reports/metrics.csv'
    if p.exists():
        st.subheader('Temporal holdout · IBM AMLSim account-associated transfers')
        st.dataframe(pd.read_csv(p),hide_index=True,width='stretch')
        st.image(str(root/'reports/precision_recall.png'))
        st.json(json.loads((root/'reports/ablation.json').read_text()))
    else: st.warning('Run python -m finguard.experiment first.')
else:
    db=Path(os.getenv('FINGUARD_DB',str(root/'finguard.db'))).resolve()
    if not db.exists():
        st.warning('Run python scripts/demo.py first to create the demonstration ledger.'); st.stop()
    with sqlite3.connect(f'file:{db}?mode=ro',uri=True) as con:
        tx=pd.read_sql_query('SELECT * FROM transactions',con)
        audit=pd.read_sql_query('SELECT * FROM audit ORDER BY id DESC LIMIT 30',con)
    if tx.empty: st.info('No transactions yet.'); st.stop()
    a,b,c=st.columns(3); a.metric('Monitored transfers',len(tx)); b.metric('Review alerts',int(tx.flagged.sum())); c.metric('Accounts',len(set(tx.sender)|set(tx.receiver)))
    st.subheader('Investigator queue')
    show_all=st.checkbox('Include transfers below the model threshold')
    queue=tx if show_all else tx[tx.flagged.eq(1)]
    queue=queue.sort_values('score',ascending=False)
    st.dataframe(queue[['tx_id','step','sender','receiver','amount','score','flagged']],hide_index=True,width='stretch')
    if not queue.empty:
        chosen=st.selectbox('Inspect transfer',queue.tx_id.tolist()); row=tx[tx.tx_id.eq(chosen)].iloc[0]
        left,right=st.columns([1,1])
        with left:
            st.subheader('Observed evidence')
            for reason in json.loads(row.reasons): st.write('• '+reason)
            st.caption('Network evidence describes observed behavior. TreeSHAP contributions explain the model log-odds, not causation or criminal intent.')
            st.json(json.loads(row.features))
        with right:
            # Same-step neighbors excluded; selected edge shown distinctly.
            context=tx[(tx.step<row.step)&(tx.step>=row.step-30)&(tx.sender.isin([row.sender,row.receiver])|tx.receiver.isin([row.sender,row.receiver]))].tail(100)
            graph=nx.from_pandas_edgelist(context,'sender','receiver',create_using=nx.DiGraph)
            graph.add_edge(row.sender,row.receiver)
            fig,ax=plt.subplots(figsize=(7,5)); pos=nx.spring_layout(graph,seed=42)
            nx.draw_networkx(graph,pos,ax=ax,node_size=420,font_size=7,node_color=['#f5b74f' if n in [row.sender,row.receiver] else '#8ad2cc' for n in graph],arrows=True)
            ax.set_title('Prior 30 steps + selected transfer'); ax.axis('off'); st.pyplot(fig); plt.close(fig)
    st.download_button('Export review queue CSV',queue.drop(columns=['features']).to_csv(index=False),'finguard_alerts.csv','text/csv')
    with st.expander('Ingestion audit trail'): st.dataframe(audit,hide_index=True)
