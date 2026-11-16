"""Generate a real graph replay from persisted model/rule outputs, not a UI mockup."""
import io
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
from finguard.data import demo

def render():
    d=demo(); results={x['tx_id']:x for x in json.loads(Path('reports/demo_results.json').read_text())}
    graph=nx.from_pandas_edgelist(d,'sender','receiver',create_using=nx.DiGraph)
    pos={f'person-{i:02}':(-1,1-i*2/19) for i in range(20)}
    pos.update({'hub':(0,0),**{f'exit-{i}':(1,.8-i*.4) for i in range(5)}}); frames=[]
    for step in [1,5,10,15,20,21,23,25]:
        view=d[d.step<=step]; g=nx.from_pandas_edgelist(view,'sender','receiver',create_using=nx.DiGraph)
        row=results[view.iloc[-1].tx_id]
        fig,axs=plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios':[1.3,1]},facecolor='#101b2c')
        for ax in axs: ax.set_facecolor('#101b2c'); ax.axis('off')
        nx.draw_networkx(g,pos,ax=axs[0],font_size=7,font_color='white',node_color=['#f5ba5d' if n=='hub' else '#258e9e' for n in g],node_size=250,labels={n:(n.replace('person-','S').replace('exit-','E')) for n in g},edge_color='#72c6c5',arrows=True)
        axs[0].set_title('20 senders → hub → 5 exits',color='white',fontsize=16)
        text=f'FINGUARD / NETWORK REPLAY\n\nSimulation step {step}\nTransfers observed: {len(view)}\n\nLatest model score: {row["score"]:.3f}\nModel threshold: {row["threshold"]:.3f}\nReview alert: {row["flagged"]}\n\nTriggered by:\n'+ ('\n'.join(row['triggers']) or 'No policy triggered')+'\n\nSynthetic demonstration\nDistinct counterparties ≠ proven laundering'
        axs[1].text(.02,.95,text,va='top',color='white',fontsize=12,linespacing=1.6)
        fig.tight_layout(); b=io.BytesIO(); fig.savefig(b,format='png',dpi=120); plt.close(fig); b.seek(0)
        frames.append(Image.open(b).convert('RGB').copy())
    frames[-1].save('reports/demo_network.png')
    frames[0].save('reports/demo_replay.gif',save_all=True,append_images=frames[1:],duration=1100,loop=0)
if __name__=='__main__': render()
