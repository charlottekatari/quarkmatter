import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
import pandas as pd, numpy as np
sys.path.insert(0,str(BASE_DIR))
from stage4_fast import Matter,Background,Prepared,find_roots
from stage3_hybrid_tov import HBARC
mat=Matter(str(BASE_DIR/'eos.zip'))
tr=pd.read_csv(str(BASE_DIR/'stage2_results/MIT_Bag_Maxwell_candidates.csv'))
out=str(BASE_DIR/'stage4_results')
rows=[]
for br,pcs in {165:[5,30,250,500,800],175:[27],185:[83,125,130],190:[112.5]}.items():
 Pt=float(tr[tr.B_root_MeV==br]['P_transition_MeV_fm^-3'].iloc[0]);B=br**4/HBARC**3
 for pc in pcs:
  bg=Background(pc,mat,Pt,B,max_step=.12)
  ob=Prepared(bg,step=.023,surf_factor=10)
  for mode in ['rapid','slow']:
   roots=find_roots(ob,mode)
   row=dict(Broot=br,Pc=pc,M=bg.mass,R=bg.R,Rq=bg.rq,scenario=mode,first_root=roots[0][0] if roots else np.nan,first_nodes=roots[0][1] if roots else np.nan,root_count=len(roots),roots=repr(roots));rows.append(row)
   print('additional',row,flush=True)
add=pd.DataFrame(rows);old=pd.read_csv(out+'/radial_mode_trials.csv');pd.concat([old,add],ignore_index=True).sort_values(['Broot','Pc','scenario']).to_csv(out+'/radial_mode_trials.csv',index=False)
checks=[]
for br,pc in [(0,350),(175,21),(175,90),(185,100),(185,200),(185,425),(190,120),(190,180)]:
 if br:
  Pt=float(tr[tr.B_root_MeV==br]['P_transition_MeV_fm^-3'].iloc[0]);B=br**4/HBARC**3
 else:Pt=B=None
 bg=Background(pc,mat,Pt,B,max_step=.1)
 for mode in (['rapid','slow'] if br else ['pure']):
  for step in [.015,.028]:
   for sf in [10,100]:
    roots=find_roots(Prepared(bg,step,sf),mode)
    checks.append(dict(Broot=br,Pc=pc,mode=mode,step_km=step,surface_factor=sf,first_root=roots[0][0] if roots else np.nan,first_nodes=roots[0][1] if roots else np.nan))
  print('check',br,pc,mode,flush=True)
pd.DataFrame(checks).to_csv(out+'/convergence_radial.csv',index=False)
