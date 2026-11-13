from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
import numpy as np,pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
out=Path(str(BASE_DIR/'stage4_results')); d=pd.read_csv(out/'radial_mode_trials.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':13,'axes.labelsize':11,'axes.grid':True,'grid.alpha':.2,'figure.dpi':140,'savefig.dpi':210})
fig,axes=plt.subplots(3,1,figsize=(9,10.2),sharex=False,layout='constrained')
colors={'rapid':'#1e5b94','slow':'#dc7b30'}
for ax,b in zip(axes,[175,185,190]):
 df=d[(d.Broot==b) & d.first_root.notna()]
 for scenario in ['rapid','slow']:
  t=df[df.scenario==scenario].sort_values('Pc')
  ax.plot(t.Pc,t.first_root*1e3,marker='o',ms=6,linewidth=1.8,color=colors[scenario],label=('Швидке перетворення' if scenario=='rapid' else 'Повільне перетворення'))
 ax.axhline(0,color='black',linewidth=1,linestyle='--')
 ax.axhspan(-10,0,color='#ffd3d3',alpha=.12)
 ax.set_title(r'$B^{1/4}='+str(b)+r'\ \mathrm{MeV}$',loc='left',fontweight='bold')
 ax.set_ylabel(r'$\omega_0^2 \times 10^3$, км$^{-2}$')
 ax.set_xlabel(r'Центральний тиск $P_c$, МеВ/фм$^3$')
 if b==175:ax.legend(frameon=False,ncol=2,loc='lower right')
fig.suptitle('Радіальна стійкість гібридних зір: дві швидкості фазового перетворення',fontsize=14,fontweight='bold')
fig.savefig(out/'01_radial_omega2_scenarios.png',bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(8.4,5.1),layout='constrained')
for b,c in zip([175,185,190],['#9467bd','#339999','#a65432']):
 df=d[(d.Broot==b)&(d.scenario=='rapid')&d.first_root.notna()].sort_values('Pc')
 ax.plot(df.Pc,df.first_root,marker='o',color=c,lw=2,label=f'{b} МеВ')
ax.axhline(0,color='black',ls='--',lw=1)
ax.set(xlabel=r'Центральний тиск $P_c$, МеВ/фм$^3$',ylabel=r'$\omega_0^2$, км$^{-2}$',title='Фундаментальна радіальна мода за швидкого переходу')
ax.legend(title=r'$B^{1/4}$',frameon=False)
fig.savefig(out/'02_radial_rapid_compare.png',bbox_inches='tight');plt.close(fig)

# clean quality: warning plot shows control at B=0 and reference
fig,ax=plt.subplots(figsize=(7.8,4.6),layout='constrained')
q=pd.read_csv(out/'convergence_radial.csv')
for br,pc,mode,color in [(175,21,'rapid','#1e5b94'),(185,100,'rapid','#ba563b'),(190,120,'slow','#369376')]:
 a=q[(q.Broot==br)&(q.Pc==pc)&(q['mode']==mode)]
 for sf,m in [(10,'o'),(100,'s')]:
  g=a[a.surface_factor==sf].sort_values('step_km')
  ax.plot(g.step_km,g.first_root*1000,marker=m,color=color,ls='-' if sf==10 else '--',lw=1.5,label=f'{br} МеВ, {mode}, поріг ×{sf}')
ax.axhline(0,c='gray',lw=1)
ax.set(xlabel='Радіальний крок, км',ylabel=r'$\omega_0^2 \times 10^3$, км$^{-2}$',title='Контроль збіжності за кроком і зовнішньою межею')
ax.legend(fontsize=8,frameon=False,ncol=2)
fig.savefig(out/'03_radial_convergence.png',bbox_inches='tight');plt.close(fig)
print('Plots created',list(str(x) for x in out.glob('*.png')))
