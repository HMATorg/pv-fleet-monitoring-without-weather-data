import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
PAL=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#4a3aa7','#e34948']
SEQ='Blues'
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,
 'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.color':'#e6e6e3','grid.linewidth':0.5,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b',
 'lines.linewidth':1.2,'savefig.dpi':300,'savefig.bbox':'tight','figure.dpi':150})
W1=3.5; W2=7.16
import matplotlib.figure as _mf
_orig=_mf.Figure.savefig
def _save(self,fname,*a,**k):
    _orig(self,fname,*a,**k)
    if str(fname).endswith('.png'): _orig(self,str(fname)[:-4]+'.pdf',*a,**k)
_mf.Figure.savefig=_save
