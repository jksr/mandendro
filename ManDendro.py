import pandas as pd
import numpy as np

class ManDendro:
    def __init__(self,Z, labels, labelcolors=None):
        self.Zdf = self.prepare_link_df(Z)
        self.currZdf = self.Zdf.copy()
        self.opq = []
        self.labels = list(labels)

        if labelcolors is not None:
            colormap = dict()
        if isinstance(labelcolors, dict):
            colormap = labelcolors
        else:
            colormap = dict(zip(self.labels, labelcolors))
        self.colormap = {x:colormap.get(x,'#ffffff') for x in labels}
    
    @staticmethod
    def prepare_link_df(Z):
        Zdf = pd.DataFrame(Z, columns=['left','right','dist','nleafs'])
        Zdf = Zdf.astype({'left':int,'right':int,'dist':float,'nleafs':int})
        Zdf.index = [len(Zdf.values)+1+ind for ind in range(len(Zdf.values))]
        return Zdf
    
    def switch_node(self, nid, undo=False):
        self.currZdf.loc[nid, 'left'], self.currZdf.loc[nid, 'right'] = \
            self.currZdf.loc[nid, 'right'], self.currZdf.loc[nid, 'left']
        if not undo:
            self.opq.append(nid)
        return self.currZdf
    
    def undo(self):
        self.switch_node(self.opq.pop())
    
    def plot_dendro(self, Zdf=None, ax=None, labelcolors=True):
        import scipy.cluster.hierarchy as sch
        if Zdf is None:
            Zdf = self.currZdf
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1,1,figsize=(20,6))
        
        ilabels = [f'{x.Index}={x.left}+{x.right}' for x in self.Zdf.itertuples()]
        
        dn = sch.dendrogram(Zdf.values, labels=self.labels, ax=ax, link_color_func=lambda x: 'k')
        ii = np.argsort(np.array(dn['dcoord'])[:, 1])
        for j, (icoord, dcoord) in enumerate(zip(dn['icoord'], dn['dcoord'])):
            x = 0.5 * sum(icoord[1:3])
            y = dcoord[1]
            ind = np.nonzero(ii == j)[0][0]
            ax.annotate(ilabels[ind], (x,y), va='top', ha='center')
        
        positions = ax.get_xticks()
        texts = ax.get_xticklabels()
        w = (positions[1]-positions[0])/2
        for pos,txt in zip(positions, texts):
            txt = txt.get_text()
            ax.axvspan(pos-w, pos+w, facecolor=self.colormap[txt], alpha=0.5)
        # plt.gca().axis('off')    
            
        return ax
    
    def get_current_linkage(self):
        return self.currZdf.values.copy()