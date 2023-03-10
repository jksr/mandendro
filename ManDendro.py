import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from typing import Dict, Optional

class ManDendro:
    """
    A class to perform manipulations on the dendrogram in a hierarchical clustering.

    Attributes:
        Zdf: Pandas DataFrame; dataframe representation of hierarchical clustering.
        currZdf: Pandas DataFrame; copy of Zdf, representing the current state of dendrogram
        opq: List[int]; list to hold the IDs of nodes which got switched.
        labels: list[string]; List of strings containing the labels of data points.
        colormap: Dict[string, string]; Dictionary containing mapping of label to colors.

    Methods:
        prepare_link_df(Z): Static method that returns data frame representing hierarchical clustering
        switch_node(nid, undo): switches the two ends of a node in the current state. 
                                    After switch, appends node ID into the operation queue (opq).
                                    If undo flag is set to True, does not append node to opq.


        undo(): Undoes last switch operation performed by reverting a switch operation from the opq list.

        plot_dendro(Zdf, ax, labelcolors):
                    Plots the Dendrogram for provided links on matplotlib axes.
                    If links are not passed in then it will plot for internal linkages stored in `currZdf`.
                    If axes are not supplied, it will create them within a new figure.
                    If Label Colors dictionary is also provided then it plots each cluster with their respective defined 
                    color palettes over the dendrogram. 

        get_current_linkage(): Returns the hierarchical clustering linkage matrix in its current state.

        """
    def __init__(self,Z, labels, labelcolors: Optional[Dict[str, str]] = None):
        """
        Initializes the ManDendro object

        Arguments:
            Z: ndarray; n*4 array representing linkage matrix in format returned by SciPy linkage function.
            labels: List; items can be any hashable object; len(labels) should match the number of rows in Z.
            labelcolors: dict; key-value pairs where keys are elements of labels and values are hex-codes for
                        RGB colors. Default value is None.


        Returns:
            None.
        """

        self.Zdf = self.prepare_link_df(Z)
        self.currZdf = self.Zdf.copy()
        self.opq = []
        self.labels = list(labels)

        if labelcolors is None:
            colormap = dict()
        elif isinstance(labelcolors, dict):
            colormap = labelcolors
        else:
            colormap = dict(zip(self.labels, labelcolors))
        #Assigning colormap different colors from palette to each unique label
        #assigning defaultcolor='white' where there is no color defined for the corresponding label.
        self.colormap = {x:colormap.get(x,'#ffffff') for x in labels}

    
    @staticmethod
    def prepare_link_df(Z: np.array) -> pd.DataFrame:
        """
        Method to convert linkage format to Pandas DataFrame

        Arguments:
            Z: ndarray; n*4 array representing linkage matrix in format returned by SciPy linkage function.

        Returns:
            Pandas DataFrame; containing the columns ['left','right','dist','nleafs'] and index starting from len(Z)+2.

        """
        Zdf = pd.DataFrame(Z, columns=['left','right','dist','nleafs'])
        Zdf = Zdf.astype({'left':int,'right':int,'dist':float,'nleafs':int})
        Zdf.index = [len(Zdf.values)+1+ind for ind in range(len(Zdf.values))]
        return Zdf

    

        
    def switch_node(self, nid: int, undo: bool=False) -> pd.DataFrame:
        """
        Method to switch the two ends of a node in the current state. 
        After switching the node, appends the node ID into the operation queue (opq).
        
        Arguments:
        nid: int; node ID
        undo: bool; optional switch operation revert flag.

        Returns:
            currZdf: Pandas DataFrame; modified dataframe after switch operation.
            Appends operation to opq list if undo=False, otherwise modifies dataframe without appending operation to opq. 
 
        """
        self.currZdf.loc[nid, 'left'], self.currZdf.loc[nid, 'right'] = \
            self.currZdf.loc[nid, 'right'], self.currZdf.loc[nid, 'left']
        if not undo:
            self.opq.append(nid)
        return self.currZdf

    def undo(self) -> None:
        """
        Reverts the last switch operation performed by reverting a switch operation from the opq list.

        Arguments:
            None.

        Returns:
            None. Reverts the last performed switch operation.

        """
        self.switch_node(self.opq.pop())
    
    def plot_dendro(self, Zdf=None, ax=None, labelcolors=True):
        """
        Plots the Dendrogram for provided links on matplotlib axes.
        
        Arguments:
            Zdf: Pandas DataFrame;
                DataFrame having the linkage information which needs to be plotted.
            ax: Matplotlib axis;
                Axis object where the plot is to be plotted.
            labelcolors: bool or dict; Specify if colors are to be applied over the labels.
                                    If True, it will use colors from previously defined label-color mapping.
                                    If dict, it will define same label-color mapping for the current plot only.

        Returns:
            Matplotlib axis object having dendrogram plot

      """
        import scipy.cluster.hierarchy as sch
        if Zdf is None:
            Zdf = self.currZdf
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1,1,figsize=(20,6))
        
        ilabels = [f'{x.Index}={x.left}+{x.right}' for x in self.Zdf.itertuples()]
        
        dend = sch.dendrogram(Zdf.values, labels=self.labels, ax=ax, link_color_func=lambda x: 'k')
        ii = np.argsort(np.array(dend['dcoord'])[:, 1])
        for j, (icoord, dcoord) in enumerate(zip(dend['icoord'], dend['dcoord'])):
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
        """
        Method to retrieve the hierarchical clustering linkage matrix in its current state.

        Arguments:
            None.

        Returns:
            Numpy ndarray: n * 4 array representing linkage matrix in the format returned 
                           by SciPy linkage function, which contains left and right indices, 
                           distance and number of leaf nodes comprising each node at the time.  
        """        
        return self.currZdf.values.copy()