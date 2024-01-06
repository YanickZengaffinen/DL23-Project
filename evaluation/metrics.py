import torch
import numpy as np
import scipy.stats as st

class ConfidenceIntervalOnAccuracy:

    def __init__(self, interval: float = 0.95):
        self.interval = interval
        self.accs = []
    
    def reset(self):
        self.accs = []

    def update(self, pred, lbls):
        self.accs.append(100*(torch.argmax(pred, dim=-1) == lbls).sum().item() / pred.shape[0])
    
    def compute(self):
        l,u = st.t.interval(confidence=self.interval, df=len(self.accs)-1, loc=np.mean(self.accs), scale=st.sem(self.accs))
        return f'{(l+u)/2:.2f} \\pm {(u-l)/2:.2f}'
    
