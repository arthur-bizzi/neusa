import numpy as np

class Metrics:
    def __init__(self,u_pred,u_exact):
        self.u_pred = u_pred
        self.u_exact = u_exact
    
    def relative_l2_error(self):
        return np.sqrt(np.sum((self.u_pred-self.u_exact)**2)/np.sum((self.u_exact)**2))

    def relative_l1_error(self):
        return np.sum(np.abs(self.u_pred-self.u_exact))/np.sum(np.abs(self.u_exact))
    
    def rl2_by_time(self):
        nt = len(self.u_pred)
        rl2s = []
        for n in range(1,nt+1):
            rl2 = np.sqrt(np.sum((self.u_pred[:n]-self.u_exact[:n])**2)/np.sum((self.u_exact[:n])**2))
            rl2s.append(rl2)
        return rl2s

    def rl1_by_time(self):
        nt = len(self.u_pred)
        rl1s = []
        for n in range(1,nt+1):
            rl1 = np.sum(np.abs(self.u_pred[:n]-self.u_exact[:n]))/np.sum(np.abs(self.u_exact[:n]))
            rl1s.append(rl1)
        return rl1s
