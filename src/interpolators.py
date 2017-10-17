from coreUtils import *

#returnFirstLarger = lambda X, ax: returnFirstLargerCy(X.squeeze(), ax)

class leftNeighboor():
    def __init__(self, t,x):
        self.t = np.array(t).squeeze()
        self.x = np.array(x).reshape((-1, self.t.size))
        
        assert( np.all(t[1::]-t[0:-1] > 0))
        self.dim = self.x.shape[0]
        
    def __call__(self, t):
        t=np.array(t).squeeze()
        thisInd = np.maximum(np.searchsorted(self.t, t)-1,0)
        x = self.x[:,thisInd]

        return x
        
        