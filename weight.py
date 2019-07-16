from numpy import loadtxt

class Weight(object):

    def __init__(self, name):
        self._name = name
        self._weights = []
        if name is not None:
            for ipoint in range(len(restarts)):
                try:
                    self._weights.append(loadtxt(self._name))
                except:
                    self._weights.append(loadtxt("%s.%d"%(self._name, ipoint)))

    def get(self, ipoint):
        if self._name is None:
            return None
        return self._weights[ipoint]
