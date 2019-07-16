from numpy import ones

class Error(object):

    def __init__(self, l):
        if l == 1:
            self._norm = lambda x : abs(x)
        elif l == 2:
            self._norm = lambda x : x**2
        else:
            raise NotImplemented()

    def compute(self, tag, f, ref, w):
        err = 0
        if w is None:
            w = ones(len(ref))
        for i in xrange(len(ref)):
            _id = tag[i]
            for k in xrange(3):
                err += w[_id-1] * self._norm(f[i][k] - ref[_id-1][k+1])
        return err
