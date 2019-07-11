import __builtin__
from fit_method import FitMethod

class FitGA(FitMethod):

    def __init__(self):
        super(FitGA, self).__init__()

    def add_argument(self):
        super(FitGA, self).add_argument()
        __builtin__.parser._arg_parser.add_argument(
            '--ga_input', default = 'ga.inp', help = 'input file for GA')

    def setup(self):
        self._input = np.
