import __builtin__

from parser import Parser
from fit_method import FitMethod
from fit_ga import FitGA

from lammpstrj_py2 import *

if __name__ == '__main__':

    __builtin__.parser = Parser()
    __builtin__.fit_method = FitGA()
    
    fit_method.add_argument()

    __builtin__.args = parser.parse()

    # read inputs
    __builtin__.lammps_input = open(args.lammps_input)
    __builtin__.restarts = np.genfromtxt(args.restarts, dtype = str)
    __builtin__.references = np.genfromtxt(args.references, dtype = str)
    __builtin__.parameters = np.genfromtxt(args.parameters, dtype = str)
