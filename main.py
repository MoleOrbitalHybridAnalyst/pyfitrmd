from __future__ import print_function
import __builtin__
import multiprocessing as mp
import re

from parser import Parser
from fit_method import FitMethod
from fit_sd import FitSD
from lammps_input import LammpsInput
from error import Error
from weight import Weight

from lammpstrj_py2 import *

#def cal_error(itask):
#    idata = batch[itask]
#    restart = restarts[idata]
#    reference = references[idata]

if __name__ == '__main__':

    __builtin__.parser = Parser()
    __builtin__.fit_method = FitSD()
    
    # fit method determines argument list
    fit_method.add_argument()

    __builtin__.args = parser.parse()
    __builtin__.lammps_input = LammpsInput(args.lammps_input)

    # initialization
    __builtin__.restarts = np.genfromtxt(args.restarts, dtype = str)
    __builtin__.references = np.genfromtxt(args.references, dtype = str)
    __builtin__.references = [np.loadtxt(_) for _ in references]
    __builtin__.parameters = np.genfromtxt(args.parameters, dtype = str)
    __builtin__.error = Error(int(args.norm_l))
    __builtin__.weight = Weight(args.weight)
    __builtin__.para_values = dict()
    counts = np.zeros(len(parameters))
    evb_par = open(args.evb_par)
    for line in evb_par:
        for ip, p in enumerate(parameters):
            m = re.match("\s*([-.0-9eE]+).+" + p, line)
            if m:
                counts[ip] += 1
                para_values[p] = float(m.group(1))
    evb_par.close()
    fit_method.setup()

    # naive error checks
    try:
        if lammps_input.has_pattern("\s*dump\s+"):
            raise BaseException(args.lammps_input + " has dump")
        if lammps_input.has_pattern("\s*run\s+"):
            raise BaseException(args.lammps_input + " has run")
        if lammps_input.has_pattern("\s*fix\s+\S+\s+\S+\s+evb"):
            raise BaseException(args.lammps_input + " has fix evb")
        if not lammps_input.has_pattern("\s*read_restart"):
            raise BaseException(args.lammps_input + " has no read_restart")
        if len(restarts) != len(references):
            raise "%s has %d lines but %s has %d lines"%\
                (args.restarts, len(restarts), args.references, len(references))
        found_evb_par = False
        evb_in = open(args.evb_in)
        for line in evb_in:
            if re.match(".+" + args.evb_par, line):
                found_evb_par = True
        if not found_evb_par:
            raise BaseException(
                "cannot find %s in %s"%(args.evb_par, args.evb_in))
        evb_in.close()
        for c, p in zip(counts, parameters):
            if c == 0:
                raise BaseException("cannot find %s in %s"%(p, args.evb_par))
    except Exception as err:
        print(err)
        exit(1)

    # simple outputs
    print("%d data points provided"%len(restarts))
    print("%d parameters to be fitted"%len(parameters))

    fit_method.update()

    # finish things
