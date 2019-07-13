import argparse

class Parser(object):

    def __init__(self):
        self._arg_parser = argparse.ArgumentParser(
         formatter_class = argparse.ArgumentDefaultsHelpFormatter)
        self._arg_parser.add_argument(
            '--restarts', default = 'restarts.list', 
            help = 'list of LAMMPS restart files')
        self._arg_parser.add_argument(
            '--references', default = 'references.list', 
            help = 'list of reference force files')
        self._arg_parser.add_argument(
            '--lammps_input', default = 'lmp.in', 
            help = 'LAMMPS input file')
        self._arg_parser.add_argument(
            '--parameters', default = 'parameters.list', 
            help = 'parameters to be fitted')
        self._arg_parser.add_argument(
            '--evb_in', default = 'evb.cfg', 
            help = 'evb input file')
        self._arg_parser.add_argument(
            '--evb_par', default = 'evb.par', 
            help = 'evb.par file included in evb_in')

    def parse(self):
        return self._arg_parser.parse_args()
