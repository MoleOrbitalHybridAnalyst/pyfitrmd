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
        self._arg_parser.add_argument(
            '--evb_fixid', default = 'evb', 
            help = 'evb fix id used in restart files')
        self._arg_parser.add_argument(
            '--norm_l', default = 2, 
            help = 'L-norm for force error')
        self._arg_parser.add_argument(
            '--weight', help = 'weight file name (prefix) for atomic forces')
        self._arg_parser.add_argument(
            '--run_steps', default = 1e8,
            help = 'total steps of optimization to run')
        self._arg_parser.add_argument(
            '--verbose_stride', default = 20,
            help = 'verbose every this steps')
        self._arg_parser.add_argument(
            '--checkpoint_stride', default = 10,
            help = 'write checkpoint every this step')
        self._arg_parser.add_argument(
            '--checkpoint_name', default = 'checkpoint',
            help = 'checkpoint file name')
        self._arg_parser.add_argument(
            '--debug', action = 'store_true',
            help = 'print debug info')


    def parse(self):
        return self._arg_parser.parse_args()
