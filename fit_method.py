from lammps import lammps

class FitMethod(object):

    def __init__(self):
        pass

    def add_argument(self):
        pass

    def setup(self):
        """
        set up class based on args parsed
        """
        pass

    def update(self):
        """
        run lammps and then
        update parameters based on lammps results
        """
        pass

    def verbose(self):
        pass

    def checkpoint(self):
        pass
