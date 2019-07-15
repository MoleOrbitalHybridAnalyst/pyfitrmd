from __future__ import print_function
from multiprocessing import cpu_count, Process, Queue
from time import time, sleep
from numpy.random import seed, shuffle
from numpy import array, argsort
from re import match

from fit_method import FitMethod, lammps

class FitSD(FitMethod):

    def __init__(self):
        super(FitSD, self).__init__()

    def add_argument(self):
        super(FitSD, self).add_argument()
        parser._arg_parser.add_argument(
            '--sd_dx', default = 1e-4, 
            help = 'displacement for numerical derivatives')
        parser._arg_parser.add_argument(
            '--sd_lr', default = 0.01, help = 'initial learning rate')
        parser._arg_parser.add_argument(
            '--sd_batch_size', default = 'ncpus', help = 'batch size')
        parser._arg_parser.add_argument(
            '--sd_ndecay', default = 100, 
            help = 'decay learning rate every this steps')
        parser._arg_parser.add_argument(
            '--sd_seed', default = 'time', 
            help = 'seed for random shuffle')

    def setup(self):
        super(FitSD, self).setup()
        if args.sd_batch_size == 'ncpus':
            self._batch_size = cpu_count()
        else:
            self._batch_size = int(args.sd_batch_size)
        self._lr = float(args.sd_lr)
        self._dx = float(args.sd_dx)
        self._ndecay = float(args.sd_ndecay)
        if args.sd_seed == 'time':
            self._sd_seed = int(time())
        else:
            self._sd_seed = int(args.sd_seed)
        seed(self._sd_seed)
        self._indexes = list(xrange(len(restarts)))
        self._cnt_pos = 0
        shuffle(self._indexes)

    def make_batch(self):
        npoints = len(self._indexes)
        end = self._cnt_pos + self._batch_size
        if  end > npoints:
            self._batch = self._indexes[self._cnt_pos:]
            self._batch.extend(self._indexes[:end-npoints])
        else:
            self._batch = \
                self._indexes[self._cnt_pos:end]
        self._cnt_pos = end % npoints

    def update_evb_in(self):
        for ip, pname in enumerate(parameters):
            for idx, dx in enumerate([self._dx, -self._dx]):
                fp = open("%s.%d.%d"%(args.evb_in, ip, idx), 'w')
                evb_in = open(args.evb_in)
                for line in evb_in:
                    if match(".+" + args.evb_par, line):
                        print("#include \"%s.%d.%d\""\
                            %(args.evb_par, ip, idx), file = fp)
                    else:
                        print(line, file = fp, end = '')
                fp.close(); evb_in.close()
                fp = open("%s.%d.%d"%(args.evb_par, ip, idx), 'w')
                evb_par = open(args.evb_par)
                for line in evb_par:
                    if match("\s*([-.0-9eE]+).+" + pname, line):
                        print(para_values[pname] + dx, file = fp)
                    else:
                        print(line, file = fp, end = '')
                fp.close(); evb_par.close()

    def one_point_gradient(self, queue, ipoint):
        lmp = lammps(cmdargs = ["-screen", "none"])
#        lmp = lammps()
        restart = restarts[ipoint]
        ref = references[ipoint]
        for line in lammps_input._lines:
            if match("\s*read_restart", line):
                lmp.command("read_restart " + restart)
            else:
                lmp.command(line)

        gradient = []
        for ip in xrange(len(parameters)):
            errors = []
            for idx in range(2):
                suffix = "%d.%d"%(ip, idx)
                lmp.command("fix %s all evb %s.%s evb.out.%s evb.top"%\
                    (args.evb_fixid, args.evb_in, suffix, suffix))
                lmp.command("run 0")
                tag = lmp.extract_atom("id", 0)
                f = lmp.extract_atom("f", 3)
                # compute error
                errors.append( error.compute(tag, f, ref, weight.get(ipoint)) )
            gradient.append( (errors[0] - errors[1]) / 2 / self._dx )
        lmp.close()
        queue.put(array(gradient))
        print(ipoint, gradient)

    def update(self):
        self.update_evb_in()
        self.make_batch()
        q = Queue()
        processes = []
        for indx in self._batch:
            p = Process(target = self.one_point_gradient, args = (q, indx))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        gradients = sum( [q.get() for indx in self._batch] )
        print(gradients)
        # TODO delete temporary evb.cfg evb.out and evb.par
        # TODO update parameters based on the gradient
