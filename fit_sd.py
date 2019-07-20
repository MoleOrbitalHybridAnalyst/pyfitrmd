from __future__ import print_function
from multiprocessing import cpu_count, Process, Queue
from time import time, sleep
from numpy.random import seed, shuffle
from numpy import array, argsort, sign
from re import match
from os import path, system
import json

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
            '--sd_lr', default = 1e-3, help = 'initial learning rate')
        parser._arg_parser.add_argument(
            '--sd_batch_size', default = 'ncpus', help = 'batch size')
        parser._arg_parser.add_argument(
            '--sd_ndecay', default = 1000, 
            help = 'decay learning rate every this steps')
        parser._arg_parser.add_argument(
            '--sd_seed', default = 'time', 
            help = 'seed for random shuffle')
        parser._arg_parser.add_argument(
            '--sd_update_bound', default = 0.2, 
            help = 'update parameters up to this ratio')
        parser._arg_parser.add_argument(
            '--sd_nobound_limit', default = 0.0,
            help = 'do not apply bound if abs of parameter lower than this')

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
        self._istep = 0
        self._bound = float(args.sd_update_bound)
        self._nb_lim = float(args.sd_nobound_limit)
        print("FitSD: batch size =", self._batch_size)

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
        # this is stupid (kind of duplicated), modify it in the future:
        fp = open("%s.step%d"%(args.evb_in, self._istep), 'w')
        evb_in = open(args.evb_in)
        for line in evb_in:
            if match(".+" + args.evb_par, line):
                print("#include \"%s.step%d\""\
                    %(args.evb_par, self._istep), file = fp)
            else:
                print(line, file = fp, end = '')
        fp.close(); evb_in.close()
        fp = open("%s.step%d"%(args.evb_par, self._istep), 'w')
        evb_par = open(args.evb_par)
        for line in evb_par:
            is_para_line = False
            for ip, pname in enumerate(parameters):
                if match("\s*([-.0-9eE]+).+" + pname + "\s+", line):
                    print(para_values[pname], file = fp)
                    is_para_line = True
                    break
            if not is_para_line:
                print(line, file = fp, end = '')
        fp.close(); evb_par.close()

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
                    is_para_line = False
                    for ip_, pname_ in enumerate(parameters):
                        if match("\s*([-.0-9eE]+).+" + pname_ + "\s+", line):
                            if ip_ == ip:
                                print(para_values[pname_] + dx, file = fp)
                            else:
                                print(para_values[pname_], file = fp)
                            is_para_line = True
                            break
                    if not is_para_line:
                        print(line, file = fp, end = '')
                fp.close(); evb_par.close()

    def one_point_gradient(self, queue, ipoint):
        lmp = lammps(cmdargs = ["-screen", "none"])
#        lmp = lammps()
        restart = restarts[ipoint]
        ref = references[ipoint]
        info = "DEBUG: istep %d ipoint %d:"%(self._istep, ipoint)
        for line in lammps_input._lines:
            if match("\s*read_restart", line):
                if args.debug:
                    print(info, "read_restart " + restart)
                lmp.command("read_restart " + restart)
            else:
                lmp.command(line)

        # do one step with current parameters to provide 
        # evb.top for following EVB calculations
        if args.debug:
            print(info, "fix %s all evb %s.step%d evb.out.step%d evb.top"%\
            (args.evb_fixid, args.evb_in, self._istep, self._istep))
        lmp.command("fix %s all evb %s.step%d evb.out.step%d evb.top"%\
            (args.evb_fixid, args.evb_in, self._istep, self._istep))
        if args.debug:
            print(info, "run 0")
        lmp.command("run 0")
#        if not path.isfile("evb.top.%d"%ipoint):
#            lmp.command("write_top evb.top.%d"%ipoint)
        if args.debug:
            print(info, "write_top evb.top.%d"%ipoint)
        lmp.command("write_top evb.top.%d"%ipoint)
        tag = lmp.extract_atom("id", 0)
        f = lmp.extract_atom("f", 3)
        curr_error = error.compute(tag, f, ref, weight.get(ipoint))
        gradient = []
        for ip in xrange(len(parameters)):
            errors = []
            for idx in range(2):
                info = "DEBUG: istep %d ipoint %d ip %d idx %d:"\
                        %(self._istep, ipoint, ip, idx)
                # @@@@
#                if args.debug:
#                    print( "istep", self._istep, "ipoint", ipoint, 
#                            "ip", ip, "idx", idx)
                suffix = "%d.%d"%(ip, idx)
                if args.debug:
                    print(info, "fix %s all evb %s.%s evb.out.%s evb.top.%d"%\
                    (args.evb_fixid, args.evb_in, suffix, suffix, ipoint))
                lmp.command("fix %s all evb %s.%s evb.out.%s evb.top.%d"%\
                    (args.evb_fixid, args.evb_in, suffix, suffix, ipoint))
                # @@@@
#                if args.debug:
#                    print( "istep", self._istep, "ipoint", ipoint, 
#                            "ip", ip, "idx", idx)
                if args.debug:
                    print(info, "run 0")
                lmp.command("run 0")
                # @@@@
#                if args.debug:
#                    print( "istep", self._istep, "ipoint", ipoint, 
#                            "ip", ip, "idx", idx)
                tag = lmp.extract_atom("id", 0)
                f = lmp.extract_atom("f", 3)
                # compute error
                errors.append( error.compute(tag, f, ref, weight.get(ipoint)) )
#            print(errors)
            gradient.append( (errors[0] - errors[1]) / 2 / self._dx )
            # sanity check for errors
            if abs(errors[0] - errors[1]) / abs(sum(errors)) > 0.1:
                print("WARNING: largely deivated errros for " +\
                    "istep = %d ipoint = %d ip = %d idx = %d"\
                    %(self._istep, ipoint, ip, idx))
        lmp.close()
        queue.put((array(gradient), curr_error))
        if args.debug:
            print("queue.put at istep %d ipoint %d"%(self._istep, ipoint))
#        print(ipoint, gradient)

    def update(self):
        super(FitSD, self).update()
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
        results = [q.get() for indx in self._batch]
        self._gradient = \
            sum( [results[_][0] for _ in xrange(self._batch_size)] )
        self._curr_error = \
            sum( [results[_][1] for _ in xrange(self._batch_size)] )
        # sanity check for evb.out.*
        print("Number of inf or nan in evb.out:")
        system("egrep \"inf|nan|Nan|NaN\" evb.out.* | wc -l")
        system("rm %s.* %s.* evb.top.* evb.out.*"%(args.evb_in, args.evb_par))
        for ip, pname in enumerate(parameters):
            move = self._lr * self._gradient[ip]
            abs_p = abs(para_values[pname])
            if abs_p > self._nb_lim and abs(move) > self._bound * abs_p:
                move = sign(move) * self._bound * abs_p
                print("WARNING:", pname, 
                      "update out of bound at step", self._istep)
            if abs_p <= self._nb_lim and abs(move) > self._bound * abs_p:
                move = sign(move) * 2 * abs_p
                print("WARNING:", pname, 
                      "update out of bound at step", self._istep)
            para_values[pname] -= move
        self._istep += 1
        if self._istep % self._ndecay == 0:
            self._lr /= 2

    def verbose(self):
        super(FitSD, self).verbose()
        print("error =", self._curr_error)
        print("para_values =", para_values)
        print("gradient = ", self._gradient)
        print("learning rate = ", self._lr)

    def checkpoint(self):
        super(FitSD, self).checkpoint()
        j = json.dumps(para_values)
        f = open("%s.step%d"%(args.checkpoint_name, self._istep), "w")
        f.write(j)
        f.close()
