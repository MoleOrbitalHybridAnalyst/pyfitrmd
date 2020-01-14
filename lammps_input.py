import re

class LammpsInput(object):

    def __init__(self, fname):
        self._fname = fname
        self._lines = []
        fp = open(fname)
        for line in fp:
            self._lines.append(line)

    def has_pattern(self, pattern):
        for line in self._lines:
            if re.match(pattern, line):
                return True
        return False
