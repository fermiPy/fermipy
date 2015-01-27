import logging 

def logLevel(level):
    '''This is a function that returns a python like
    level from a HEASOFT like level.'''

    levels_dict = {0:50,
                1:40,
                2:30,
                3:20,
                4:10}

    if not isinstance(level,int):
        level = int(level)

    if level > 4:
        level = 4

    return levels_dict[level]


class Logger(object):

    def __init__(self, base, name, loglevel=logging.DEBUG):
        name = name.replace('.log','')
        logger = logging.getLogger('%s' % name)    # log_namespace can be replaced with your namespace 
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            fh = logging.FileHandler(base+'_'+name+'.log')
            fh.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(loglevel)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            logger.addHandler(fh)
            logger.addHandler(ch)
        self._logger = logger

    def get(self):
        return self._logger

