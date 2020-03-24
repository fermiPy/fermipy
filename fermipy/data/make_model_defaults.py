import os
from fermipy import gtutils
from fermipy.utils import tolist
import yaml
import numpy as np

gtutils.init_function_pars()


par_names = gtutils.FUNCTION_PAR_NAMES

o = {}

for k, v in par_names.items():

    o.setdefault(k,{})
    o[k]['par_names'] = v
    o[k]['norm_par'] = gtutils.FUNCTION_NORM_PARS[k]
    o[k]['defaults'] = gtutils.FUNCTION_DEFAULT_PARS[k]

    for pname, p in o[k]['defaults'].items():
        o[k]['defaults'][pname]['error'] = np.nan

o['CompositeSource'] = {'defaults' : {}, 'norm_par' : None,
                        'par_names': [] }

modelfile = os.path.join('$FERMIPY_DATA_DIR', 'models.yaml')


yaml.dump(tolist(o),
          open(os.path.expandvars(modelfile), 'w'))#,default_flow_style=False)
