import os
import copy
import yaml
import numpy as np
import healpy as hp
import fermipy.utils as utils
import argparse

def clone_configs(basedir,base_configs,opt_configs,scripts):
    """
    """
    config = {}
    for c in base_configs:
        config = utils.merge_dict(config,yaml.load(open(c)),
                                  add_new_keys=True)   
        pass

    bash_scripts = []
    for script_in in scripts:
        bash_script = """
cat $0
python {script} --config={config}
"""
        scriptdir = os.path.join(basedir,'scripts')
        utils.mkdir(scriptdir)
        os.system('cp %s %s'%(script_in,scriptdir))
        bash_scripts.append(bash_script)

    for name, vdict in opt_configs.items():

        dirname = os.path.join(basedir,name)    
        utils.mkdir(dirname)
 
        cfgfile = os.path.abspath(os.path.join(dirname,'config.yaml'))
        for script_in,bash_script in zip(scripts,bash_scripts):
            script = os.path.basename(script_in)
            scriptpath = os.path.abspath(os.path.join(dirname,script))
            os.system('ln -sf %s %s'%(os.path.abspath(os.path.join(scriptdir,script)),
                                      scriptpath))
            runscript = os.path.abspath(os.path.join(dirname,
                                                     os.path.splitext(script)[0] + '.sh'))
            with open(os.path.join(runscript),'wt') as f:
                f.write(bash_script.format(source=name,config=cfgfile,
                                           script=scriptpath))

        if not config:
            continue
 
        c = copy.deepcopy(config)
        c = utils.merge_dict(c,vdict,add_new_keys=True)
        yaml.dump(utils.tolist(c),open(cfgfile,'w'),default_flow_style=False)
        pass


def main():

    usage = "usage: %(prog)s [config files]"
    description = "Clone configuration files and make directory structure"
    parser = argparse.ArgumentParser(usage=usage,description=description)

    parser.add_argument('--basedir', default = None, required=True)
    parser.add_argument('--source_list', default = None, required=True,
                        help='YAML file containing a list of sources to be '
                        'analyzed.')
    parser.add_argument('--script', action='append', required=True,
                        help='The python script.')
    parser.add_argument('configs', nargs='*', default = None,
                        help='One or more configuration files that will be merged '
                        'to construct the analysis configuration.')

    args = parser.parse_args()

    #src_dict = {}
    #src_list = yaml.load(open(args.source_list))
    
    #for src,sdict in src_list.items():
    #    src_dict[src] = sdict

    src_dict = yaml.load(open(args.source_list))
    
    clone_configs(args.basedir,args.configs,src_dict,args.script)

if __name__ == "__main__":
    main()
        
