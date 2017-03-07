# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import copy
import yaml
from fermipy import utils
import argparse
import subprocess


def cmd_exists(cmd):
    return subprocess.call("type " + cmd, shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE) == 0


def clone_configs(basedir, base_configs, opt_configs, scripts, args=''):
    """
    """
    config = {}
    for c in base_configs:
        config = utils.merge_dict(config, yaml.load(open(c)),
                                  add_new_keys=True)

    scriptdir = os.path.abspath(os.path.join(basedir, 'scripts'))
    utils.mkdir(scriptdir)
    bash_scripts = []
    for script_in in scripts:
        bash_script = """
cat $0
{scriptexe} --config={config} {args}
"""

        if os.path.isfile(script_in):
            script = os.path.basename(script_in)
            scriptpath = os.path.join(scriptdir, script)
            scriptexe = 'python ' + scriptpath
            os.system('cp %s %s' % (script_in, scriptdir))
        elif cmd_exists(script_in):
            scriptexe = script_in
            script = script_in
        else:
            raise Exception('Could not find script: %s' % script_in)

        bash_scripts.append((script, scriptexe, bash_script))

    for name, vdict in opt_configs.items():

        dirname = os.path.abspath(os.path.join(basedir, name))
        utils.mkdir(dirname)

        cfgfile = os.path.join(dirname, 'config.yaml')
        for script_in, bash_script in zip(scripts, bash_scripts):
            runscript = os.path.splitext(bash_script[0])[0] + '.sh'
            runscript = os.path.join(dirname, runscript)
            with open(os.path.join(runscript), 'wt') as f:
                f.write(bash_script[2].format(source=name,
                                              scriptexe=bash_script[1],
                                              config=cfgfile,
                                              args=args))

        if not config:
            continue

        c = copy.deepcopy(config)
        c = utils.merge_dict(c, vdict, add_new_keys=True)
        yaml.dump(utils.tolist(c), open(cfgfile, 'w'), default_flow_style=False)


def main():
    usage = "usage: %(prog)s [config files]"
    description = "Clone configuration files and make directory structure"
    parser = argparse.ArgumentParser(usage=usage, description=description)

    parser.add_argument('--basedir', default=None, required=True)
    parser.add_argument('--source_list', default=None, required=True, action='append',
                        help='YAML file containing a list of sources to be '
                             'analyzed.')
    parser.add_argument('--script', action='append', required=True,
                        help='The python script.')
    parser.add_argument('--args', default='',
                        help='Extra script arguments.')
    parser.add_argument('--num_config', default=None, type=int,
                        help='Number of analyis directories to create.')
    parser.add_argument('configs', nargs='*', default=None,
                        help='One or more configuration files that will be merged '
                             'to construct the analysis configuration.')

    args = parser.parse_args()

    src_dict = {}
    for x in args.source_list:
        src_dict.update(yaml.load(open(x)))
        
    if args.num_config:
        src_dict = { k : src_dict[k] for k in
                     list(src_dict.keys())[:args.num_config] }
    
    clone_configs(args.basedir, args.configs, src_dict, args.script, args.args)


if __name__ == "__main__":
    main()
