# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import copy
import yaml
from fermipy import utils
from fermipy.gtanalysis import GTAnalysis
import argparse


def run_analysis(config):
    print('Running analysis...')

    gta = GTAnalysis(config)
    gta.setup()
    gta.optimize()

    gta.print_roi()

    # Localize and generate SED for first source in ROI
    srcname = gta.roi.sources[0].name

    gta.free_source(srcname)
    gta.fit()

    gta.localize(srcname)
    gta.sed(srcname)

    gta.write_roi('roi', make_plots=True)
    gta.tsmap(make_plots=True)
    gta.residmap(make_plots=True)


def create_config(args):

    config = dict(
        fileio=dict(
            outdir=None,
        ),
        data=dict(
            evfile=None,
            ltcube=None,
            scfile=None,
        ),
        binning=dict(
            roiwidth=10.0,
            binsz=0.1,
            binsperdec=8,
            coordsys='GAL',
        ),
        selection=dict(
            emin=None,
            emax=None,
            logemin=None,
            logemax=None,
            tmin=239557417,
            tmax=512994417,
            zmax=100,
            evclass=128,
            evtype=3,
            target=None,
            ra=None,
            dec=None,
            glon=None,
            glat=None,
        ),
        gtlike=dict(
            edisp=True,
            irfs='P8R2_SOURCE_V6',
            edisp_disable=['isodiff', 'galdiff']
        ),
        model=dict(
            src_roiwidth=15.0,
            galdiff='$FERMI_DIFFUSE_DIR/v5r0/gll_iem_v06.fits',
            isodiff='iso_P8R2_SOURCE_V6_v06.txt',
            catalogs=['3FGL'],
        ),
    )

    if args['config'] is not None:
        config = utils.merge_dict(config, yaml.load(open(args['config'])))

    if args['outdir'] is not None:
        config['fileio']['outdir'] = os.path.abspath(args['outdir'])

    for s in ['data', 'selection']:
        for k in config[s].keys():
            if k in args and args[k] is not None:
                config[s][k] = args[k]

    return config


def main():
    usage = "%(prog)s [config_file] [options]"
    description = """
Run a quick analysis of an ROI performing the basic data and model
preparation and optimizing source parameters.  If the input config
file does not exist a new one will be created using the options
provided on the command-line.
"""
    parser = argparse.ArgumentParser(usage=usage, description=description)

    parser.add_argument('--config', default=None,
                        help='Set an existing configuration file that will be used as the '
                        'baseline configuration.  Note that parameters set with command-line '
                        'options (emin, emax, etc.) will override any settings in this file.')
    parser.add_argument('--outdir', default=None,
                        help='Set the path to the analysis directory.')
    parser.add_argument('--evfile', default=None,
                        help='Set the path to the FT1 file or list of FT1 files.')
    parser.add_argument('--scfile', default=None,
                        help='Set the path to the FT2 file or list of FT2 files.')
    parser.add_argument('--ltcube', default=None,
                        help='Set the path to the LT cube file.')
    parser.add_argument('--emin', default=None,
                        help='Minimum energy selection (MeV).')
    parser.add_argument('--emax', default=None,
                        help='Maximum energy selection (MeV).')
    parser.add_argument('--logemin', default=None,
                        help='Minimum energy selection (log10(MeV)).')
    parser.add_argument('--logemax', default=None,
                        help='Maximum energy selection (log10(MeV)).')
    parser.add_argument('--target', default=None,
                        help='Name of a catalog source.')
    parser.add_argument('--ra', default=None,
                        help='RA of ROI center.')
    parser.add_argument('--dec', default=None,
                        help='DEC of ROI center.')
    parser.add_argument('config_file', default=None,
                        help='Path to a configuration file.  If this file does not exist then '
                        'a new configuration file will be created.')

    args = vars(parser.parse_args())

    if not 'FERMI_DIFFUSE_DIR' in os.environ:
        os.environ['FERMI_DIFFUSE_DIR'] = '$GLAST_EXT/diffuseModels'

    if not args['config_file']:
        args['config_file'] = os.path.join(args['outdir'], 'config.yaml')

    if not os.path.isdir(os.path.dirname(args['config_file'])):
        utils.mkdir(os.path.dirname(args['config_file']))

    # Create a config file
    if not os.path.isfile(args['config_file']):
        configpath = args['config_file']
        config = create_config(args)
        yaml.dump(config, open(configpath, 'w'))
        cfgstr = yaml.dump(config, default_flow_style=False)
        print('Creating new configuration...')
        print(cfgstr)
        config_file = configpath
    else:
        print('Using existing configuration...')
        config_file = args['config_file']

    run_analysis(config_file)


if __name__ == "__main__":
    main()
