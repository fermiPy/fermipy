#!/usr/bin/env python

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Set up a stacking analysis
"""
from __future__ import absolute_import, division, print_function

import os
import copy

from shutil import copyfile


from fermipy.utils import load_yaml, write_yaml

from fermipy.jobs.utils import is_null, is_not_null
from fermipy.jobs.link import Link

from fermipy.jobs import defaults

from . import NameFactory

NAME_FACTORY = NameFactory(basedir=('.'))


class PrepareTargets(Link):
    """Small class to preprare analysis pipeline.

    """
    appname = 'stack-prepare-targets'
    linkname_default = 'prepare-targets'
    usage = '%s [options]' % (appname)
    description = "Prepare directories for target analyses"

    default_options = dict(ttype=defaults.common['ttype'],
                           rosters=defaults.common['rosters'],
                           config=defaults.common['config'],
                           spatial_models=defaults.common['spatial_models'],
                           alias_dict=defaults.common['alias_dict'],
                           sims=defaults.sims['sims'],
                           dry_run=defaults.common['dry_run'])

    __doc__ += Link.construct_docstring(default_options)

    @classmethod
    def _write_data_target_config(cls, base_config, target, target_dir):
        """ Write a fermipy configuration file for one target.

        Parameters
        ----------

        base_config : dict
            Baseline configuration

        target : `dict`
            Specific target

        target_dir : str
            Directory to write to

        Returns
        -------

        output : dict
            The configuration for this specific target

        """

        target_config_path = os.path.join(target_dir, 'config.yaml')
        target_config = base_config.copy()
        target_config['selection']['ra'] = target['ra']
        target_config['selection']['dec'] = target['dec']
        write_yaml(target_config, target_config_path)
        return target_config

    @classmethod
    def _write_sim_target_config(cls, target_config, target_dir, sim_target_dir):
        """ Write a fermipy configurate file for one target
        for simulated analysis.

        This largely copies the configuration for flight data.
        It does make a few changes to point some of the input
        files (like the exposure map) to the flight data verions
        to avoid having to re-compute them.

        Parameters
        ----------

        target_config : dict
            Configuration for flight data analysis for this target

        target_dir : str
            Analysis directory for flight data for this target

        sim_target_dir : str
            Directory to write to


        Returns
        -------

        output : dict
            The configuration for this specific target

        """
        sim_target_config_path = os.path.join(sim_target_dir, 'config.yaml')
        sim_target_config = copy.deepcopy(target_config)

        comps = sim_target_config.get('components', [sim_target_config])

        for i, comp in enumerate(comps):
            comp_name = "%02i" % i
            if not comp.has_key('gtlike'):
                comp['gtlike'] = {}
            comp['gtlike']['bexpmap'] = os.path.abspath(os.path.join(target_dir, 'bexpmap_%s.fits' % comp_name))
            comp['gtlike']['srcmap'] = os.path.abspath(os.path.join(sim_target_dir, 'srcmap_%s.fits' % comp_name))
            comp['gtlike']['use_external_srcmap'] = True

        write_yaml(sim_target_config, sim_target_config_path)
        return sim_target_config

    @classmethod
    def _write_profile_yaml(cls, target, profile_path, targ_ver, spatial):
        """ Write a yaml file describing the spatial profile of the target.

        Parameters
        ----------

        target : dict
            Specific target

        profile_path : str
            Path for the output file

        targ_ver : str
            Version of the target, used for bookkeeping

        spatial : str
            Spatial model, one of ['point', 'radial', 'map']

        Returns
        -------

        output : dict
            The description of the target spatial profile

        """

        source_model = dict(SpectrumType='PowerLaw',
                            RA=target['ra'],
                            DEC=target['dec'])

        if spatial in [None, 'point']:
            source_model.update(dict(SpatialModel='PointSource'))
        elif spatial in ['map']:
            if target.j_map_file is None:
                j_map_file = profile_path.replace('.yaml', '.fits')
                target.write_jmap_wcs(j_map_file)
            source_model.update(dict(SpatialModel='DiffuseSource',
                                     SpatialType='SpatialMap',
                                     Spatial_Filename=target['map']))
        elif spatial in ['radial']:
            target.j_rad_file = profile_path.replace('.yaml', '.dat')
            target.write_j_rad_file()
            source_model.update(dict(SpatialModel='DiffuseSource',
                                     SpatialType='RadialProfile',
                                     radialprofile=target['radial']))
        else:
            raise ValueError('Did not recognize spatial type %s' % spatial)

        ver_name = "%s_%s" % (targ_ver, spatial)
        profile_dict = dict(name=ver_name,
                            source_model=source_model)
        write_yaml(profile_dict, profile_path)
        return profile_dict

    @classmethod
    def _write_astro_value_yaml(cls, target, astro_val_path):
        """Write a yaml file describing the J-factor and D-factor of one target.

        Parameters
        ----------

        target : dict
            Specific target

        astro_val_path : str
            Path for the output file

        Returns
        -------

        output : dict
            The description of the targe

        """
        astro_profile_data = target.copy()
        write_yaml(astro_profile_data, astro_val_path)
        return astro_profile_data

    @classmethod
    def _write_sim_yaml(cls, target, sim, sim_target_dir, target_key):
        """Write a yaml file describing the 'True' target parameters
        for simulated data.

        Parameters
        ----------

        target : dict
            Specific target

        sim : str
            Name of the simulation scenario.

        sim_target_dir : str
            Directory to write to

        target_key : str
            Version of the target to write

        Returns
        -------

        output : dict
            The description of the 'True' target parameters

        """

        sim_profile_yaml = os.path.join('config', 'sim_%s.yaml' % sim)
        sim_profile = load_yaml(sim_profile_yaml)
        injected_source = sim_profile.get('injected_source', None)
        if injected_source is not None:
            sim_profile['injected_source']['source_model'][
                'norm']['value'] = target['norm']
        sim_out_path = os.path.join(
            sim_target_dir, 'sim_%s_%s.yaml' %
            (sim, target_key))
        write_yaml(sim_profile, sim_out_path)
        return sim_profile

    @classmethod
    def _write_target_dirs(cls, ttype, roster_dict,
                           base_config, sims, spatial_models,
                           aliases):
        """ Create and populate directoris for target analysis.

        Parameters
        ----------

        ttype : str
            Target type, used for bookeeping and directory naming

        roster_dict : dict
            Dictionary of dict objects with the analysis targets

        base_config : dict
            Baseline configuration

        sims : list
            List of names of simulation scenarios

        spatial_models : dict
            Dictionary types of spatial models to use in analysis

        aliases : dict
            Optional dictionary to remap target verion keys

        """
        target_dict = {}

        target_info_dict = {}
        roster_info_dict = {}

        for roster_name, rost in roster_dict.items():
            for target_name, target in rost.items():

                if aliases is not None:
                    try:
                        ver_key = aliases[target.version]
                    except KeyError:
                        ver_key = target.version
                else:
                    ver_key = target.version
                target_key = "%s:%s" % (target_name, ver_key)
                print("Writing %s" % (target_key))
                name_keys = dict(target_type=ttype,
                                 target_name=target_name,
                                 target_version=ver_key,
                                 fullpath=True)
                astro_val_path = NAME_FACTORY.astro_valuefile(**name_keys)
                target_dir = NAME_FACTORY.targetdir(**name_keys)
                try:
                    os.makedirs(target_dir)
                except OSError:
                    pass
                cls._write_astro_value_yaml(target, astro_val_path)

                for sim in sims:
                    name_keys['sim_name'] = sim
                    sim_target_dir = NAME_FACTORY.sim_targetdir(**name_keys)
                    sim_astro_val_path = NAME_FACTORY.sim_astro_valuefile(**name_keys)
                    try:
                        os.makedirs(sim_target_dir)
                    except OSError:
                        pass
                    cls._write_astro_value_yaml(target, sim_astro_val_path)
                    cls._write_sim_yaml(target, sim, sim_target_dir, ver_key)
                name_keys.pop('sim_name')

                write_config = False
                if target_name in target_dict:
                    # Already made the config for this target
                    target_config = target_dict[target_name].copy()
                else:
                    # Make the config for this target
                    target_config = cls._write_data_target_config(base_config,
                                                                  target, target_dir)
                    target_dict[target_name] = target_config
                    write_config = True

                write_sim_config = write_config
                for spatial in spatial_models:
                    ver_string = "%s_%s" % (ver_key, spatial)
                    roster_key = "%s_%s" % (roster_name, spatial)
                    full_key = "%s:%s" % (target_name, ver_string)

                    name_keys['profile'] = ver_string
                    profile_path = NAME_FACTORY.profilefile(**name_keys)

                    if target_name in target_info_dict:
                        target_info_dict[target_name].append(ver_string)
                    else:
                        target_info_dict[target_name] = [ver_string]

                    cls._write_profile_yaml(target, profile_path,
                                            ver_key, spatial)

                    if roster_key in roster_info_dict:
                        roster_info_dict[roster_key].append(full_key)
                    else:
                        roster_info_dict[roster_key] = [full_key]

                    for sim in sims:
                        name_keys['sim_name'] = sim
                        sim_target_dir = NAME_FACTORY.sim_targetdir(**name_keys)
                        sim_profile_path = NAME_FACTORY.sim_profilefile(**name_keys)
                        if write_sim_config:
                            cls._write_sim_target_config(target_config,
                                                         target_dir, sim_target_dir)
                        cls._write_profile_yaml(target, sim_profile_path,
                                                ver_key, spatial)
                    write_sim_config = False

        roster_file = os.path.join(ttype, 'roster_list.yaml')
        target_file = os.path.join(ttype, 'target_list.yaml')

        write_yaml(roster_info_dict, roster_file)
        write_yaml(target_info_dict, target_file)

        for sim in sims:
            sim_dir = os.path.join("%s_sim" % ttype, "sim_%s" % sim)
            sim_roster_file = os.path.join(sim_dir, 'roster_list.yaml')
            sim_target_file = os.path.join(sim_dir, 'target_list.yaml')
            try:
                os.makedirs(sim_dir)
            except OSError:
                pass
            copyfile(roster_file, sim_roster_file)
            copyfile(target_file, sim_target_file)

    def run_analysis(self, argv):
        """Run this analysis"""
        args = self._parser.parse_args(argv)

        if not args.rosters:
            raise RuntimeError("You must specify at least one target roster")

        if is_null(args.ttype):
            raise RuntimeError("You must specify a target type")

        if is_null(args.sims):
            sims = []
        else:
            sims = args.sims

        if is_null(args.alias_dict):
            aliases = None
        else:
            aliases = load_yaml(args.alias_dict)

        name_keys = dict(target_type=args.ttype,
                         fullpath=True)
        config_file = NAME_FACTORY.ttypeconfig(**name_keys)

        if is_not_null(args.config):
            config_file = args.config

        roster_dict = {}
        for roster in args.rosters:
            a_roster = load_yaml(roster)
            roster_dict.update(a_roster)

        base_config = load_yaml(config_file)
        self._write_target_dirs(args.ttype, roster_dict, base_config,
                                sims, args.spatial_models, aliases)


def register_classes():
    """Register these classes with the `LinkFactory` """
    PrepareTargets.register_class()
