# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Compute the residual cosmic-ray contamination
"""
from __future__ import absolute_import, division, print_function

import sys
import argparse

import numpy as np

import healpy

from fermipy.skymap import HpxMap
from fermipy import fits_utils
from fermipy.jobs.file_archive import FileFlags
from fermipy.jobs.scatter_gather import ConfigMaker
from fermipy.jobs.lsf_impl import build_sg_from_link
from fermipy.jobs.chain import add_argument, Link, Chain
from fermipy.diffuse.binning import Component
from fermipy.diffuse.name_policy import NameFactory
from fermipy.diffuse.gt_split_and_bin import create_sg_split_and_bin
from fermipy.diffuse.job_library import create_sg_gtexpcube2
from fermipy.diffuse import defaults as diffuse_defaults



NAME_FACTORY_DIRTY = NameFactory()
NAME_FACTORY_CLEAN = NameFactory()


class ResidualCRAnalysis(object):
    """Small class to analyze the residual cosmic-ray contaimination.
    """
    default_options = dict(ccube_dirty=(None, 'Input counts cube for dirty event class.', str),
                           ccube_clean=(None, 'Input counts cube for clean event class.', str),
                           bexpcube_dirty=(None, 'Input exposure cube for dirty event class.', str),
                           bexpcube_clean=(None, 'Input exposure cube for clean event class.', str),
                           hpx_order=diffuse_defaults.residual_cr['hpx_order_fitting'],
                           coordsys=diffuse_defaults.residual_cr['coordsys'],
                           outfile=(None, 'Name of output file', str),
                           select_factor=(5.0, 'Pixel selection factor for Aeff Correction',
                                          float),
                           mask_factor=(2.0, 'Pixel selection factor for output mask',
                                        float),
                           sigma=(3.0, 'Width of gaussian to smooth output maps [degrees]', float),
                           full_output=(False, 'Include diagnostic output', bool),
                           clobber=(False, 'Overwrite output file', bool),)

    def __init__(self, **kwargs):
        """C'tor
        """
        self.parser = ResidualCRAnalysis._make_parser()
        self.link = ResidualCRAnalysis._make_link(**kwargs)

    @staticmethod
    def _make_parser():
        """Make an argument parser for this class """
        usage = "usage: %(prog)s [options] "
        description = "Compute the residual cosmic-ray contamination."

        parser = argparse.ArgumentParser(usage=usage, description=description)
        for key, val in ResidualCRAnalysis.default_options.items():
            add_argument(parser, key, val)
        return parser

    @staticmethod
    def _make_link(**kwargs):
        link = Link(kwargs.pop('linkname', 'residual_cr'),
                    appname='fermipy-residual-cr',
                    options=ResidualCRAnalysis.default_options.copy(),
                    file_args=dict(ccube_dirty=FileFlags.input_mask,
                                   bexpcube_dirty=FileFlags.input_mask,
                                   ccube_clean=FileFlags.input_mask,
                                   bexpcube_clean=FileFlags.input_mask,
                                   outfile=FileFlags.output_mask),
                    **kwargs)
        return link

    @staticmethod
    def _match_cubes(ccube_clean, ccube_dirty,
                     bexpcube_clean, bexpcube_dirty,
                     hpx_order):
        """ Match the HEALPIX scheme and order of all the input cubes

        return a dictionary of cubes with the same HEALPIX scheme and order
        """
        if hpx_order == ccube_clean.hpx.order:
            ccube_clean_at_order = ccube_clean
        else:
            ccube_clean_at_order = ccube_clean.ud_grade(hpx_order, preserve_counts=True)

        if hpx_order == ccube_dirty.hpx.order:
            ccube_dirty_at_order = ccube_dirty
        else:
            ccube_dirty_at_order = ccube_dirty.ud_grade(hpx_order, preserve_counts=True)

        if hpx_order == bexpcube_clean.hpx.order:
            bexpcube_clean_at_order = bexpcube_clean
        else:
            bexpcube_clean_at_order = bexpcube_clean.ud_grade(hpx_order, preserve_counts=True)

        if hpx_order == bexpcube_dirty.hpx.order:
            bexpcube_dirty_at_order = bexpcube_dirty
        else:
            bexpcube_dirty_at_order = bexpcube_dirty.ud_grade(hpx_order, preserve_counts=True)

        if ccube_dirty_at_order.hpx.nest != ccube_clean.hpx.nest:
            ccube_dirty_at_order = ccube_dirty_at_order.swap_scheme()

        if bexpcube_clean_at_order.hpx.nest != ccube_clean.hpx.nest:
            bexpcube_clean_at_order = bexpcube_clean_at_order.swap_scheme()

        if bexpcube_dirty_at_order.hpx.nest != ccube_clean.hpx.nest:
            bexpcube_dirty_at_order = bexpcube_dirty_at_order.swap_scheme()

        ret_dict = dict(ccube_clean=ccube_clean_at_order,
                        ccube_dirty=ccube_dirty_at_order,
                        bexpcube_clean=bexpcube_clean_at_order,
                        bexpcube_dirty=bexpcube_dirty_at_order)
        return ret_dict

    @staticmethod
    def _compute_intensity(ccube, bexpcube):
        """ Compute the intensity map
        """
        bexp_data = np.sqrt(bexpcube.data[0:-1, 0:] * bexpcube.data[1:, 0:])
        intensity_data = ccube.data / bexp_data
        intensity_map = HpxMap(intensity_data, ccube.hpx)
        return intensity_map

    @staticmethod
    def _compute_mean(map1, map2):
        """ Make a map that is the mean of two maps
        """
        data = (map1.data + map2.data) / 2.
        return HpxMap(data, map1.hpx)

    @staticmethod
    def _compute_ratio(top, bot):
        """ Make a map that is the ratio of two maps
        """
        data = np.where(bot.data > 0, top.data / bot.data, 0.)
        return HpxMap(data, top.hpx)

    @staticmethod
    def _compute_diff(map1, map2):
        """ Make a map that is the difference of two maps
        """
        data = map1.data - map2.data
        return HpxMap(data, map1.hpx)

    @staticmethod
    def _compute_product(map1, map2):
        """ Make a map that is the product of two maps
        """
        data = map1.data * map2.data
        return HpxMap(data, map1.hpx)

    @staticmethod
    def _compute_counts_from_intensity(intensity, bexpcube):
        """ Make the counts map from the intensity
        """
        data = intensity.data * np.sqrt(bexpcube.data[1:] * bexpcube.data[0:-1])
        return HpxMap(data, intensity.hpx)

    @staticmethod
    def _compute_counts_from_model(model, bexpcube):
        """ Make the counts maps from teh mdoe
        """
        data = model.data * bexpcube.data
        ebins = model.hpx.ebins
        ratio = ebins[1:] / ebins[0:-1]
        half_log_ratio = np.log(ratio) / 2.
        int_map = ((data[0:-1].T * ebins[0:-1]) + (data[1:].T * ebins[1:])) * half_log_ratio
        return HpxMap(int_map.T, model.hpx)

    @staticmethod
    def _make_bright_pixel_mask(intensity_mean, mask_factor=5.0):
        """ Make of mask of all the brightest pixels """
        mask = np.zeros((intensity_mean.data.shape), bool)
        nebins = len(intensity_mean.data)
        sum_intensity = intensity_mean.data.sum(0)
        mean_intensity = sum_intensity.mean()
        for i in range(nebins):
            mask[i, 0:] = sum_intensity > (mask_factor * mean_intensity)
        return HpxMap(mask, intensity_mean.hpx)

    @staticmethod
    def _get_aeff_corrections(intensity_ratio, mask):
        """ Compute a correction for the effective area from the brighter pixesl
        """
        nebins = len(intensity_ratio.data)
        aeff_corrections = np.zeros((nebins))
        for i in range(nebins):
            bright_pixels_intensity = intensity_ratio.data[i][mask.data[i]]
            mean_bright_pixel = bright_pixels_intensity.mean()
            aeff_corrections[i] = 1. / mean_bright_pixel

        print ("Aeff correction: ", aeff_corrections)
        return aeff_corrections

    @staticmethod
    def _apply_aeff_corrections(intensity_map, aeff_corrections):
        """ Multipy a map by the effective area correction
        """
        data = aeff_corrections * intensity_map.data.T
        return HpxMap(data.T, intensity_map.hpx)

    @staticmethod
    def _fill_masked_intensity_resid(intensity_resid, bright_pixel_mask):
        """ Fill the pixels used to compute the effective area correction with the mean intensity
        """
        filled_intensity = np.zeros((intensity_resid.data.shape))
        nebins = len(intensity_resid.data)
        for i in range(nebins):
            masked = bright_pixel_mask.data[i]
            unmasked = np.invert(masked)
            mean_intensity = intensity_resid.data[i][unmasked].mean()
            filled_intensity[i] = np.where(masked, mean_intensity, intensity_resid.data[i])
        return HpxMap(filled_intensity, intensity_resid.hpx)

    @staticmethod
    def _smooth_hpx_map(hpx_map, sigma):
        """ Smooth a healpix map using a Gaussian
        """
        if hpx_map.hpx.ordering == "NESTED":
            ring_map = hpx_map.swap_scheme()
        else:
            ring_map = hpx_map
        ring_data = ring_map.data.copy()
        nebins = len(hpx_map.data)
        smoothed_data = np.zeros((hpx_map.data.shape))
        for i in range(nebins):
            smoothed_data[i] = healpy.sphtfunc.smoothing(
                ring_data[i], sigma=np.radians(sigma), verbose=False)

        smoothed_data.clip(0., 1e99)
        smoothed_ring_map = HpxMap(smoothed_data, ring_map.hpx)
        if hpx_map.hpx.ordering == "NESTED":
            return smoothed_ring_map.swap_scheme()
        else:
            return smoothed_ring_map

    @staticmethod
    def _intergral_to_differential(hpx_map, gamma=-2.0):
        """ Convert integral quantity to differential quantity

        Here we are assuming the spectrum is a powerlaw with index gamma and we
        are using log-log-quadrature to compute the integral quantities.
        """
        nebins = len(hpx_map.data)
        diff_map = np.zeros((nebins + 1, hpx_map.hpx.npix))
        ebins = hpx_map.hpx.ebins
        ratio = ebins[1:] / ebins[0:-1]
        half_log_ratio = np.log(ratio) / 2.
        ratio_gamma = np.power(ratio, gamma)
        #ratio_inv_gamma = np.power(ratio, -1. * gamma)

        diff_map[0] = hpx_map.data[0] / ((ebins[0] + ratio_gamma[0] * ebins[1]) * half_log_ratio[0])
        for i in range(nebins):
            diff_map[i + 1] = (hpx_map.data[i] / (ebins[i + 1] *
                                                  half_log_ratio[i])) - (diff_map[i] / ratio[i])
        return HpxMap(diff_map, hpx_map.hpx)

    @staticmethod
    def _differential_to_integral(hpx_map):
        """ Convert a differential map to an integral map

        Here we are using log-log-quadrature to compute the integral quantities.
        """
        ebins = hpx_map.hpx.ebins
        ratio = ebins[1:] / ebins[0:-1]
        half_log_ratio = np.log(ratio) / 2.
        int_map = ((hpx_map.data[0:-1].T * ebins[0:-1]) +
                   (hpx_map.data[1:].T * ebins[1:])) * half_log_ratio
        return HpxMap(int_map.T, hpx_map.hpx)

    def run(self, argv):
        """Run this analysis"""
        args = self.parser.parse_args(argv)

        # Read the input maps
        ccube_dirty = HpxMap.create_from_fits(args.ccube_dirty, hdu='SKYMAP')
        bexpcube_dirty = HpxMap.create_from_fits(args.bexpcube_dirty, hdu='HPXEXPOSURES')
        ccube_clean = HpxMap.create_from_fits(args.ccube_clean, hdu='SKYMAP')
        bexpcube_clean = HpxMap.create_from_fits(args.bexpcube_clean, hdu='HPXEXPOSURES')

        # Decide what HEALPix order to work at
        if args.hpx_order:
            hpx_order = args.hpx_order
        else:
            hpx_order = ccube_dirty.hpx.order

        # Cast all the input maps to match ccube_clean
        cube_dict = ResidualCRAnalysis._match_cubes(ccube_clean, ccube_dirty,
                                                    bexpcube_clean, bexpcube_dirty, hpx_order)

        # Intenstiy maps
        intensity_clean = ResidualCRAnalysis._compute_intensity(cube_dict['ccube_clean'],
                                                                cube_dict['bexpcube_clean'])
        intensity_dirty = ResidualCRAnalysis._compute_intensity(cube_dict['ccube_dirty'],
                                                                cube_dict['bexpcube_dirty'])
        # Mean & ratio of intensity maps
        intensity_mean = ResidualCRAnalysis._compute_mean(intensity_dirty,
                                                          intensity_clean)
        intensity_ratio = ResidualCRAnalysis._compute_ratio(intensity_dirty,
                                                            intensity_clean)
        # Selecting the bright pixels for Aeff correction and to mask when filling output map
        bright_pixel_select = ResidualCRAnalysis._make_bright_pixel_mask(intensity_mean,
                                                                         args.select_factor)
        bright_pixel_mask = ResidualCRAnalysis._make_bright_pixel_mask(intensity_mean,
                                                                       args.mask_factor)
        # Compute thte Aeff corrections using the brightest pixels
        aeff_corrections = ResidualCRAnalysis._get_aeff_corrections(intensity_ratio,
                                                                    bright_pixel_select)
        # Apply the Aeff corrections and get the intensity residual
        corrected_dirty = ResidualCRAnalysis._apply_aeff_corrections(intensity_dirty,
                                                                     aeff_corrections)
        corrected_ratio = ResidualCRAnalysis._compute_ratio(corrected_dirty,
                                                            intensity_clean)
        intensity_resid = ResidualCRAnalysis._compute_diff(corrected_dirty,
                                                           intensity_clean)
        # Replace the masked pixels with the map mean to avoid features associates with sources
        filled_resid = ResidualCRAnalysis._fill_masked_intensity_resid(intensity_resid,
                                                                       bright_pixel_mask)
        # Smooth the map
        smooth_resid = ResidualCRAnalysis._smooth_hpx_map(filled_resid,
                                                          args.sigma)
        # Convert to a differential map
        out_model = ResidualCRAnalysis._intergral_to_differential(smooth_resid)

        # Make the ENERGIES HDU
        out_energies = ccube_dirty.hpx.make_energies_hdu()

        # Write the maps
        cubes = dict(SKYMAP=out_model)
        fits_utils.write_maps(None, cubes,
                              args.outfile, energy_hdu=out_energies)

        if args.full_output:
            # Some diagnostics
            check = ResidualCRAnalysis._differential_to_integral(out_model)
            check_resid = ResidualCRAnalysis._compute_diff(smooth_resid, check)
            counts_resid =\
                ResidualCRAnalysis._compute_counts_from_intensity(intensity_resid,
                                                                  cube_dict['bexpcube_dirty'])
            pred_counts\
                = ResidualCRAnalysis._compute_counts_from_model(out_model,
                                                                cube_dict['bexpcube_dirty'])
            pred_resid = ResidualCRAnalysis._compute_diff(pred_counts, counts_resid)

            out_ebounds = ccube_dirty.hpx.make_energy_bounds_hdu()
            cubes = dict(INTENSITY_CLEAN=intensity_clean,
                         INTENSITY_DIRTY=intensity_dirty,
                         INTENSITY_RATIO=intensity_ratio,
                         CORRECTED_DIRTY=corrected_dirty,
                         CORRECTED_RATIO=corrected_ratio,
                         INTENSITY_RESID=intensity_resid,
                         PIXEL_SELECT=bright_pixel_select,
                         PIXEL_MASK=bright_pixel_mask,
                         FILLED_RESID=filled_resid,
                         SMOOTH_RESID=smooth_resid,
                         CHECK=check,
                         CHECK_RESID=check_resid,
                         COUNTS_RESID=counts_resid,
                         PRED_COUNTS=pred_counts,
                         PRED_RESID=pred_resid)

            fits_utils.write_maps(None, cubes,
                                  args.outfile.replace('.fits', '_full.fits'),
                                  energy_hdu=out_ebounds)


class ConfigMaker_ResidualCR(ConfigMaker):
    """Small class to generate configurations for this script
    """
    default_options = dict(comp=diffuse_defaults.residual_cr['binning_yaml'],
                           data_dirty=diffuse_defaults.residual_cr['dataset_clean_yaml'],
                           data_clean=diffuse_defaults.residual_cr['dataset_dirty_yaml'],
                           irf_ver=diffuse_defaults.residual_cr['irf_ver'],
                           hpx_order=diffuse_defaults.residual_cr['hpx_order_fitting'],
                           coordsys=diffuse_defaults.residual_cr['coordsys'],
                           select_factor=(5.0, 'Pixel selection factor for Aeff Correction',
                                          float),
                           mask_factor=(2.0, 'Pixel selection factor for output mask',
                                        float),
                           sigma=(3.0, 'Width of gaussian to smooth output maps [degrees]', float),
                           full_output=(False, 'Include diagnostic output', bool))

    def __init__(self, link, **kwargs):
        """C'tor
        """
        ConfigMaker.__init__(self, link,
                             options=kwargs.get('options', self.default_options.copy()))

    def build_job_configs(self, args):
        """Hook to build job configurations
        """
        input_config = {}
        job_configs = {}

        components = Component.build_from_yamlfile(args['comp'])
        NAME_FACTORY_DIRTY.update_base_dict(args['data_dirty'])
        NAME_FACTORY_CLEAN.update_base_dict(args['data_clean'])

        for comp in components:
            zcut = "zmax%i" % comp.zmax
            key = comp.make_key('{ebin_name}_{evtype_name}')
            name_keys = dict(zcut=zcut,
                             ebin=comp.ebin_name,
                             psftype=comp.evtype_name,
                             coordsys=args['coordsys'],
                             irf_ver=args['irf_ver'],
                             fullpath=True)
            outfile = NAME_FACTORY_DIRTY.residual_cr(**name_keys)
            if args['hpx_order']:
                hpx_order = min(comp.hpx_order, args['hpx_order'])
            else:
                hpx_order = comp.hpx_order
            job_configs[key] = dict(bexpcube_dirty=NAME_FACTORY_DIRTY.bexpcube(**name_keys),
                                    ccube_dirty=NAME_FACTORY_DIRTY.ccube(
                                        **name_keys).replace('.fits', '.fits.gz'),
                                    bexpcube_clean=NAME_FACTORY_CLEAN.bexpcube(**name_keys),
                                    ccube_clean=NAME_FACTORY_CLEAN.ccube(
                                        **name_keys).replace('.fits', '.fits.gz'),
                                    outfile=outfile,
                                    hpx_order=hpx_order,
                                    logfile=outfile.replace('.fits', '.log'))

        output_config = {}
        return input_config, job_configs, output_config

def create_link_residual_cr(**kwargs):
    """Build and return a `Link` object that can invoke `ResidualCRAnalysis` """
    analyzer = ResidualCRAnalysis(**kwargs)
    return analyzer.link

def create_sg_residual_cr(**kwargs):
    """Build and return a ScatterGather object that can invoke this script"""
    analyzer = ResidualCRAnalysis()
    link = analyzer.link
    link.linkname = kwargs.pop('linkname', link.linkname)
    appname = kwargs.pop('appname', 'gt-residual-cr-sg')

    lsf_args = {'W': 1500,
                'R': 'rhel60'}

    usage = "%s [options]"%(appname)
    description = "Copy source maps from the library to a analysis directory"

    config_maker = ConfigMaker_ResidualCR(link)
    lsf_sg = build_sg_from_link(link, config_maker,
                                lsf_args=lsf_args,
                                usage=usage,
                                description=description,
                                linkname=link.linkname,
                                appname=appname,
                                **kwargs)
    return lsf_sg



class ResidualCRChain(Chain):
    """Small class to preform analysis of residual cosmic-ray contamination
    """
    def __init__(self, linkname):
        """C'tor
        """
        link_sb_clean = create_sg_split_and_bin(linkname="%s.sb_clean"%linkname,
                                                mapping={'data':'dataset_clean_yaml',
                                                         'hpx_order':'hpx_order_binning',
                                                         'inputlist':'ft1file',
                                                         'comp':'binning_yaml'})
        link_sb_dirty = create_sg_split_and_bin(linkname="%s.sb_dirty"%linkname,
                                                mapping={'data':'dataset_dirty_yaml',
                                                         'hpx_order':'hpx_order_binning',
                                                         'inputlist':'ft1file',
                                                         'comp':'binning_yaml'})
        link_excube_clean = create_sg_gtexpcube2(linkname="%s.expcube_clean"%linkname,
                                                 mapping={'cmap':'ccube_clean',
                                                          'outfile':'bexpcube_clean',
                                                          'data':'dataset_clean_yaml',
                                                          'hpx_order':'hpx_order_binning',
                                                          'comp':'binning_yaml'})
        link_excube_dirty = create_sg_gtexpcube2(linkname="%s.expcube_dirty"%linkname,
                                                 mapping={'cmap':'ccube_dirty',
                                                          'outfile':'bexpcube_dirty',
                                                          'data':'dataset_dirty_yaml',
                                                          'hpx_order':'hpx_order_binning',
                                                          'comp':'binning_yaml'})
        link_cr_analysis = create_sg_residual_cr(linkname="%s.cr_analysis"%linkname,
                                                 mapping={'data_dirty':'dataset_dirty_yaml',
                                                          'data_clean':'dataset_clean_yaml',
                                                          'hpx_order':'hpx_order_fitting',
                                                          'comp':'binning_yaml'})

        options = diffuse_defaults.residual_cr.copy()
        options['dry_run'] = (False, 'Print commands but do not run', bool)
        Chain.__init__(self, linkname,
                       appname='FIXME',
                       links=[link_sb_clean, link_sb_dirty,
                              link_excube_clean, link_excube_dirty,
                              link_cr_analysis],
                       options=options,
                       argmapper=self._map_arguments,
                       parser=ResidualCRChain._make_parser())

    @staticmethod
    def _make_parser():
        """Make an argument parser for this chain """
        usage = "FIXME [options]"
        description = "Run residual cosmic-ray analysis"

        parser = argparse.ArgumentParser(usage=usage, description=description)
        return parser

    def _map_arguments(self, input_dict):
        """Map from the top-level arguments to the arguments provided to
        the indiviudal links """
        output_dict = input_dict.copy()
        data_clean = input_dict['dataset_clean_yaml']
        data_dirty = input_dict['dataset_dirty_yaml']
        if data_clean is not None:
            NAME_FACTORY_CLEAN.update_base_dict(input_dict['dataset_clean_yaml'])
            output_dict['bexpcube_clean'] = NAME_FACTORY_CLEAN.bexpcube()
            output_dict['ccube_clean'] = NAME_FACTORY_CLEAN.ccube()
        if data_dirty is not None:
            NAME_FACTORY_DIRTY.update_base_dict(input_dict['dataset_dirty_yaml'])
            output_dict['bexpcube_dirty'] = NAME_FACTORY_DIRTY.bexpcube()
            output_dict['ccube_dirty'] = NAME_FACTORY_DIRTY.ccube()
        return output_dict

    def run_argparser(self, argv):
        """Initialize a link with a set of arguments using argparser
        """
        args = Link.run_argparser(self, argv)
        for link in self._links.values():
            link.run_link(stream=sys.stdout, dry_run=True)
        return args


def create_chain_residual_cr(**kwargs):
    """Build and return a `ResidualCRChain` object """
    ret_chain = ResidualCRChain(linkname=kwargs.pop('linkname', 'ResidualCR'))
    return ret_chain


def main_single():
    """Entry point for command line use for single job """
    gtsmp = ResidualCRAnalysis()
    gtsmp.run(sys.argv[1:])


def main_batch():
    """Entry point for command line use for dispatching batch jobs """
    lsf_sg = create_sg_residual_cr()
    lsf_sg(sys.argv)

def main_chain():
    """Energy point for running the entire Cosmic-ray analysis """
    the_chain = ResidualCRChain('ResidualCR')
    args = the_chain.run_argparser(sys.argv[1:])
    the_chain.run_chain(sys.stdout, args.dry_run)
    the_chain.finalize(args.dry_run)


if __name__ == '__main__':
    main_chain()


