# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import copy
import shutil
import collections
import logging
import tempfile
import filecmp
import time
import json
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column, vstack
import fermipy
import fermipy.defaults as defaults
import fermipy.utils as utils
import fermipy.wcs_utils as wcs_utils
import fermipy.fits_utils as fits_utils
import fermipy.gtutils as gtutils
import fermipy.srcmap_utils as srcmap_utils
import fermipy.skymap as skymap
import fermipy.plotting as plotting
import fermipy.irfs as irfs
import fermipy.sed as sed
import fermipy.lightcurve as lightcurve
from fermipy.residmap import ResidMapGenerator
from fermipy.tsmap import TSMapGenerator, TSCubeGenerator
from fermipy.sourcefind import SourceFind
from fermipy.extension import ExtensionFit
from fermipy.utils import merge_dict
from fermipy.utils import create_hpx_disk_region_string
from fermipy.utils import resolve_file_path
from fermipy.skymap import Map, HpxMap
from fermipy.hpx_utils import HPX
from fermipy.roi_model import ROIModel
from fermipy.ltcube import LTCube
from fermipy.plotting import AnalysisPlotter
from fermipy.logger import Logger, log_level
from fermipy.config import ConfigSchema
from fermipy.docstring_utils import DocstringMeta
from fermipy.fitcache import FitCache
# pylikelihood
import GtApp
import FluxDensity
from LikelihoodState import LikelihoodState
from fermipy.gtutils import BinnedAnalysis, SummedLikelihood
import BinnedAnalysis as ba
import pyLikelihood as pyLike

norm_parameters = {
    'ConstantValue': ['Value'],
    'PowerLaw': ['Prefactor'],
    'PowerLaw2': ['Integral'],
    'BrokenPowerLaw': ['Prefactor'],
    'SmoothBrokenPowerLaw': ['Prefactor'],
    'LogParabola': ['norm'],
    'PLSuperExpCutoff': ['Prefactor'],
    'ExpCutoff': ['Prefactor'],
    'FileFunction': ['Normalization'],
    'DMFitFunction': ['sigmav'],
    'Gaussian': ['Prefactor'],
}

shape_parameters = {
    'ConstantValue': [],
    'PowerLaw': ['Index'],
    'PowerLaw2': ['Index'],
    'BrokenPowerLaw': ['Index1', 'Index2'],
    'SmoothBrokenPowerLaw': ['Index1', 'Index2'],
    'LogParabola': ['alpha', 'beta'],
    'PLSuperExpCutoff': ['Index1', 'Cutoff'],
    'ExpCutoff': ['Index1', 'Cutoff'],
    'FileFunction': [],
    'DMFitFunction': ['mass'],
    'Gaussian': ['Mean', 'Sigma'],
}

index_parameters = {
    'ConstantValue': [],
    'PowerLaw': ['Index'],
    'PowerLaw2': ['Index'],
    'BrokenPowerLaw': ['Index1', 'Index2'],
    'SmoothBrokenPowerLaw': ['Index1', 'Index2'],
    'LogParabola': ['alpha', 'beta'],
    'PLSuperExpCutoff': ['Index1', 'Index2'],
    'ExpCutoff': ['Index1'],
    'FileFunction': [],
    'DMFitFunction': [],
    'Gaussian': [],
}


def create_sc_table(scfile, colnames=None):
    """Load an FT2 file from a file or list of files."""

    if utils.is_fits_file(scfile) and colnames is None:
        return create_table_from_fits(scfile, 'SC_DATA')

    if utils.is_fits_file(scfile):
        files = [scfile]
    else:
        files = [line.strip() for line in open(scfile, 'r')]

    tables = [create_table_from_fits(f, 'SC_DATA', colnames)
              for f in files]

    return vstack(tables)


def create_table_from_fits(fitsfile, hduname, colnames=None):
    """Memory efficient function for loading a table from a FITS
    file."""

    if colnames is None:
        return Table.read(fitsfile, hduname)

    h = fits.open(fitsfile, memmap=True)
    cols = []
    for k in colnames:
        data = h[hduname].data.field(k)
        cols += [Column(name=k, data=data)]
    return Table(cols)


def get_spectral_index(src, egy):
    """Compute the local spectral index of a source."""
    delta = 1E-5
    f0 = src.spectrum()(pyLike.dArg(egy * (1 - delta)))
    f1 = src.spectrum()(pyLike.dArg(egy * (1 + delta)))

    if f0 > 0 and f1 > 0:
        gamma = np.log10(f0 / f1) / np.log10((1 - delta) / (1 + delta))
    else:
        gamma = np.nan

    return gamma


def run_gtapp(appname, logger, kw, **kwargs):

    loglevel = kwargs.get('loglevel', logging.INFO)

    logger.log(loglevel, 'Running %s.', appname)
    t0 = time.time()
    filter_dict(kw, None)
    kw = utils.unicode_to_str(kw)
    gtapp = GtApp.GtApp(str(appname))

    for k, v in kw.items():

        if (appname == 'gtbin' and k == 'scfile' and
            not v.startswith('@') and
                not utils.is_fits_file(v)):
            v = '@' + v
        gtapp[k] = v

    logger.log(loglevel, gtapp.command())
    stdin, stdout = gtapp.runWithOutput(print_command=False)

    for line in stdout:
        logger.log(loglevel, line.strip())

        # Capture return code?

    t1 = time.time()
    logger.log(loglevel, 'Finished %s. Execution time: %.2f s',
               appname, t1 - t0)


def filter_dict(d, val):
    for k, v in d.items():
        if v == val:
            del d[k]


class GTAnalysis(fermipy.config.Configurable, sed.SEDGenerator,
                 ResidMapGenerator, TSMapGenerator, TSCubeGenerator,
                 SourceFind, ExtensionFit, lightcurve.LightCurve):
    """High-level analysis interface that manages a set of analysis
    component objects.  Most of the functionality of the Fermipy
    package is provided through the methods of this class.  The class
    constructor accepts a dictionary that defines the configuration
    for the analysis.  Keyword arguments to the constructor can be
    used to override parameters in the configuration dictionary.
    """

    __metaclass__ = DocstringMeta

    _docstring_registry = {
        'extension': defaults.extension,
        'sed': defaults.sed,
        'localize': defaults.localize,
        'tsmap': defaults.tsmap,
        'residmap': defaults.residmap,
        'lightcurve': defaults.lightcurve,
        'find_sources': defaults.sourcefind,
    }

    defaults = {'logging': defaults.logging,
                'fileio': defaults.fileio,
                'optimizer': defaults.optimizer,
                'binning': defaults.binning,
                'selection': defaults.selection,
                'model': defaults.model,
                'data': defaults.data,
                'ltcube': defaults.ltcube,
                'gtlike': defaults.gtlike,
                'mc': defaults.mc,
                'residmap': defaults.residmap,
                'tsmap': defaults.tsmap,
                'tscube': defaults.tscube,
                'sourcefind': defaults.sourcefind,
                'sed': defaults.sed,
                'lightcurve': defaults.lightcurve,
                'extension': defaults.extension,
                'localize': defaults.localize,
                'roiopt': defaults.roiopt,
                'plotting': defaults.plotting,
                'components': (None, '', list)}

    def __init__(self, config, **kwargs):

        # Setup directories
        self._rootdir = os.getcwd()
        self._outdir = None
        validate = kwargs.pop('validate', True)
        self._loglevel = kwargs.pop('loglevel', logging.INFO)

        super(GTAnalysis, self).__init__(config, validate=validate,
                                         **kwargs)

        self._projtype = self.config['binning']['projtype']
        self._tmin = self.config['selection']['tmin']
        self._tmax = self.config['selection']['tmax']

        # Set random seed
        np.random.seed(self.config['mc']['seed'])

        # Destination directory for output data products
        if self.config['fileio']['outdir'] is not None:
            self._outdir = os.path.join(self._rootdir,
                                        self.config['fileio']['outdir'])
            utils.mkdir(self._outdir)
        else:
            raise Exception('Save directory not defined.')

        if self.config['fileio']['logfile'] is None:
            self.config['fileio']['logfile'] = os.path.join(self.outdir,
                                                            'fermipy')

        self.logger = Logger.get(self.__class__.__name__,
                                 self.config['fileio']['logfile'],
                                 log_level(self.config['logging']
                                           ['verbosity']))

        self.logger.log(self.loglevel, '\n' + '-' * 80 + '\n' +
                        "fermipy version {} ".
                        format(fermipy.__version__) + '\n' +
                        'ScienceTools version %s', fermipy.get_st_version())
        self.print_config(self.logger, loglevel=logging.DEBUG)

        # Working directory (can be the same as savedir)
        if self.config['fileio']['usescratch']:
            self.config['fileio']['workdir'] = tempfile.mkdtemp(
                prefix=os.environ['USER'] + '.',
                dir=self.config['fileio']['scratchdir'])
            self.logger.info('Created working directory: %s',
                             self.config['fileio']['workdir'])
        else:
            self._config['fileio']['workdir'] = self._outdir

        if 'FERMIPY_WORKDIR' not in os.environ:
            os.environ['FERMIPY_WORKDIR'] = self.config['fileio']['workdir']

        # put pfiles into savedir
        os.environ['PFILES'] = \
            self.workdir + ';' + os.environ['PFILES'].split(';')[-1]

        # Create Plotter
        self._plotter = AnalysisPlotter(self.config['plotting'],
                                        fileio=self.config['fileio'],
                                        logging=self.config['logging'])

        # Setup the ROI definition
        self._roi = ROIModel.create(self.config['selection'],
                                    self.config['model'],
                                    fileio=self.config['fileio'],
                                    coordsys=self.config['binning']['coordsys'])

        self._like = None
        self._components = []
        configs = self._create_component_configs()

        for cfg in configs:
            comp = self._create_component(cfg, loglevel=self.loglevel)
            self._components.append(comp)

        for c in self.components:
            for s in c.roi.sources:
                if s.name not in self.roi:
                    self.roi.load_source(s)

        self._files = {}
        self._files['ccube'] = os.path.join(self.workdir, 'ccube.fits')

        ebin_edges = np.zeros(0)
        roiwidths = np.zeros(0)
        binsz = np.zeros(0)
        for c in self.components:
            ebin_edges = np.concatenate((ebin_edges, c.log_energies))
            roiwidths = np.insert(roiwidths, 0, c.roiwidth)
            binsz = np.insert(binsz, 0, c.binsz)

        self._ebin_edges = np.sort(np.unique(ebin_edges.round(5)))
        self._enumbins = len(self._ebin_edges) - 1
        self._loge_bounds = np.array([self._ebin_edges[0],
                                      self._ebin_edges[-1]])

        self._roi_data = {
            'loglike': np.nan,
            'npred': 0.0,
            'counts': np.zeros(self.enumbins),
            'model_counts': np.zeros(self.enumbins),
            'energies': np.copy(self.energies),
            'log_energies': np.copy(self.log_energies),
            'loge_bounds': np.copy(self.loge_bounds),
            'components': []
        }

        for c in self._components:
            comp_model = [{'loglike': np.nan,
                           'npred': 0.0,
                           'counts': np.zeros(c.enumbins),
                           'model_counts': np.zeros(c.enumbins),
                           'energies': np.copy(c.energies),
                           'log_energies': np.copy(c.log_energies),
                           'src_expscale': copy.deepcopy(c.src_expscale),
                           }]

            self._roi_data['components'] += comp_model

        self._roiwidth = max(roiwidths)
        self._binsz = min(binsz)
        self._npix = int(np.round(self._roiwidth / self._binsz))

        if self.projtype == 'HPX':
            self._hpx_region = create_hpx_disk_region_string(self._roi.skydir,
                                                             coordsys=self.config[
                                                                 'binning'][
                                                                 'coordsys'],
                                                             radius=0.5 *
                                                             self.config[
                                                                 'binning'][
                                                                 'roiwidth'])
            self._proj = HPX.create_hpx(-1,
                                        self.config['binning'][
                                            'hpx_ordering_scheme'] == "NESTED",
                                        self.config['binning']['coordsys'],
                                        self.config['binning']['hpx_order'],
                                        self._hpx_region,
                                        self.energies)

        else:
            self._skywcs = wcs_utils.create_wcs(self._roi.skydir,
                                                coordsys=self.config['binning'][
                                                    'coordsys'],
                                                projection=self.config[
                                                    'binning']['proj'],
                                                cdelt=self._binsz,
                                                crpix=1.0 + 0.5 *
                                                (self._npix - 1),
                                                naxis=2)

            self._proj = wcs_utils.create_wcs(self._roi.skydir,
                                              coordsys=self.config[
                                                  'binning']['coordsys'],
                                              projection=self.config[
                                                  'binning']['proj'],
                                              cdelt=self._binsz,
                                              crpix=1.0 + 0.5 *
                                              (self._npix - 1),
                                              naxis=3,
                                              energies=self.energies)

            # Update projection of ROI object
            proj = wcs_utils.WCSProj(self._proj,
                                     np.array([self.npix, self.npix]))
            self._roi.set_projection(proj)

        if self.config['fileio']['usescratch']:
            self.stage_input()

    def __del__(self):
        self.stage_output()
        self.cleanup()

    @property
    def loglevel(self):
        """Return the default loglevel."""
        return self._loglevel

    @property
    def workdir(self):
        """Return the analysis working directory."""
        return self.config['fileio']['workdir']

    @property
    def outdir(self):
        """Return the analysis output directory."""
        return self._outdir

    @property
    def roi(self):
        """Return the ROI object."""
        return self._roi

    @property
    def plotter(self):
        """Return the plotter instance."""
        return self._plotter

    @property
    def like(self):
        """Return the global likelihood object."""
        return self._like

    @property
    def components(self):
        """Return the list of analysis components."""
        return self._components

    @property
    def energies(self):
        """Return the energy bin edges in MeV."""
        return 10**self._ebin_edges

    @property
    def log_energies(self):
        """Return the energy bin edges in log10(E/MeV)."""
        return self._ebin_edges

    @property
    def enumbins(self):
        """Return the number of energy bins."""
        return self._enumbins

    @property
    def npix(self):
        """Return the number of energy bins."""
        return self._npix

    @property
    def loge_bounds(self):
        """Current analysis energy bounds in log10(E/MeV)."""
        return self._loge_bounds

    @property
    def projtype(self):
        """Return the type of projection to use"""
        return self._projtype

    @property
    def tmin(self):
        """Return the MET time for the start of the observation."""
        return self._tmin

    @property
    def tmax(self):
        """Return the MET time for the end of the observation."""
        return self._tmax

    @property
    def files(self):
        return self._files

    @staticmethod
    def create(infile, config=None):
        """Create a new instance of GTAnalysis from an analysis output file
        generated with `~fermipy.GTAnalysis.write_roi`.  By default
        the new instance will inherit the configuration of the saved
        analysis instance.  The configuration may be overriden by
        passing a configuration file path with the ``config``
        argument.

        Parameters
        ----------

        infile : str
            Path to the ROI results file.

        config : str
            Path to a configuration file.  This will override the
            configuration in the ROI results file.

        """

        infile = os.path.abspath(infile)
        roi_file, roi_data = utils.load_data(infile)

        if config is None:
            config = roi_data['config']
            validate = False
        else:
            validate = True

        gta = GTAnalysis(config, validate=validate)
        gta.setup(init_sources=False)
        gta.load_roi(infile)
        return gta

    def clone(self, config, **kwargs):
        """Make a clone of this analysis instance."""
        gta = GTAnalysis(config, **kwargs)
        gta._roi = copy.deepcopy(self.roi)
        for i, c in enumerate(self.components):
            gta.components[i]._roi = copy.deepcopy(c.roi)

        return gta

    def set_log_level(self, level):
        self.logger.handlers[1].setLevel(level)
        for c in self.components:
            c.logger.handlers[1].setLevel(level)

    def _update_roi(self):

        rm = self._roi_data

        rm['loglike'] = -self.like()
        rm['model_counts'].fill(0)
        rm['npred'] = 0
        for i, c in enumerate(self.components):
            rm['components'][i]['loglike'] = -c.like()
            rm['components'][i]['model_counts'].fill(0)
            rm['components'][i]['npred'] = 0

        for name in self.like.sourceNames():

            # EAC, this is one of the only things we need from setup()
            # self.update_source(name)

            src = self.roi.get_source_by_name(name)
            rm['model_counts'] += src['model_counts']
            rm['npred'] += np.sum(src['model_counts'])
            mc = self.model_counts_spectrum(name)

            for i, c in enumerate(self.components):
                rm['components'][i]['model_counts'] += mc[i]
                rm['components'][i]['npred'] += np.sum(mc[i])

    def _update_srcmap(self, name, src, **kwargs):

        for c in self.components:
            c._update_srcmap(name, src, **kwargs)

        if self._fitcache is not None:
            self._fitcache.update_source(name)

    def _create_srcmap_cache(self, name, src):
        for c in self.components:
            c._create_srcmap_cache(name, src)

    def _clear_srcmap_cache(self):
        for c in self.components:
            c._srcmap_cache.clear()

    def reload_source(self, name, init_source=True):
        """Delete and reload a source in the model.  This will update
        the spatial model of this source to the one defined in the XML
        model."""

        for c in self.components:
            c.reload_source(name)

        if init_source:
            self._init_source(name)

        self.like.model = self.like.components[0].model

    def reload_sources(self, names, init_source=True):

        for c in self.components:
            c.reload_sources(names)

        if init_source:
            for name in names:
                self._init_source(name)

        self.like.model = self.like.components[0].model

    def set_source_morphology(self, name, **kwargs):
        """Set the spatial model of a source.

        Parameters
        ----------
        name : str
           Source name.

        spatial_model : str
           Spatial model name (PointSource, RadialGaussian, etc.).

        spatial_pars : dict
           Dictionary of spatial parameters (optional).

        use_cache : bool        
           Generate the spatial model by interpolating the cached source
           map.

        use_pylike : bool

        """

        name = self.roi.get_source_by_name(name).name
        src = self.roi[name]

        spatial_model = kwargs.get('spatial_model', src['SpatialModel'])
        spatial_pars = kwargs.get('spatial_pars', {})
        use_pylike = kwargs.get('use_pylike', True)
        psf_scale_fn = kwargs.get('psf_scale_fn', None)
        update_source = kwargs.get('update_source', False)

        if hasattr(pyLike.BinnedLikelihood, 'setSourceMapImage') and not use_pylike:
            src.set_spatial_model(spatial_model, spatial_pars)
            self._update_srcmap(src.name, src, psf_scale_fn=psf_scale_fn)
        else:
            src = self.delete_source(name, loglevel=logging.DEBUG,
                                     save_template=False)
            src.set_spatial_model(spatial_model, spatial_pars)
            self.add_source(src.name, src, init_source=False,
                            use_pylike=use_pylike, loglevel=logging.DEBUG)

        if update_source:
            self.update_source(name)

    def set_source_spectrum(self, name, spectrum_type='PowerLaw',
                            spectrum_pars=None, update_source=True):
        """Set the spectral model of a source.  This function can be
        used to change the spectral type of a source or modify its
        spectral parameters.  If called with
        spectrum_type='FileFunction' and spectrum_pars=None, the
        source spectrum will be replaced with a FileFunction with the
        same differential flux distribution as the original spectrum.

        Parameters
        ----------

        name : str
           Source name.

        spectrum_type : str
           Spectrum type (PowerLaw, etc.).

        spectrum_pars : dict
           Dictionary of spectral parameters (optional).

        update_source : bool
           Recompute all source characteristics (flux, TS, NPred)
           using the new spectral model of the source.

        """
        name = self.roi.get_source_by_name(name).name

        if spectrum_type == 'FileFunction':
            self._create_filefunction(name, spectrum_pars)
        else:
            fn = gtutils.create_spectrum_from_dict(spectrum_type,
                                                   spectrum_pars)
            self.like.setSpectrum(str(name), fn)

        # Get parameters
        src = self.components[0].like.logLike.getSource(str(name))
        pars_dict = gtutils.get_function_pars_dict(src.spectrum())

        self.roi[name]['SpectrumType'] = spectrum_type
        self.roi[name].set_spectral_pars(pars_dict)
        for c in self.components:
            c.roi[name]['SpectrumType'] = spectrum_type
            c.roi[name].set_spectral_pars(pars_dict)

        if update_source:
            self.update_source(name)

    def set_source_dnde(self, name, dnde, update_source=True):
        """Set the differential flux distribution of a source with the
        FileFunction spectral type.

        Parameters
        ----------
        name : str
           Source name.

        dnde : `~numpy.ndarray`
           Array of differential flux values (cm^{-2} s^{-1} MeV^{-1}).
        """
        name = self.roi.get_source_by_name(name).name

        if self.roi[name]['SpectrumType'] != 'FileFunction':
            msg = 'Wrong spectral type: %s' % self.roi[name]['SpectrumType']
            self.logger.error(msg)
            raise Exception(msg)

        xy = self.get_source_dnde(name)

        if len(dnde) != len(xy[0]):
            msg = 'Wrong length for dnde array: %i' % len(dnde)
            self.logger.error(msg)
            raise Exception(msg)

        for c in self.components:
            src = c.like.logLike.getSource(str(name))
            spectrum = src.spectrum()
            file_function = pyLike.FileFunction_cast(spectrum)
            file_function.setSpectrum(10**xy[0], dnde)

        if update_source:
            self.update_source(name)

    def get_source_dnde(self, name):
        """Return differential flux distribution of a source.  For
        sources with FileFunction spectral type this returns the
        internal differential flux array.

        Returns
        -------
        loge : `~numpy.ndarray`
           Array of energies at which the differential flux is
           evaluated (log10(E/MeV)).

        dnde : `~numpy.ndarray`
           Array of differential flux values (cm^{-2} s^{-1} MeV^{-1})
           evaluated at energies in ``loge``.

        """
        name = self.roi.get_source_by_name(name).name

        if self.roi[name]['SpectrumType'] != 'FileFunction':

            src = self.components[0].like.logLike.getSource(str(name))
            spectrum = src.spectrum()
            file_function = pyLike.FileFunction_cast(spectrum)
            loge = file_function.log_energy()
            logdnde = file_function.log_dnde()

            loge = np.log10(np.exp(loge))
            dnde = np.exp(logdnde)

            return loge, dnde

        else:
            ebinsz = (self.log_energies[-1] -
                      self.log_energies[0]) / self.enumbins
            loge = utils.extend_array(self.log_energies, ebinsz, 0.5, 6.5)

            dnde = np.array([self.like[name].spectrum()(pyLike.dArg(10 ** egy))
                             for egy in loge])

            return loge, dnde

    def _create_filefunction(self, name, spectrum_pars):
        """Replace the spectrum of an existing source with a
        FileFunction."""

        spectrum_pars = {} if spectrum_pars is None else spectrum_pars

        if 'loge' in spectrum_pars:
            loge = spectrum_pars.get('loge')
        else:
            ebinsz = (self.log_energies[-1] -
                      self.log_energies[0]) / self.enumbins
            loge = utils.extend_array(self.log_energies, ebinsz, 0.5, 6.5)

        # Get the values
        dnde = np.zeros(len(loge))
        if 'dnde' in spectrum_pars:
            dnde = spectrum_pars.get('dnde')
        else:
            dnde = np.array([self.like[name].spectrum()(pyLike.dArg(10 ** egy))
                             for egy in loge])

        filename = \
            os.path.join(self.workdir,
                         '%s_filespectrum.txt' % (name.lower().replace(' ', '_')))

        # Create file spectrum txt file
        np.savetxt(filename, np.vstack((10**loge, dnde)).T)
        self.like.setSpectrum(name, str('FileFunction'))

        self.roi[name]['Spectrum_Filename'] = filename
        # Update
        for c in self.components:
            src = c.like.logLike.getSource(str(name))
            spectrum = src.spectrum()

            spectrum.getParam(str('Normalization')).setBounds(1E-3, 1E3)

            file_function = pyLike.FileFunction_cast(spectrum)
            file_function.readFunction(str(filename))
            c.roi[name]['Spectrum_Filename'] = filename

    def _create_component_configs(self):
        configs = []

        components = self.config['components']

        common_config = GTBinnedAnalysis.get_config()
        common_config = merge_dict(common_config, self.config,
                                   add_new_keys=True)

        if components is None or len(components) == 0:
            cfg = copy.copy(common_config)
            cfg['file_suffix'] = '_00'
            cfg['name'] = '00'
            configs.append(cfg)
        elif isinstance(components, dict):
            for i, k in enumerate(sorted(components.keys())):
                cfg = copy.copy(common_config)
                cfg = merge_dict(cfg, components[k], add_new_keys=True)
                cfg['file_suffix'] = '_' + k
                cfg['name'] = k
                configs.append(cfg)
        elif isinstance(components, list):
            for i, c in enumerate(components):
                cfg = copy.copy(common_config)
                cfg = merge_dict(cfg, c, add_new_keys=True)
                cfg['file_suffix'] = '_%02i' % i
                cfg['name'] = '%02i' % i
                configs.append(cfg)
        else:
            raise Exception('Invalid type for component block.')

        return configs

    def _create_component(self, cfg, **kwargs):

        self.logger.debug("Creating Analysis Component: " + cfg['name'])

        cfg['fileio']['workdir'] = self.config['fileio']['workdir']

        for k in cfg.keys():
            if not k in GTBinnedAnalysis.defaults:
                cfg.pop(k)

        comp = GTBinnedAnalysis(cfg, logging=self.config['logging'], **kwargs)

        return comp

    def stage_output(self):
        """Copy data products to final output directory."""

        if self.workdir == self.outdir:
            return
        elif not os.path.isdir(self.workdir):
            self.logger.error('Working directory does not exist.')
            return

        regex = self.config['fileio']['outdir_regex']
        savefits = self.config['fileio']['savefits']
        files = os.listdir(self.workdir)
        self.logger.info('Staging files to %s', self.outdir)

        fitsfiles = []
        for c in self.components:
            for f in c.files.values():

                if f is None:
                    continue

                fitsfiles += [os.path.basename(f)]

        for f in files:

            wpath = os.path.join(self.workdir, f)
            opath = os.path.join(self.outdir, f)

            if not utils.match_regex_list(regex, os.path.basename(f)):
                continue
            if os.path.isfile(opath) and filecmp.cmp(wpath, opath, False):
                continue
            if not savefits and f in fitsfiles:
                continue

            self.logger.debug('Copying ' + f)
            shutil.copy(wpath, self.outdir)

        self.logger.info('Finished.')

    def stage_input(self):
        """Copy input files to working directory."""

        if self.workdir == self.outdir:
            return
        elif not os.path.isdir(self.workdir):
            self.logger.error('Working directory does not exist.')
            return

        self.logger.info('Staging files to %s', self.workdir)

        files = [os.path.join(self.outdir, f)
                 for f in os.listdir(self.outdir)]

        regex = copy.deepcopy(self.config['fileio']['workdir_regex'])

        for f in files:

            if not os.path.isfile(f):
                continue
            if not utils.match_regex_list(regex, os.path.basename(f)):
                continue

            self.logger.debug('Copying ' + os.path.basename(f))
            shutil.copy(f, self.workdir)

        for c in self.components:
            for f in c.files.values():

                if f is None:
                    continue

                wpath = os.path.join(self.workdir, os.path.basename(f))
                opath = os.path.join(self.outdir, os.path.basename(f))

                if os.path.isfile(wpath):
                    continue
                elif os.path.isfile(opath):
                    self.logger.debug('Copying ' + os.path.basename(f))
                    shutil.copy(opath, self.workdir)

        self.logger.info('Finished.')

    def setup(self, init_sources=True, overwrite=False, **kwargs):
        """Run pre-processing for each analysis component and
        construct a joint likelihood object.  This function performs
        the following tasks: data selection (gtselect, gtmktime),
        data binning (gtbin), and model generation (gtexpcube2,gtsrcmaps).

        Parameters
        ----------

        init_sources : bool

           Choose whether to compute properties (flux, TS, etc.) for
           individual sources.

        overwrite : bool

           Run all pre-processing steps even if the output file of
           that step is present in the working directory.  By default
           this function will skip any steps for which the output file
           already exists.

        """

        loglevel = kwargs.get('loglevel', self.loglevel)

        self.logger.log(loglevel, 'Running setup.')

        # Run setup for each component
        for i, c in enumerate(self.components):
            c.setup(overwrite=overwrite)

        # Create likelihood
        self._create_likelihood()

        # Determine tmin, tmax
        for i, c in enumerate(self._components):
            self._tmin = (c.tmin if self._tmin is None
                          else min(self._tmin, c.tmin))
            self._tmax = (c.tmax if self._tmax is None
                          else min(self._tmax, c.tmax))

        if init_sources:

            self.logger.log(loglevel, 'Initializing source properties')
            for name in self.like.sourceNames():
                self.logger.debug('Initializing source %s', name)
                self._init_source(name)
            self._update_roi()

        self.logger.log(loglevel, 'Finished setup.')

    def _create_likelihood(self, srcmdl=None):
        """Instantiate the likelihood object for each component and
        create a SummedLikelihood."""

        self._like = SummedLikelihood()
        for c in self.components:
            c._create_binned_analysis(srcmdl)
            self._like.addComponent(c.like)

        self.like.model = self.like.components[0].model
        self._fitcache = None
        self._init_roi_model()

    def _init_roi_model(self):

        rm = self._roi_data

        rm['counts'] = np.zeros(self.enumbins)
        rm['loglike'] = -self.like()

        cmaps = []
        proj_type = 0
        for i, c in enumerate(self.components):
            cm = c.counts_map()
            cmaps += [cm]
            if isinstance(cm, Map):
                rm['components'][i]['counts'] = \
                    np.squeeze(
                        np.apply_over_axes(np.sum, cm.counts, axes=[1, 2]))
            elif isinstance(cm, HpxMap):
                proj_type = 1
                rm['components'][i]['counts'] = \
                    np.squeeze(np.apply_over_axes(np.sum, cm.counts, axes=[1]))
            rm['components'][i]['loglike'] = -c.like()

        if proj_type == 0:
            shape = (self.enumbins, self.npix, self.npix)
        elif proj_type == 1:
            shape = (self.enumbins, self._proj.npix)

        self._coadd_maps(cmaps, shape, rm)

    def _init_source(self, name):

        src = self.roi.get_source_by_name(name)
        src.update_data({'sed': None})
        sd = self.get_src_model(name, paramsonly=True)
        src.update_data(sd)

        for c in self.components:
            src = c.roi.get_source_by_name(name)
            src.update_data(sd)

        return src

    def cleanup(self):

        if self.workdir == self.outdir:
            return
        elif os.path.isdir(self.workdir):
            self.logger.info('Deleting working directory: ' +
                             self.workdir)
            shutil.rmtree(self.workdir)

    def generate_model(self, model_name=None):
        """Generate model maps for all components.  model_name should
        be a unique identifier for the model.  If model_name is None
        then the model maps will be generated using the current
        parameters of the ROI."""

        for i, c in enumerate(self._components):
            c.generate_model(model_name=model_name)

            # If all model maps have the same spatial/energy binning we
            # could generate a co-added model map here

    def set_energy_range(self, logemin, logemax):
        """Set the energy bounds of the analysis.  This restricts the
        evaluation of the likelihood to the data that falls in this
        range.  Input values will be rounded to the closest bin edge
        value.  If either argument is None then the lower or upper
        bound of the analysis instance will be used.

        Parameters
        ----------

        logemin : float
           Lower energy bound in log10(E/MeV).

        logemax : float
           Upper energy bound in log10(E/MeV).

        Returns
        -------

        eminmax : array
           Minimum and maximum energy in log10(E/MeV).

        """

        if logemin is None:
            logemin = self.log_energies[0]
        else:
            imin = int(utils.val_to_edge(self.log_energies, logemin)[0])
            logemin = self.log_energies[imin]

        if logemax is None:
            logemax = self.log_energies[-1]
        else:
            imax = int(utils.val_to_edge(self.log_energies, logemax)[0])
            logemax = self.log_energies[imax]

        self._loge_bounds = np.array([logemin, logemax])
        self._roi_data['loge_bounds'] = np.copy(self.loge_bounds)
        for c in self.components:
            c.set_energy_range(logemin, logemax)

        return self._loge_bounds

    def counts_map(self):
        """Return a `~fermipy.skymap.Map` representation of the counts map.

        Returns
        -------

        map : `~fermipy.skymap.Map`

        """
        return self._ccube

    def model_counts_map(self, name=None, exclude=None):
        """Return the model counts map for a single source, a list of
        sources, or for the sum of all sources in the ROI.  The
        exclude parameter can be used to exclude one or more
        components when generating the model map.

        Parameters
        ----------
        name : str or list of str

           Parameter controlling the set of sources for which the
           model counts map will be calculated.  If name=None the
           model map will be generated for all sources in the ROI.

        exclude : str or list of str

           List of sources that will be excluded when calculating the
           model map.

        Returns
        -------

        map : `~fermipy.skymap.Map`
        """

        maps = [c.model_counts_map(name, exclude) for c in self.components]

        if self.projtype == "HPX":
            shape = (self.enumbins, self._proj.npix)
            cmap = skymap.make_coadd_map(maps, self._proj, shape)
        elif self.projtype == "WCS":
            shape = (self.enumbins, self.npix, self.npix)
            cmap = skymap.make_coadd_map(maps, self._proj, shape)
        else:
            raise Exception(
                "Did not recognize projection type %s", self.projtype)
        return cmap

    def model_counts_spectrum(self, name, logemin=None, logemax=None,
                              summed=False):
        """Return the predicted number of model counts versus energy
        for a given source and energy range.  If summed=True return
        the counts spectrum summed over all components otherwise
        return a list of model spectra."""

        if logemin is None:
            logemin = self.log_energies[0]
        if logemax is None:
            logemax = self.log_energies[-1]

        if summed:
            cs = np.zeros(self.enumbins)
            imin = utils.val_to_bin_bounded(self.log_energies,
                                            logemin + 1E-7)[0]
            imax = utils.val_to_bin_bounded(self.log_energies,
                                            logemax - 1E-7)[0] + 1

            for c in self.components:
                ecenter = 0.5 * (c.log_energies[:-1] + c.log_energies[1:])
                counts = c.model_counts_spectrum(name, self.log_energies[0],
                                                 self.log_energies[-1])

                cs += np.histogram(ecenter,
                                   weights=counts,
                                   bins=self.log_energies)[0]

            return cs[imin:imax]
        else:
            cs = []
            for c in self.components:
                cs += [c.model_counts_spectrum(name, logemin, logemax)]
            return cs

    def get_sources(self, cuts=None, distance=None, skydir=None,
                    minmax_ts=None, minmax_npred=None, exclude=None,
                    square=False):
        """Retrieve list of sources in the ROI satisfying the given
        selections.

        Returns
        -------
        srcs : list
            A list of `~fermipy.roi_model.Model` objects.

        """

        coordsys = self.config['binning']['coordsys']
        return self.roi.get_sources(skydir, distance, cuts,
                                    minmax_ts, minmax_npred,
                                    exclude, square,
                                    coordsys=coordsys)

    def add_source(self, name, src_dict, free=None, init_source=True,
                   save_source_maps=True, use_pylike=True,
                   use_single_psf=False, **kwargs):
        """Add a source to the ROI model.  This function may be called
        either before or after `~fermipy.gtanalysis.GTAnalysis.setup`.

        Parameters
        ----------
        name : str
            Source name.

        src_dict : dict or `~fermipy.roi_model.Source` object
            Dictionary or source object defining the source properties
            (coordinates, spectral parameters, etc.).

        free : bool
            Initialize the source with a free normalization parameter.

        use_pylike : bool
            Create source maps with pyLikelihood.

        use_single_psf : bool 
            Use the PSF model calculated for the ROI center.  If false
            then a new model will be generated using the position of
            the source.

        """

        if self.roi.has_source(name):
            msg = 'Source %s already exists.' % name
            self.logger.error(msg)
            raise Exception(msg)

        loglevel = kwargs.pop('loglevel', self.loglevel)

        self.logger.log(loglevel, 'Adding source ' + name)

        src = self.roi.create_source(name, src_dict, rescale=True)

        for c in self.components:
            c.add_source(name, src_dict, free=free,
                         save_source_maps=save_source_maps,
                         use_pylike=use_pylike,
                         use_single_psf=use_single_psf)

        if self._like is None:
            return

        if self.config['gtlike']['edisp'] and src.name not in \
                self.config['gtlike']['edisp_disable']:
            self.set_edisp_flag(src.name, True)

        self.like.syncSrcParams(str(name))
        self.like.model = self.like.components[0].model
        # if free is not None:
        #    self.free_norm(name, free, loglevel=logging.DEBUG)

        if init_source:
            self._init_source(name)
            self._update_roi()

        if self._fitcache is not None:
            self._fitcache.update_source(name)

    def add_sources_from_roi(self, names, roi, free=False, **kwargs):
        """Add multiple sources to the current ROI model copied from another ROI model.

        Parameters
        ----------

        names : list
            List of str source names to add.

        roi : `~fermipy.roi_model.ROIModel` object
            The roi model from which to add sources.

        free : bool
            Initialize the source with a free normalization paramter.

        """

        for name in names:
            self.add_source(name, roi[name].data, free=free, **kwargs)

    def delete_source(self, name, save_template=True, delete_source_map=False,
                      build_fixed_wts=True, **kwargs):
        """Delete a source from the ROI model.

        Parameters
        ----------
        name : str
            Source name.

        save_template : bool
            Keep the SpatialMap FITS template associated with this
            source.

        delete_source_map : bool
            Delete the source map associated with this source from the
            source maps file.

        Returns
        -------
        src : `~fermipy.roi_model.Model`
            The deleted source object.

        """

        if not self.roi.has_source(name):
            self.logger.error('No source with name: %s', name)
            return

        loglevel = kwargs.pop('loglevel', self.loglevel)

        self.logger.log(loglevel, 'Deleting source %s', name)

        # STs require a source to be freed before deletion
        if self.like is not None:
            self.free_norm(name, loglevel=logging.DEBUG)

        for c in self.components:
            c.delete_source(name, save_template=save_template,
                            delete_source_map=delete_source_map,
                            build_fixed_wts=build_fixed_wts)

        src = self.roi.get_source_by_name(name)
        self.roi.delete_sources([src])
        if self.like is not None:
            self.like.model = self.like.components[0].model
            self._update_roi()
        return src

    def delete_sources(self, cuts=None, distance=None,
                       skydir=None, minmax_ts=None, minmax_npred=None,
                       exclude=None, square=False):
        """Delete sources in the ROI model satisfying the given
        selection criteria.

        Parameters
        ----------
        cuts : dict
            Dictionary of [min,max] selections on source properties.

        distance : float
            Cut on angular distance from ``skydir``.  If None then no
            selection will be applied.

        skydir : `~astropy.coordinates.SkyCoord`
            Reference sky coordinate for ``distance`` selection.  If
            None then the distance selection will be applied with
            respect to the ROI center.

        minmax_ts : list
            Free sources that have TS in the range [min,max].  If
            either min or max are None then only a lower (upper) bound
            will be applied.  If this parameter is none no selection
            will be applied.

        minmax_npred : list
            Free sources that have npred in the range [min,max].  If
            either min or max are None then only a lower (upper) bound
            will be applied.  If this parameter is none no selection
            will be applied.

        square : bool
            Switch between applying a circular or square (ROI-like)
            selection on the maximum projected distance from the ROI
            center.

        Returns
        -------
        srcs : list
            A list of `~fermipy.roi_model.Model` objects.

        """

        srcs = self.roi.get_sources(skydir=skydir, distance=distance, cuts=cuts,
                                    minmax_ts=minmax_ts, minmax_npred=minmax_npred,
                                    exclude=exclude, square=square,
                                    coordsys=self.config['binning']['coordsys'])

        for s in srcs:
            self.delete_source(s.name, build_fixed_wts=False)

        if self.like is not None:
            # Build fixed model weights in one pass
            for c in self.components:
                c.like.logLike.buildFixedModelWts()

            self._update_roi()

        return srcs

    def free_sources_by_name(self, names, free=True, pars=None,
                             **kwargs):
        """Free all sources with names matching ``names``.

        Parameters
        ----------
        names : list
            List of source names.

        free : bool
            Choose whether to free (free=True) or fix (free=False)
            source parameters.

        pars : list
            Set a list of parameters to be freed/fixed for each
            source.  If none then all source parameters will be
            freed/fixed.  If pars='norm' then only normalization
            parameters will be freed.

        Returns
        -------
        srcs : list
            A list of `~fermipy.roi_model.Model` objects.            
        """

        if names is None:
            return

        names = [names] if not isinstance(names, list) else names
        names = [self.roi.get_source_by_name(t).name for t in names]
        srcs = [s for s in self.roi.sources if s.name in names]
        for s in srcs:
            self.free_source(s.name, free=free, pars=pars, **kwargs)

        return srcs

    def free_sources(self, free=True, pars=None, cuts=None,
                     distance=None, skydir=None, minmax_ts=None, minmax_npred=None,
                     exclude=None, square=False, **kwargs):
        """Free or fix sources in the ROI model satisfying the given
        selection.  When multiple selections are defined, the selected
        sources will be those satisfying the logical AND of all
        selections (e.g. distance < X && minmax_ts[0] < ts <
        minmax_ts[1] && ...).

        Parameters
        ----------

        free : bool
            Choose whether to free (free=True) or fix (free=False)
            source parameters.

        pars : list
            Set a list of parameters to be freed/fixed for each
            source.  If none then all source parameters will be
            freed/fixed.  If pars='norm' then only normalization
            parameters will be freed.

        cuts : dict
            Dictionary of [min,max] selections on source properties.

        distance : float
            Cut on angular distance from ``skydir``.  If None then no
            selection will be applied.

        skydir : `~astropy.coordinates.SkyCoord`
            Reference sky coordinate for ``distance`` selection.  If
            None then the distance selection will be applied with
            respect to the ROI center.

        minmax_ts : list
            Free sources that have TS in the range [min,max].  If
            either min or max are None then only a lower (upper) bound
            will be applied.  If this parameter is none no selection
            will be applied.

        minmax_npred : list
            Free sources that have npred in the range [min,max].  If
            either min or max are None then only a lower (upper) bound
            will be applied.  If this parameter is none no selection
            will be applied.

        exclude : list
            Names of sources that will be excluded from the selection.

        square : bool
            Switch between applying a circular or square (ROI-like)
            selection on the maximum projected distance from the ROI
            center.

        Returns
        -------
        srcs : list
            A list of `~fermipy.roi_model.Model` objects.

        """

        srcs = self.roi.get_sources(skydir=skydir, distance=distance,
                                    cuts=cuts, minmax_ts=minmax_ts,
                                    minmax_npred=minmax_npred, exclude=exclude,
                                    square=square,
                                    coordsys=self.config['binning']['coordsys'])

        for s in srcs:
            self.free_source(s.name, free=free, pars=pars, **kwargs)

        return srcs

    def set_edisp_flag(self, name, flag=True):
        """Enable or disable the energy dispersion correction for the
        given source."""
        src = self.roi.get_source_by_name(name)
        name = src.name

        for c in self.components:
            c.like[name].src.set_edisp_flag(flag)

    def scale_parameter(self, name, par, scale):

        idx = self.like.par_index(name, par)
        self.like[idx].setScale(self.like[idx].getScale() * scale)
        self._sync_params(name)

    def _set_value_bounded(self, idx, value):
        bounds = list(self.like.model[idx].getBounds())
        value = max(value, bounds[0])
        value = min(value, bounds[1])
        self.like[idx].setValue(value)

    def set_parameter(self, name, par, value, true_value=True, scale=None,
                      bounds=None, update_source=True):
        """
        Update the value of a parameter.  Parameter bounds will
        automatically be adjusted to encompass the new parameter
        value.

        Parameters
        ----------

        name : str
            Source name.

        par : str
            Parameter name.

        value : float
            Parameter value.  By default this argument should be the
            unscaled (True) parameter value.

        scale : float
            Parameter scale (optional).  Value argument is interpreted
            with respect to the scale parameter if it is provided.

        update_source : bool
            Update the source dictionary for the object.

        """
        name = self.roi.get_source_by_name(name).name
        idx = self.like.par_index(name, par)
        current_bounds = list(self.like.model[idx].getBounds())

        if scale is not None:
            self.like[idx].setScale(scale)
        else:
            scale = self.like.model[idx].getScale()

        if true_value:
            current_bounds[0] = min(current_bounds[0], value / scale)
            current_bounds[1] = max(current_bounds[1], value / scale)
        else:
            current_bounds[0] = min(current_bounds[0], value)
            current_bounds[1] = max(current_bounds[1], value)

        # update current bounds to encompass new value
        self.like[idx].setBounds(*current_bounds)

        if true_value:
            for p in self.like[idx].pars:
                p.setTrueValue(value)
        else:
            self.like[idx].setValue(value)

        if bounds is not None:
            self.like[idx].setBounds(*bounds)

        self._sync_params(name)

        if update_source:
            self.update_source(name)

    def set_parameter_scale(self, name, par, scale):
        """Update the scale of a parameter while keeping its value constant."""
        name = self.roi.get_source_by_name(name).name
        idx = self.like.par_index(name, par)
        current_bounds = list(self.like.model[idx].getBounds())
        current_scale = self.like.model[idx].getScale()
        current_value = self.like[idx].getValue()

        self.like[idx].setScale(scale)
        self.like[idx].setValue(current_value * current_scale / scale)
        self.like[idx].setBounds(current_bounds[0] * current_scale / scale,
                                 current_bounds[1] * current_scale / scale)
        self._sync_params(name)

    def set_parameter_bounds(self, name, par, bounds):
        """Set the bounds of a parameter.

        Parameters
        ----------

        name : str
            Source name.

        par : str
            Parameter name.

        bounds : list
            Upper and lower bound.

        """
        idx = self.like.par_index(name, par)
        self.like[idx].setBounds(*bounds)
        self._sync_params(name)

    def free_parameter(self, name, par, free=True):
        idx = self.like.par_index(name, par)
        self.like[idx].setFree(free)
        self._sync_params(name)

    def free_source(self, name, free=True, pars=None, **kwargs):
        """Free/Fix parameters of a source.

        Parameters
        ----------

        name : str
            Source name.

        free : bool
            Choose whether to free (free=True) or fix (free=False)
            source parameters.

        pars : list
            Set a list of parameters to be freed/fixed for this source.  If
            none then all source parameters will be freed/fixed with the
            exception of those defined in the skip_pars list.

        """

        free_pars = self.get_free_param_vector()

        loglevel = kwargs.pop('loglevel', self.loglevel)

        # Find the source
        src = self.roi.get_source_by_name(name)
        name = src.name

        if pars is None or (isinstance(pars, list) and not pars):
            pars = []
            pars += norm_parameters.get(src['SpectrumType'], [])
            pars += shape_parameters.get(src['SpectrumType'], [])
        elif pars == 'norm':
            pars = []
            pars += norm_parameters.get(src['SpectrumType'], [])
        elif pars == 'shape':
            pars = []
            pars += shape_parameters.get(src['SpectrumType'], [])
        elif isinstance(pars, list):
            pass
        else:
            raise Exception('Invalid parameter list.')

        # Deduce here the names of all parameters from the spectral type
        src_par_names = pyLike.StringVector()
        self.like[name].src.spectrum().getParamNames(src_par_names)

        par_indices = []
        par_names = []
        for p in src_par_names:
            if pars is not None and p not in pars:
                continue

            idx = self.like.par_index(name, p)
            if free == free_pars[idx]:
                continue

            par_indices.append(idx)
            par_names.append(p)

        if len(par_names) == 0:
            return

        if free:
            self.logger.log(loglevel, 'Freeing parameters for %-22s: %s',
                            name, par_names)
        else:
            self.logger.log(loglevel, 'Fixing parameters for %-22s: %s',
                            name, par_names)

        for (idx, par_name) in zip(par_indices, par_names):
            self.like[idx].setFree(free)
        self._sync_params_state(name)

    def set_norm_scale(self, name, value):
        name = self.get_source_name(name)
        normPar = self.like.normPar(name)
        normPar.setScale(value)
        self._sync_params(name)

    def set_norm(self, name, value, update_source=True):
        name = self.get_source_name(name)
        par = self.like.normPar(name).getName()
        self.set_parameter(name, par, value, true_value=False,
                           update_source=update_source)

    def set_norm_bounds(self, name, bounds):
        name = self.get_source_name(name)
        par = self.like.normPar(name).getName()
        self.set_parameter_bounds(name, par, bounds)

    def free_norm(self, name, free=True, **kwargs):
        """Free/Fix normalization of a source.

        Parameters
        ----------

        name : str
            Source name.

        free : bool
            Choose whether to free (free=True) or fix (free=False).

        """

        name = self.get_source_name(name)
        normPar = self.like.normPar(name).getName()
        self.free_source(name, pars=[normPar], free=free, **kwargs)

    def free_index(self, name, free=True, **kwargs):
        """Free/Fix index of a source.

        Parameters
        ----------

        name : str
            Source name.

        free : bool
            Choose whether to free (free=True) or fix (free=False).

        """
        src = self.roi.get_source_by_name(name)
        self.free_source(name, free=free,
                         pars=index_parameters.get(src['SpectrumType'], []),
                         **kwargs)

    def free_shape(self, name, free=True, **kwargs):
        """Free/Fix shape parameters of a source.

        Parameters
        ----------

        name : str
            Source name.

        free : bool
            Choose whether to free (free=True) or fix (free=False).
        """
        src = self.roi.get_source_by_name(name)
        self.free_source(name, free=free,
                         pars=shape_parameters[src['SpectrumType']],
                         **kwargs)

    def _sync_params(self, name):
        self.like.syncSrcParams(str(name))
        src = self.components[0].like.logLike.getSource(str(name))
        spectral_pars = gtutils.get_function_pars_dict(src.spectrum())
        self.roi[name].set_spectral_pars(spectral_pars)
        for c in self.components:
            c.roi[name].set_spectral_pars(spectral_pars)

    def _sync_params_state(self, name=None):

        if name is None:
            self.like.syncSrcParams()
            names = self.like.sourceNames()
        else:
            self.like.syncSrcParams(str(name))
            names = [name]

        for name in names:

            src = self.like[name].src
            pars = gtutils.get_function_pars(src.spectrum())

            for p in pars:
                self.roi[name].spectral_pars[p['name']]['free'] = p['free']
                for c in self.components:
                    c.roi[name].spectral_pars[p['name']]['free'] = p['free']

    def get_norm(self, name):
        name = self.get_source_name(name)
        return self.like.normPar(name).getValue()

    def get_params(self, freeonly=False):

        params = {}
        for srcName in self.like.sourceNames():

            par_names = pyLike.StringVector()
            src = self.components[0].like.logLike.getSource(str(srcName))
            src.spectrum().getParamNames(par_names)

#            for parName in self.get_free_source_params(srcName):
            for parName in par_names:
                idx = self.like.par_index(srcName, parName)
                par = self.like.model[idx]
                bounds = par.getBounds()

                is_norm = parName == self.like.normPar(srcName).getName()

                if freeonly and not par.isFree():
                    continue

                params[idx] = {'src_name': srcName,
                               'par_name': parName,
                               'value': par.getValue(),
                               'error': par.error(),
                               'scale': par.getScale(),
                               'idx': idx,
                               'free': par.isFree(),
                               'is_norm': is_norm,
                               'bounds': bounds}

        return [params[k] for k in sorted(params.keys())]

    def get_free_param_vector(self):
        free = []
        for p in self.like.params():
            free.append(p.isFree())
        return free

    def set_free_param_vector(self, free):
        for i, t in enumerate(free):
            if t:
                self.like.thaw(i)
            else:
                self.like.freeze(i)

        self._sync_params_state()

    def _latch_free_params(self):
        self._free_params = self.get_free_param_vector()

    def _restore_free_params(self):
        self.set_free_param_vector(self._free_params)

    def _latch_state(self):
        self._saved_state = LikelihoodState(self.like)
        return self._saved_state

    def _restore_state(self):
        if self._saved_state is None:
            return
        self._saved_state.restore()

    def get_free_source_params(self, name):
        name = self.get_source_name(name)
        spectrum = self.like[name].src.spectrum()
        parNames = pyLike.StringVector()
        spectrum.getFreeParamNames(parNames)
        return [str(p) for p in parNames]

    def get_source_name(self, name):
        """Return the name of a source as it is defined in the
        pyLikelihood model object."""
        if name not in self.like.sourceNames():
            name = self.roi.get_source_by_name(name).name
        return name

    def zero_source(self, name, **kwargs):
        normPar = self.like.normPar(name).getName()
        self.scale_parameter(name, normPar, 1E-10)
        self.free_source(name, free=False, **kwargs)

    def unzero_source(self, name, **kwargs):
        normPar = self.like.normPar(name).getName()
        self.scale_parameter(name, normPar, 1E10)

    def optimize(self, **kwargs):
        """Iteratively optimize the ROI model.  The optimization is
        performed in three sequential steps:

        * Free the normalization of the N largest components (as
          determined from NPred) that contain a fraction ``npred_frac``
          of the total predicted counts in the model and perform a
          simultaneous fit of the normalization parameters of these
          components.

        * Individually fit the normalizations of all sources that were
          not included in the first step in order of their npred
          values.  Skip any sources that have NPred <
          ``npred_threshold``.

        * Individually fit the shape and normalization parameters of
          all sources with TS > ``shape_ts_threshold`` where TS is
          determined from the first two steps of the ROI optimization.

        To ensure that the model is fully optimized this method can be
        run multiple times.

        Parameters
        ----------

        npred_frac : float
            Threshold on the fractional number of counts in the N
            largest components in the ROI.  This parameter determines
            the set of sources that are fit in the first optimization
            step.

        npred_threshold : float
            Threshold on the minimum number of counts of individual
            sources.  This parameter determines the sources that are
            fit in the second optimization step.

        shape_ts_threshold : float
            Threshold on source TS used for determining the sources
            that will be fit in the third optimization step.

        max_free_sources : int
            Maximum number of sources that will be fit simultaneously
            in the first optimization step.

        skip : list
            List of str source names to skip while optimizing.

        optimizer : dict
            Dictionary that overrides the default optimizer settings.

        """

        loglevel = kwargs.pop('loglevel', self.loglevel)

        self.logger.log(loglevel, 'Starting')

        loglike0 = -self.like()
        self.logger.debug('LogLike: %f' % loglike0)

        # Extract options from kwargs
        config = copy.deepcopy(self.config['roiopt'])
        config['optimizer'] = copy.deepcopy(self.config['optimizer'])
        fermipy.config.validate_config(kwargs, config)
        config = merge_dict(config, kwargs)

        # Extract options from kwargs
        npred_frac_threshold = config['npred_frac']
        npred_threshold = config['npred_threshold']
        shape_ts_threshold = config['shape_ts_threshold']
        max_free_sources = config['max_free_sources']
        skip = copy.deepcopy(config['skip'])

        o = defaults.make_default_dict(defaults.roiopt_output)
        o['config'] = config
        o['loglike0'] = loglike0

        # preserve free parameters
        free = self.get_free_param_vector()

        # Fix all parameters
        self.free_sources(free=False, loglevel=logging.DEBUG)

        # Free norms of sources for which the sum of npred is a
        # fraction > npred_frac of the total model counts in the ROI
        npred_sum = 0
        skip_sources = skip if skip != None else []
        joint_norm_fit = []
        for s in sorted(self.roi.sources, key=lambda t: t['npred'],
                        reverse=True):

            npred_sum += s['npred']
            npred_frac = npred_sum / self._roi_data['npred']

            if s.name in skip_sources:
                continue

            self.free_norm(s.name, loglevel=logging.DEBUG)
            joint_norm_fit.append(s.name)

            if npred_frac > npred_frac_threshold:
                break
            if s['npred'] < npred_threshold:
                break
            if len(joint_norm_fit) >= max_free_sources:
                break

        self.fit(loglevel=logging.DEBUG, **config['optimizer'])
        self.free_sources(free=False, loglevel=logging.DEBUG)

        # Step through remaining sources and re-fit normalizations
        for s in sorted(self.roi.sources, key=lambda t: t['npred'],
                        reverse=True):

            if s.name in skip_sources or s.name in joint_norm_fit:
                continue

            if s['npred'] < npred_threshold:
                self.logger.debug(
                    'Skipping %s with npred %10.3f', s.name, s['npred'])
                continue

            self.logger.debug('Fitting %s npred: %10.3f TS: %10.3f',
                              s.name, s['npred'], s['ts'])
            self.free_norm(s.name, loglevel=logging.DEBUG)
            self.fit(loglevel=logging.DEBUG, **config['optimizer'])
            self.logger.debug('Post-fit Results npred: %10.3f TS: %10.3f',
                              s['npred'], s['ts'])
            self.free_norm(s.name, free=False, loglevel=logging.DEBUG)

        # Refit spectral shape parameters for sources with TS >
        # shape_ts_threshold
        for s in sorted(self.roi.sources,
                        key=lambda t: t['ts'] if np.isfinite(t['ts']) else 0,
                        reverse=True):

            if s.name in skip_sources:
                continue

            if s['ts'] < shape_ts_threshold or not np.isfinite(s['ts']):
                continue

            self.logger.debug('Fitting shape %s TS: %10.3f', s.name, s['ts'])
            self.free_source(s.name, loglevel=logging.DEBUG)
            self.fit(loglevel=logging.DEBUG, **config['optimizer'])
            self.free_source(s.name, free=False, loglevel=logging.DEBUG)

        self.set_free_param_vector(free)

        loglike1 = -self.like()

        o['loglike1'] = loglike1
        o['dloglike'] = loglike1 - loglike0

        self.logger.log(loglevel, 'Finished')
        self.logger.log(loglevel, 'LogLike: %f Delta-LogLike: %f',
                        loglike1, loglike1 - loglike0)

        return o

    def profile_norm(self, name, logemin=None, logemax=None, reoptimize=False,
                     xvals=None, npts=None, fix_shape=True, savestate=True,
                     **kwargs):
        """Profile the normalization of a source.

        Parameters
        ----------

        name : str
            Source name.

        reoptimize : bool
            Re-optimize free parameters in the model at each point
            in the profile likelihood scan.

        """

        self.logger.debug('Profiling %s', name)

        if savestate:
            saved_state = LikelihoodState(self.like)

        if fix_shape:
            self.free_sources(False, pars='shape', loglevel=logging.DEBUG)

        if npts is None:
            npts = self.config['gtlike']['llscan_npts']

        # Find the source
        name = self.roi.get_source_by_name(name).name
        parName = self.like.normPar(name).getName()

        loge_bounds = self.loge_bounds
        if logemin is not None or logemax is not None:
            self.set_energy_range(logemin, logemax)

        # Find a sequence of values for the normalization scan
        if xvals is None:
            if reoptimize:
                xvals = self._find_scan_pts_reopt(name, npts=npts,
                                                  **kwargs)
            else:
                xvals = self._find_scan_pts(name, npts=9)
                lnlp = self.profile(name, parName,
                                    reoptimize=False, xvals=xvals)
                lims = utils.get_parameter_limits(lnlp['xvals'],
                                                  lnlp['dloglike'],
                                                  ul_confidence=0.99)

                if not np.isfinite(lims['ul']):
                    self.logger.warning('Upper limit not found.  '
                                        'Refitting normalization.')
                    self.like.optimize(0)
                    xvals = self._find_scan_pts(name, npts=npts)
                    lnlp = self.profile(name, parName,
                                        reoptimize=False,
                                        xvals=xvals)
                    lims = utils.get_parameter_limits(lnlp['xvals'],
                                                      lnlp['dloglike'],
                                                      ul_confidence=0.99)

                if np.isfinite(lims['ll']):
                    xhi = np.linspace(lims['x0'], lims['ul'], npts - npts // 2)
                    xlo = np.linspace(lims['ll'], lims['x0'], npts // 2)
                    xvals = np.concatenate((xlo[:-1], xhi))
                    xvals = np.insert(xvals, 0, 0.0)
                elif np.abs(lnlp['dloglike'][0] - lims['lnlmax']) > 0.1:
                    lims['ll'] = 0.0
                    xhi = np.linspace(lims['x0'], lims['ul'],
                                      (npts + 1) - (npts + 1) // 2)
                    xlo = np.linspace(lims['ll'], lims['x0'], (npts + 1) // 2)
                    xvals = np.concatenate((xlo[:-1], xhi))
                else:
                    xvals = np.linspace(0, lims['ul'], npts)

        o = self.profile(name, parName,
                         reoptimize=reoptimize, xvals=xvals,
                         savestate=savestate, **kwargs)

        if savestate:
            saved_state.restore()

        if logemin is not None or logemax is not None:
            self.set_energy_range(*loge_bounds)

        self.logger.debug('Finished')

        return o

    def _find_scan_pts(self, name, logemin=None, logemax=None, npts=20):

        par = self.like.normPar(name)

        loge_bounds = [self.loge_bounds[0] if logemin is None else logemin,
                       self.loge_bounds[1] if logemax is None else logemax]

        val = par.getValue()

        if val == 0:
            par.setValue(1.0)
            self.like.syncSrcParams(str(name))
            cs = self.model_counts_spectrum(name,
                                            loge_bounds[0],
                                            loge_bounds[1],
                                            summed=True)
            npred = np.sum(cs)
            val = 1. / npred
            npred = 1.0
            par.setValue(0.0)
            self.like.syncSrcParams(str(name))
        else:
            cs = self.model_counts_spectrum(name,
                                            loge_bounds[0],
                                            loge_bounds[1],
                                            summed=True)
            npred = np.sum(cs)

        if npred < 10:
            val *= 1. / min(1.0, npred)
            xvals = val * 10 ** np.linspace(-1.0, 3.0, npts - 1)
            xvals = np.insert(xvals, 0, 0.0)
        else:

            npts_lo = npts // 2
            npts_hi = npts - npts_lo
            xhi = np.linspace(0, 1, npts_hi)
            xlo = np.linspace(-1, 0, npts_lo)
            xvals = val * 10 ** np.concatenate((xlo[:-1], xhi))
            xvals = np.insert(xvals, 0, 0.0)

        return xvals

    def _find_scan_pts_reopt(self, name, logemin=None, logemax=None, npts=20,
                             dloglike_thresh=3.0, **kwargs):

        parName = self.like.normPar(name).getName()

        npts = max(npts, 5)
        xvals = self._find_scan_pts(name, logemin=logemin, logemax=logemax,
                                    npts=20)

        lnlp0 = self.profile(name, parName, logemin=logemin, logemax=logemax,
                             reoptimize=False, xvals=xvals, **kwargs)
        xval0 = self.like.normPar(name).getValue()
        lims0 = utils.get_parameter_limits(lnlp0['xvals'], lnlp0['dloglike'],
                                           ul_confidence=0.99)

        if not np.isfinite(lims0['ll']) and lims0['x0'] > 1E-6:
            xvals = np.array([0.0, lims0['x0'],
                              lims0['x0'] + lims0['err_hi'], lims0['ul']])
        elif not np.isfinite(lims0['ll']) and lims0['x0'] < 1E-6:
            xvals = np.array([0.0, lims0['x0'] + lims0['err_hi'], lims0['ul']])
        else:
            xvals = np.array([lims0['ll'],
                              lims0['x0'] - lims0['err_lo'], lims0['x0'],
                              lims0['x0'] + lims0['err_hi'], lims0['ul']])

        lnlp1 = self.profile(name, parName, logemin=logemin, logemax=logemax,
                             reoptimize=True, xvals=xvals, **kwargs)

        loglike = copy.deepcopy(lnlp1['loglike'])
        dloglike = copy.deepcopy(lnlp1['dloglike'])
        dloglike0 = dloglike[-1]
        xup = xvals[-1]

        for i in range(20):

            lims1 = utils.get_parameter_limits(xvals, dloglike,
                                               ul_confidence=0.99)

#            print('iter',i,np.abs(np.abs(dloglike0) - utils.onesided_cl_to_dlnl(0.99)),xup)
#            print(loglike)

            if np.abs(np.abs(dloglike0) - utils.onesided_cl_to_dlnl(0.99)) < 0.1:
                break

            if not np.isfinite(lims1['ul']) or np.abs(dloglike[-1]) < 1.0:
                xup = 2.0 * xvals[-1]
            else:
                xup = lims1['ul']

            lnlp = self.profile(name, parName, logemin=logemin,
                                logemax=logemax,
                                reoptimize=True, xvals=[xup], **kwargs)
            dloglike0 = lnlp['dloglike']
            loglike0 = lnlp['loglike']

            loglike = np.concatenate((loglike, loglike0))
            dloglike = np.concatenate((dloglike, dloglike0))
            xvals = np.concatenate((xvals, [xup]))
            isort = np.argsort(xvals)
            dloglike = dloglike[isort]
            loglike = loglike[isort]
            xvals = xvals[isort]

        if np.isfinite(lims1['ll']):
            xlo = np.concatenate(
                ([0.0], np.linspace(lims1['ll'], xval0, (npts + 1) // 2 - 1)))
        elif np.abs(dloglike[0]) > 0.1:
            xlo = np.linspace(0.0, xval0, (npts + 1) // 2)
        else:
            xlo = np.array([0.0, xval0])

        if np.isfinite(lims1['ul']):
            xhi = np.linspace(xval0, lims1['ul'], npts + 1 - len(xlo))[1:]
        else:
            xhi = np.linspace(xval0, lims0['ul'], npts + 1 - len(xlo))[1:]

        xvals = np.concatenate((xlo, xhi))
        return xvals

    def profile(self, name, parName, logemin=None, logemax=None,
                reoptimize=False,
                xvals=None, npts=None, savestate=True, **kwargs):
        """Profile the likelihood for the given source and parameter.

        Parameters
        ----------

        name : str
           Source name.

        parName : str
           Parameter name.

        reoptimize : bool
           Re-fit nuisance parameters at each step in the scan.  Note
           that enabling this option will only re-fit parameters that
           were free when the method was executed.

        Returns
        -------

        lnlprofile : dict
           Dictionary containing results of likelihood scan.

        """
        # Find the source
        name = self.roi.get_source_by_name(name).name

        par = self.like.normPar(name)
        parName = self.like.normPar(name).getName()
        idx = self.like.par_index(name, parName)
        bounds = self.like.model[idx].getBounds()
        value = self.like.model[idx].getValue()
        loge_bounds = self.loge_bounds

        optimizer = kwargs.get('optimizer', self.config['optimizer'])

        if savestate:
            saved_state = self._latch_state()

        # If parameter is fixed temporarily free it
        par.setFree(True)
        if optimizer['optimizer'] == 'NEWTON':
            self._create_fitcache()

        if logemin is not None or logemax is not None:
            loge_bounds = self.set_energy_range(logemin, logemax)
        else:
            loge_bounds = self.loge_bounds

        loglike0 = -self.like()

        if xvals is None:

            err = par.error()
            val = par.getValue()
            if err <= 0 or val <= 3 * err:
                xvals = 10 ** np.linspace(-2.0, 2.0, 51)
                if val < xvals[0]:
                    xvals = np.insert(xvals, val, 0)
            else:
                xvals = np.linspace(0, 1, 25)
                xvals = np.concatenate((-1.0 * xvals[1:][::-1], xvals))
                xvals = val * 10 ** xvals

        # Update parameter bounds to encompass scan range
        self.like[idx].setBounds(min(min(xvals), value, bounds[0]),
                                 max(max(xvals), value, bounds[1]))

        o = {'xvals': xvals,
             'npred': np.zeros(len(xvals)),
             'dnde': np.zeros(len(xvals)),
             'flux': np.zeros(len(xvals)),
             'eflux': np.zeros(len(xvals)),
             'dloglike': np.zeros(len(xvals)),
             'loglike': np.zeros(len(xvals))
             }

        if reoptimize and hasattr(self.like.components[0].logLike,
                                  'setUpdateFixedWeights'):

            for c in self.components:
                c.like.logLike.setUpdateFixedWeights(False)

        for i, x in enumerate(xvals):

            self.like[idx] = x
            if self.like.nFreeParams() > 1 and reoptimize:
                # Only reoptimize if not all frozen
                self.like.freeze(idx)
                fit_output = self._fit(errors=False, **optimizer)
                loglike1 = fit_output['loglike']
                self.like.thaw(idx)
            else:
                loglike1 = -self.like()

            flux = self.like[name].flux(10 ** loge_bounds[0],
                                        10 ** loge_bounds[1])
            eflux = self.like[name].energyFlux(10 ** loge_bounds[0],
                                               10 ** loge_bounds[1])
            prefactor = self.like[idx]

            o['dloglike'][i] = loglike1 - loglike0
            o['loglike'][i] = loglike1
            o['dnde'][i] = prefactor.getTrueValue()
            o['flux'][i] = flux
            o['eflux'][i] = eflux

            cs = self.model_counts_spectrum(name,
                                            loge_bounds[0],
                                            loge_bounds[1], summed=True)
            o['npred'][i] += np.sum(cs)

        self.like[idx] = value

        if reoptimize and hasattr(self.like.components[0].logLike,
                                  'setUpdateFixedWeights'):

            for c in self.components:
                c.like.logLike.setUpdateFixedWeights(True)

        # Restore model parameters to original values
        if savestate:
            saved_state.restore()

        self.like[idx].setBounds(*bounds)
        if logemin is not None or logemax is not None:
            self.set_energy_range(*loge_bounds)

        return o

    def constrain_norms(self, srcNames, cov_scale=1.0):
        """Constrain the normalizations of one or more sources by
        adding gaussian priors with sigma equal to the parameter
        error times a scaling factor."""

        # Get the covariance matrix

        for name in srcNames:
            par = self.like.normPar(name)

            err = par.error()
            val = par.getValue()

            if par.error() == 0.0 or not par.isFree():
                continue

            self.add_gauss_prior(name, par.getName(),
                                 val, err * cov_scale)

    def add_gauss_prior(self, name, parName, mean, sigma):

        par = self.like[name].funcs["Spectrum"].params[parName]
        par.addGaussianPrior(mean, sigma)

    def remove_prior(self, name, parName):

        par = self.like[name].funcs["Spectrum"].params[parName]
        par.removePrior()

    def remove_priors(self):
        """Clear all priors."""

        for src in self.roi.sources:

            for par in self.like[src.name].funcs["Spectrum"].params.values():
                par.removePrior()

    def _create_optObject(self, **kwargs):
        """ Make MINUIT or NewMinuit type optimizer object """

        optimizer = kwargs.get('optimizer',
                               self.config['optimizer']['optimizer'])

        self.logger.debug("Creating optimizer: %s", optimizer)

        if optimizer.upper() == 'MINUIT':
            optObject = pyLike.Minuit(self.like.logLike)
        elif optimizer.upper() == 'NEWMINUIT':
            optObject = pyLike.NewMinuit(self.like.logLike)
        else:
            optFactory = pyLike.OptimizerFactory_instance()
            optObject = optFactory.create(str(optimizer), self.like.logLike)
        return optObject

    def _fit_optimizer_iter(self, **kwargs):

        min_fit_quality = kwargs.get('min_fit_quality', 3)
        retries = kwargs.get('retries', 3)
        covar = kwargs.get('covar', True)

        #saved_state = LikelihoodState(self.like)

        quality = 0
        niter = 0
        while niter < retries:
            self.logger.debug("Fit iteration: %i" % niter)
            niter += 1
            quality, status, edm, loglike = self._fit_optimizer(**kwargs)

            if quality >= min_fit_quality and status == 0:
                break

        num_free = self.like.nFreeParams()
        o = {'values': np.ones(num_free) * np.nan,
             'errors': np.ones(num_free) * np.nan,
             'indices': np.zeros(num_free, dtype=int),
             'is_norm': np.empty(num_free, dtype=bool),
             'src_names': num_free * [None],
             'par_names': num_free * [None]}

        o['fit_quality'] = quality
        o['fit_status'] = status
        o['edm'] = edm
        o['niter'] = niter
        o['loglike'] = loglike
        o['fit_success'] = True

        if quality < min_fit_quality or o['fit_status']:
            o['fit_success'] = False

        if covar:
            o['covariance'] = np.array(self.like.covariance)
            o['errors'] = np.diag(o['covariance'])**0.5
            errinv = 1. / o['errors']
            o['correlation'] = \
                o['covariance'] * errinv[:, np.newaxis] * errinv[np.newaxis, :]

        free_params = self.get_params(True)
        for i, p in enumerate(free_params):
            o['values'][i] = p['value']
            o['errors'][i] = p['error']
            o['indices'][i] = p['idx']
            o['src_names'][i] = p['src_name']
            o['par_names'][i] = p['par_name']
            o['is_norm'][i] = p['is_norm']

        return o

    def _fit_optimizer(self, **kwargs):

        errors = kwargs.get('errors', True)
        optObject = self._create_optObject(optimizer=kwargs.get('optimizer',
                                                                'MINUIT'))

        kw = {}

        quality = 3
        status = 0
        edm = 0
        loglike = 0

        try:
            if errors:
                kw['verbosity'] = kwargs.get('verbosity', 0)
                kw['tol'] = kwargs.get('tol', None)
                kw['covar'] = kwargs.get('covar', True)
                kw['optObject'] = optObject
                self.like.fit(**kw)
            else:
                kw['verbosity'] = kwargs.get('verbosity', 0)
                kw['tol'] = kwargs.get('tol', None)
                kw['optObject'] = optObject
                self.like.optimize(**kw)

            status = optObject.getRetCode()

            if isinstance(optObject, pyLike.Minuit):
                quality = optObject.getQuality()

            if isinstance(optObject, pyLike.Minuit) or \
                    isinstance(optObject, pyLike.NewMinuit):
                edm = optObject.getDistance()

            loglike = optObject.stat().value()

        except Exception:
            self.logger.error('Likelihood optimization failed.', exc_info=True)
            quality = 0
            status = 1

        return quality, status, edm, loglike

    def _fit(self, **kwargs):

        optimizer = kwargs.get('optimizer',
                               self.config['optimizer']['optimizer']).upper()

        if optimizer == 'NEWTON':

            params = self.get_params(True)
            for p in params:

                if p['free'] and not p['is_norm']:
                    optimizer = 'MINUIT'
                    kwargs['optimizer'] = 'MINUIT'
                    self.logger.debug(
                        'Found non-norm free parameter.  Reverting to MINUIT.')
                    break

        if optimizer == 'NEWTON':
            return self._fit_newton(**kwargs)
        else:
            return self._fit_optimizer_iter(**kwargs)

    def fit(self, update=True, **kwargs):
        """Run the likelihood optimization.  This will execute a fit of all
        parameters that are currently free in the model and update the
        charateristics of the corresponding model components (TS,
        npred, etc.).  The fit will be repeated N times (set with the
        `retries` parameter) until a fit quality greater than or equal
        to `min_fit_quality` and a fit status code of 0 is obtained.
        If the fit does not succeed after N retries then all parameter
        values will be reverted to their state prior to the execution
        of the fit.

        Parameters
        ----------

        update : bool
           Update the model dictionary for all sources with free
           parameters.

        tol : float
           Set the optimizer tolerance.

        verbosity : int
           Set the optimizer output level.

        optimizer : str
           Set the likelihood optimizer (e.g. MINUIT or NEWMINUIT).

        retries : int
           Set the number of times to rerun the fit when the fit quality
           is < 3.

        min_fit_quality : int
           Set the minimum fit quality.  If the fit quality is smaller
           than this value then all model parameters will be
           restored to their values prior to the fit.

        reoptimize : bool
           Refit background sources when updating source properties
           (TS and likelihood profiles).

        Returns
        -------

        fit : dict
           Dictionary containing diagnostic information from the fit
           (fit quality, parameter covariances, etc.).

        """

        loglevel = kwargs.pop('loglevel', self.loglevel)
        self.logger.log(loglevel, "Starting fit.")

        # Extract options from kwargs
        config = copy.deepcopy(self.config['optimizer'])
        config.setdefault('covar', True)
        config.setdefault('reoptimize', False)
        config = utils.merge_dict(config, kwargs)

        num_free = self.like.nFreeParams()

        loglike0 = -self.like()

        # Initialize output dict
        o = {'fit_quality': 3,
             'fit_status': 0,
             'fit_success': True,
             'dloglike': 0.0,
             'edm': 0.0,
             'loglike': loglike0,
             'covariance': None,
             'correlation': None,
             'values': np.ones(num_free) * np.nan,
             'errors': np.ones(num_free) * np.nan,
             'indices': np.zeros(num_free, dtype=int),
             'is_norm': np.empty(num_free, dtype=bool),
             'src_names': num_free * [None],
             'par_names': num_free * [None],
             'config': config
             }

        if not num_free:
            self.logger.log(loglevel, "Skipping fit.  No free parameters.")
            return o

        saved_state = LikelihoodState(self.like)

        fit_output = self._fit(**config)
        o.update(fit_output)

        self.logger.debug("Fit complete.")
        o['dloglike'] = o['loglike'] - loglike0

        if not o['fit_success']:
            self.logger.error('%s failed with status code %i fit quality %i',
                              config['optimizer'], o['fit_status'],
                              o['fit_quality'])
            saved_state.restore()
            return o

        if update:

            free_params = self.get_params(True)
            self._extract_correlation(o, free_params)
            for name in self.like.sourceNames():
                freePars = self.get_free_source_params(name)
                if len(freePars) == 0:
                    continue
                self.update_source(name, reoptimize=config['reoptimize'])

            # Update roi model counts
            self._update_roi()

        self.logger.log(loglevel, "Fit returned successfully. " +
                        "Quality: %3i Status: %3i",
                        o['fit_quality'], o['fit_status'])
        self.logger.log(loglevel, "LogLike: %12.3f DeltaLogLike: %12.3f ",
                        o['loglike'], o['dloglike'])
        return o

    def _create_fitcache(self, **kwargs):

        create_fitcache = kwargs.get('create_fitcache', False)
        if self._fitcache is not None and not create_fitcache:
            return self._fitcache

        tol = kwargs.get('tol', self.config['optimizer']['tol'])
        max_iter = kwargs.get('max_iter',
                              self.config['optimizer']['max_iter'])
        init_lambda = kwargs.get('init_lambda',
                                 self.config['optimizer']['init_lambda'])
        use_reduced = kwargs.get('use_reduced', True)

        self.logger.debug('Creating FitCache')
        self.logger.debug('\ntol: %.5g\nmax_iter: %i\ninit_lambda: %.5g',
                          tol, max_iter, init_lambda)

        params = self.get_params()
        self._fitcache = FitCache(self.like, params,
                                  tol, max_iter, init_lambda, use_reduced)

        return self._fitcache

    def _fit_newton(self, fitcache=None, ebin=None, **kwargs):
        """Fast fitting method using newton fitter."""

        tol = kwargs.get('tol', self.config['optimizer']['tol'])
        max_iter = kwargs.get('max_iter',
                              self.config['optimizer']['max_iter'])
        init_lambda = kwargs.get('init_lambda',
                                 self.config['optimizer']['init_lambda'])
        use_reduced = kwargs.get('use_reduced', True)

        free_params = self.get_params(True)
        free_norm_params = [p for p in free_params if p['is_norm'] is True]

        if len(free_params) != len(free_norm_params):
            msg = 'Executing Newton fitter with one ' + \
                'or more free shape parameters.'
            self.logger.error(msg)
            raise Exception(msg)

        verbosity = kwargs.get('verbosity', 0)
        if fitcache is None:
            fitcache = self._create_fitcache(**kwargs)

        fitcache.update(self.get_params(),
                        tol, max_iter, init_lambda, use_reduced)

        logemin = self.loge_bounds[0]
        logemax = self.loge_bounds[1]
        imin = int(utils.val_to_edge(self.log_energies, logemin)[0])
        imax = int(utils.val_to_edge(self.log_energies, logemax)[0])

        if ebin is not None:
            fitcache.fitcache.setEnergyBin(ebin)
        elif imin == 0 and imax == self.enumbins:
            fitcache.fitcache.setEnergyBin(-1)
        else:
            fitcache.fitcache.setEnergyBins(imin, imax)

        num_free = len(free_norm_params)

        o = {'fit_status': 0,
             'fit_quality': 3,
             'fit_success': True,
             'edm': 0,
             'loglike': None,
             'values': np.ones(num_free) * np.nan,
             'errors': np.ones(num_free) * np.nan,
             'indices': np.zeros(num_free, dtype=int),
             'is_norm': np.empty(num_free, dtype=bool),
             'src_names': num_free * [None],
             'par_names': num_free * [None],
             }

        if num_free == 0:
            return o

        ref_vals = np.array(fitcache.fitcache.refValues())
        free = np.array(fitcache.fitcache.currentFree())
        norm_vals = ref_vals[free]

        norm_idxs = []
        for i, p in enumerate(free_norm_params):

            norm_idxs += [p['idx']]
            o['indices'][i] = p['idx']
            o['src_names'][i] = p['src_name']
            o['par_names'][i] = p['par_name']
            o['is_norm'][i] = p['is_norm']

        o['fit_status'] = fitcache.fit(verbose=verbosity)
        o['edm'] = fitcache.fitcache.currentEDM()

        pars, errs, cov = fitcache.get_pars()
        pars *= norm_vals
        errs *= norm_vals
        cov = cov * np.outer(norm_vals, norm_vals)

        o['values'] = pars
        o['errors'] = errs
        o['covariance'] = cov
        errinv = np.zeros_like(o['errors'])
        m = o['errors'] > 0
        errinv[m] = 1. / o['errors'][m]
        o['correlation'] = o['covariance'] * np.outer(errinv, errinv)

        if o['fit_status'] in [-2, 0]:
            for idx, val, err in zip(norm_idxs, pars, errs):
                self._set_value_bounded(idx, val)
                self.like[idx].setError(err)
            self.like.syncSrcParams()
            o['fit_success'] = True
        else:
            o['fit_success'] = False

        if o['fit_status']:
            self.logger.error('Error in NEWTON fit. Fit Status: %i',
                              o['fit_status'])

        # FIXME: Figure out why currentLogLike gets out of sync
        #loglike = fitcache.fitcache.currentLogLike()
        #prior_vals, prior_errs, has_prior = gtutils.get_priors(self.like)
        #loglike -= np.sum(has_prior) * np.log(np.sqrt(2 * np.pi))            
        loglike = -self.like()        
        o['loglike'] = loglike

        return o

    def fit_correlation(self):

        saved_state = LikelihoodState(self.like)
        self.free_sources(False, loglevel=logging.DEBUG)
        self.free_sources(pars='norm', loglevel=logging.DEBUG)
        fit_results = self.fit(loglevel=logging.DEBUG, min_fit_quality=2)
        free_params = self.get_params(True)
        self._extract_correlation(fit_results, free_params)
        saved_state.restore()

    def _extract_correlation(self, fit_results, free_params):

        for i, p0 in enumerate(free_params):
            if not p0['is_norm']:
                continue

            src = self.roi[p0['src_name']]
            for j, p1 in enumerate(free_params):

                if not p1['is_norm']:
                    continue

                src['correlation'][p1['src_name']] = fit_results[
                    'correlation'][i, j]

    def load_xml(self, xmlfile):
        """Load model definition from XML.

        Parameters
        ----------
        xmlfile : str
            Name of the input XML file.

        """

        self.logger.info('Loading XML')

        for c in self.components:
            c.load_xml(xmlfile)

        for name in self.like.sourceNames():
            self.update_source(name)

        self._fitcache = None

        self.logger.info('Finished Loading XML')

    def write_xml(self, xmlfile):
        """Save current model definition as XML file.

        Parameters
        ----------
        xmlfile : str
            Name of the output XML file.

        """

        # Write a common XML file?

        for c in self._components:
            c.write_xml(xmlfile)

    def _restore_counts_maps(self):
        """
        Revert counts maps to their state prior to injecting any simulated
        components.
        """

        for c in self.components:
            c.restore_counts_maps()

        if hasattr(self.like.components[0].logLike, 'setCountsMap'):
            self._init_roi_model()
        else:
            self.write_xml('tmp')
            self._like = SummedLikelihood()
            for i, c in enumerate(self._components):
                c._create_binned_analysis()
                self._like.addComponent(c.like)
            self._init_roi_model()
            self.load_xml('tmp')

    def simulate_source(self, src_dict=None):
        """
        Inject simulated source counts into the data.

        Parameters
        ----------
        src_dict : dict
           Dictionary defining the spatial and spectral properties of
           the source that will be injected.

        """

        self._fitcache = None

        if src_dict is None:
            src_dict = {}
        else:
            src_dict = copy.deepcopy(src_dict)

        skydir = wcs_utils.get_target_skydir(src_dict, self.roi.skydir)

        src_dict.setdefault('ra', skydir.ra.deg)
        src_dict.setdefault('dec', skydir.dec.deg)
        src_dict.setdefault('SpatialModel', 'PointSource')
        src_dict.setdefault('SpatialWidth', 0.3)
        src_dict.setdefault('Index', 2.0)
        src_dict.setdefault('Prefactor', 1E-13)

        self.add_source('mcsource', src_dict, free=True,
                        init_source=False)
        for c in self.components:
            c.simulate_roi('mcsource', clear=False)

        self.delete_source('mcsource')

        if hasattr(self.like.components[0].logLike, 'setCountsMap'):
            self._init_roi_model()
        else:
            self.write_xml('tmp')
            self._like = SummedLikelihood()
            for i, c in enumerate(self._components):
                c._create_binned_analysis('tmp.xml')
                self._like.addComponent(c.like)
            self._init_roi_model()
            self.load_xml('tmp')

    def simulate_roi(self, name=None, randomize=True, restore=False):
        """Generate a simulation of the ROI using the current best-fit model
        and replace the data counts cube with this simulation.  The
        simulation is created by generating an array of Poisson random
        numbers with expectation values drawn from the model cube of
        the binned analysis instance.  This function will update the
        counts cube both in memory and in the source map file.  The
        counts cube can be restored to its original state by calling
        this method with ``restore`` = True.

        Parameters
        ----------
        name : str
           Name of the model component to be simulated.  If None then
           the whole ROI will be simulated.

        restore : bool
           Restore the data counts cube to its original state.
        """

        self.logger.info('Simulating ROI')

        self._fitcache = None

        if restore:
            self.logger.info('Restoring')
            self._restore_counts_maps()
            self.logger.info('Finished')
            return

        for c in self.components:
            c.simulate_roi(name=name, clear=True, randomize=randomize)

        if hasattr(self.like.components[0].logLike, 'setCountsMap'):
            self._init_roi_model()
        else:
            self.write_xml('tmp')
            self._like = SummedLikelihood()
            for i, c in enumerate(self._components):
                c._create_binned_analysis('tmp.xml')
                self._like.addComponent(c.like)
            self._init_roi_model()
            self.load_xml('tmp')

        self.logger.info('Finished')

    def write_model_map(self, model_name, name=None):
        """Save the counts model map to a FITS file.

        Parameters
        ----------
        model_name : str
            String that will be append to the name of the output file.
        name : str
            Name of the component.

        Returns
        -------

        """
        maps = [c.write_model_map(model_name, name) for c in self.components]

        outfile = os.path.join(self.workdir,
                               'mcube_%s.fits' % (model_name))

        if self.projtype == "HPX":
            shape = (self.enumbins, self._proj.npix)
            model_counts = skymap.make_coadd_map(maps, self._proj, shape)
            fits_utils.write_hpx_image(
                model_counts.counts, self._proj, outfile)
        elif self.projtype == "WCS":
            shape = (self.enumbins, self.npix, self.npix)
            model_counts = skymap.make_coadd_map(maps, self._proj, shape)
            fits_utils.write_fits_image(
                model_counts.counts, self._proj, outfile)
        else:
            raise Exception(
                "Did not recognize projection type %s", self.projtype)
        return [model_counts] + maps

    def print_roi(self, loglevel=logging.INFO):
        """Print information about the spectral and spatial properties
        of the ROI (sources, diffuse components)."""
        self.logger.log(loglevel, '\n' + str(self.roi))

    def print_params(self, allpars=False, loglevel=logging.INFO):
        """Print information about the model parameters (values,
        errors, bounds, scale)."""

        pars = self.get_params()

        o = '\n'
        o += '%4s %-20s%10s%10s%10s%10s%10s%5s\n' % (
            'idx', 'parname', 'value', 'error',
            'min', 'max', 'scale', 'free')

        o += '-' * 80 + '\n'

        src_pars = collections.OrderedDict()
        for p in pars:

            src_pars.setdefault(p['src_name'], [])
            src_pars[p['src_name']] += [p]

        free_sources = []
        for k, v in src_pars.items():

            for p in v:
                if not p['free']:
                    continue

                free_sources += [k]

        for k, v in src_pars.items():

            if not allpars and k not in free_sources:
                continue

            o += '%s\n' % k
            for p in v:

                o += '%4i %-20.19s' % (p['idx'], p['par_name'])
                o += '%10.3g%10.3g' % (p['value'], p['error'])
                o += '%10.3g%10.3g%10.3g' % (p['bounds'][0], p['bounds'][1],
                                             p['scale'])

                if p['free']:
                    o += '    *'
                else:
                    o += '     '

                o += '\n'

        self.logger.log(loglevel, o)

    def print_model(self, loglevel=logging.INFO):

        o = '\n'
        o += '%-20s%8s%8s%7s%10s%10s%12s%5s\n' % (
            'sourcename', 'offset', 'norm', 'eflux', 'index',
            'ts', 'npred', 'free')
        o += '-' * 80 + '\n'

        for s in sorted(self.roi.sources, key=lambda t: t['offset']):
            if s.diffuse:
                continue

            normVal = self.like.normPar(s.name).getValue()
            fixed = self.like[s.name].fixedSpectrum()

            if fixed:
                free_str = ' '
            else:
                free_str = '*'

            if s['SpectrumType'] == 'PowerLaw':
                index = s['dnde1000_index']
            else:
                index = 0.5 * (s['dnde1000_index'] +
                               s['dnde10000_index'])

            o += '%-20.19s%8.3f%8.3f%10.3g%7.2f%10.2f%12.1f%5s\n' % (
                s['name'], s['offset'], normVal, s['eflux'], index,
                s['ts'], s['npred'], free_str)

        for s in sorted(self.roi.sources, key=lambda t: t['offset']):
            if not s.diffuse:
                continue

            normVal = self.like.normPar(s.name).getValue()
            fixed = self.like[s.name].fixedSpectrum()

            if fixed:
                free_str = ' '
            else:
                free_str = '*'

            if s['SpectrumType'] == 'PowerLaw':
                index = s['dnde1000_index']
            else:
                index = 0.5 * (s['dnde1000_index'] +
                               s['dnde10000_index'])

            o += '%-20.19s%8s%8.3f%10.3g%7.2f%10.2f%12.1f%5s\n' % (
                s['name'],
                '---', normVal, s['eflux'], index, s['ts'], s['npred'], free_str)

        self.logger.log(loglevel, o)

    def load_roi(self, infile, reload_sources=False):
        """This function reloads the analysis state from a previously
        saved instance generated with
        `~fermipy.gtanalysis.GTAnalysis.write_roi`.

        Parameters
        ----------

        infile : str

        reload_sources : bool
           Regenerate source maps for non-diffuse sources.

        """

        infile = utils.resolve_path(infile, workdir=self.workdir)
        roi_file, roi_data = utils.load_data(infile, workdir=self.workdir)

        self.logger.info('Loading ROI file: %s', roi_file)

        key_map = {'dfde': 'dnde',
                   'dfde100': 'dnde100',
                   'dfde1000': 'dnde1000',
                   'dfde10000': 'dnde10000',
                   'dfde_index': 'dnde_index',
                   'dfde100_index': 'dnde100_index',
                   'dfde1000_index': 'dnde1000_index',
                   'dfde10000_index': 'dnde10000_index',
                   'e2dfde': 'e2dnde',
                   'e2dfde100': 'e2dnde100',
                   'e2dfde1000': 'e2dnde1000',
                   'e2dfde10000': 'e2dnde10000',
                   'Npred': 'npred',
                   'logLike': 'loglike',
                   'dlogLike': 'dloglike',
                   'emin': 'e_min',
                   'ectr': 'e_ctr',
                   'emax': 'e_max',
                   'logemin': 'loge_min',
                   'logectr': 'loge_ctr',
                   'logemax': 'loge_max',
                   'ref_dfde': 'ref_dnde',
                   'ref_e2dfde': 'ref_e2dnde',
                   'ref_dfde_emin': 'ref_dnde_e_min',
                   'ref_dfde_emax': 'ref_dnde_e_max',
                   }

        self._roi_data = utils.update_keys(roi_data['roi'], key_map)

        if 'erange' in self._roi_data:
            self._roi_data['loge_bounds'] = self._roi_data.pop('erange')

        self._loge_bounds = self._roi_data.setdefault('loge_bounds',
                                                       self.loge_bounds)

        sources = roi_data.pop('sources')
        sources = utils.update_keys(sources, key_map)
        for k0, v0 in sources.items():
            for k, v in defaults.source_flux_output.items():
                if k not in v0:
                    continue
                if v[2] == float and isinstance(v0[k], np.ndarray):
                    sources[k0][k], sources[k0][k + '_err'] \
                        = v0[k][0], v0[k][1]

        self.roi.load_sources(sources.values())
        for i, c in enumerate(self.components):
            c.roi.load_sources(sources.values())
            if 'src_expscale' in self._roi_data['components'][i]:
                c._src_expscale = copy.deepcopy(self._roi_data['components']
                                                [i]['src_expscale'])

        self._create_likelihood(infile)
        self.set_energy_range(self.loge_bounds[0], self.loge_bounds[1])

        if reload_sources:
            names = [s.name for s in self.roi.sources if not s.diffuse]
            self.reload_sources(names, False)

        self.logger.info('Finished Loading ROI')

    def write_roi(self, outfile=None,
                  save_model_map=False, fmt='npy', **kwargs):
        """Write current state of the analysis to a file.  This method
        writes an XML model definition, a ROI dictionary, and a FITS
        source catalog file.  A previously saved analysis state can be
        reloaded from the ROI dictionary file with the
        `~fermipy.gtanalysis.GTAnalysis.load_roi` method.

        Parameters
        ----------

        outfile : str
            String prefix of the output files.  The extension of this
            string will be stripped when generating the XML, YAML and
            npy filenames.

        make_plots : bool
            Generate diagnostic plots.

        save_model_map : bool
            Save the current counts model to a FITS file.

        fmt : str
            Set the output file format (yaml or npy).

        """
        # extract the results in a convenient format

        make_plots = kwargs.get('make_plots', False)

        if outfile is None:
            pathprefix = os.path.join(self.config['fileio']['workdir'],
                                      'results')
        elif not os.path.isabs(outfile):
            pathprefix = os.path.join(self.config['fileio']['workdir'],
                                      outfile)
        else:
            pathprefix = outfile

        pathprefix = utils.strip_suffix(pathprefix,
                                        ['fits', 'yaml', 'npy'])
#        pathprefix, ext = os.path.splitext(pathprefix)
        prefix = os.path.basename(pathprefix)

        xmlfile = pathprefix + '.xml'
        fitsfile = pathprefix + '.fits'
        npyfile = pathprefix + '.npy'
        ymlfile = pathprefix + '.yaml'

        self.write_xml(xmlfile)
        self.write_fits(fitsfile)

        if not self.config['gtlike']['use_external_srcmap']:
            for c in self.components:
                c.like.logLike.saveSourceMaps(str(c.files['srcmap']))

        if save_model_map:
            self.write_model_map(prefix)

        o = {}
        o['roi'] = copy.deepcopy(self._roi_data)
        o['config'] = copy.deepcopy(self.config)
        o['version'] = fermipy.__version__
        o['stversion'] = fermipy.get_st_version()
        o['sources'] = {}

        for s in self.roi.sources:
            o['sources'][s.name] = copy.deepcopy(s.data)

        for i, c in enumerate(self.components):
            o['roi']['components'][i][
                'src_expscale'] = copy.deepcopy(c.src_expscale)

        if fmt == 'yaml':
            self.logger.info('Writing %s...', ymlfile)
            utils.write_yaml(o, ymlfile)
        elif fmt == 'npy':
            self.logger.info('Writing %s...', npyfile)
            np.save(npyfile, o)
        else:
            raise Exception('Unrecognized output format: %s' % fmt)

        if make_plots:
            self.make_plots(prefix, None,
                            **kwargs.get('plotting', {}))

    def write_fits(self, fitsfile):

        self.logger.info('Writing %s...', fitsfile)

        tab = self.roi.create_table()
        hdu_data = fits.table_to_hdu(tab)
        hdu_data.name = 'CATALOG'

        tab_srcs = []
        tab_params = []
        for i, c in enumerate(self.components):
            tab = self.roi.create_source_table()
            tab['component'] = i
            tab['expscale'] = np.nan
            
            for k,v in c.src_expscale.items():
                m = tab['source_name'] == k
                tab['expscale'][m] = v
            
            tab_srcs += [tab]

            tab = self.roi.create_param_table()
            tab['component'] = i
            tab_params += [tab]

        from astropy.table import vstack
            
        hdu_srcs = fits.table_to_hdu(vstack(tab_srcs))
        hdu_srcs.name = 'SOURCES'
        hdu_params = fits.table_to_hdu(vstack(tab_params))
        hdu_params.name = 'PARAMS'        
        hdu_roi = fits.table_to_hdu(self.create_roi_table())
        hdu_roi.name = 'ROI'
        
        hdus = [fits.PrimaryHDU(), hdu_data, hdu_roi, hdu_srcs, hdu_params]
        hdus[0].header['CONFIG'] = json.dumps(self.config)
        hdus[1].header['CONFIG'] = json.dumps(self.config)
        fits_utils.write_hdus(hdus, fitsfile)

    def create_roi_table(self):

        rd = copy.deepcopy(self._roi_data)
        loge_bounds = rd.pop('loge_bounds').tolist()
        
        tab = fits_utils.dict_to_table(rd)
        tab['component'] = -1
        tab.meta['loge_bounds'] = loge_bounds
        
        row_dict = {}
        for i, c in enumerate(rd['components']):
            c['component'] = i

            row = []
            for k in tab.columns:

                shape = tab.columns[k].shape
                ndim = tab.columns[k].ndim                
                if ndim == 1:
                    val = c[k]
                else:
                    val = np.ones(shape[1:])*np.nan
                    val[:len(c[k])] = c[k]
                row += [val]
            tab.add_row(row)
        return tab
        
    def make_plots(self, prefix, mcube_map=None, **kwargs):
        """Make diagnostic plots using the current ROI model."""

        #mcube_maps = kwargs.pop('mcube_maps', None)
        if mcube_map is None:
            mcube_map = self.model_counts_map()

        plotter = plotting.AnalysisPlotter(self.config['plotting'],
                                           fileio=self.config['fileio'],
                                           logging=self.config['logging'])
        plotter.run(self, mcube_map, prefix=prefix, **kwargs)

    def bowtie(self, name, fd=None, loge=None):
        """Generate a spectral uncertainty band (bowtie) for the given
        source.  This will create an uncertainty band on the
        differential flux as a function of energy by propagating the
        errors on the global fit parameters.  Note that this band only
        reflects the uncertainty for parameters that are currently
        free in the model.

        Parameters
        ----------

        name : str
           Source name.

        fd : FluxDensity
           Flux density object.  If this parameter is None then one
           will be created.

        loge : array-like
           Sequence of energies in log10(E/MeV) at which the flux band
           will be evaluated.

        """

        if loge is None:
            logemin = self.log_energies[0]
            logemax = self.log_energies[-1]
            loge = np.linspace(logemin, logemax, 50)

        o = {'energies': 10**loge,
             'log_energies': loge,
             'dnde': np.zeros(len(loge)) * np.nan,
             'dnde_lo': np.zeros(len(loge)) * np.nan,
             'dnde_hi': np.zeros(len(loge)) * np.nan,
             'dnde_err': np.zeros(len(loge)) * np.nan,
             'dnde_ferr': np.zeros(len(loge)) * np.nan,
             'pivot_energy': np.nan}

        try:
            if fd is None:
                fd = FluxDensity.FluxDensity(self.like, name)
        except RuntimeError:
            self.logger.error('Failed to create FluxDensity',
                              exc_info=True)
            return o

        dnde = [fd.value(10 ** x) for x in loge]
        dnde_err = [fd.error(10 ** x) for x in loge]

        dnde = np.array(dnde)
        dnde_err = np.array(dnde_err)
        m = dnde > 0

        fhi = np.zeros_like(dnde)
        flo = np.zeros_like(dnde)
        ferr = np.zeros_like(dnde)

        fhi[m] = dnde[m] * (1.0 + dnde_err[m] / dnde[m])
        flo[m] = dnde[m] / (1.0 + dnde_err[m] / dnde[m])
        ferr[m] = 0.5 * (fhi[m] - flo[m]) / dnde[m]
        fhi[~m] = dnde_err[~m]

        o['dnde'] = dnde
        o['dnde_lo'] = flo
        o['dnde_hi'] = fhi
        o['dnde_err'] = dnde_err
        o['dnde_ferr'] = ferr

        try:
            o['pivot_energy'] = 10 ** utils.interpolate_function_min(loge, o[
                                                                     'dnde_ferr'])
        except Exception:
            self.logger.error('Failed to compute pivot energy',
                              exc_info=True)

        return o

    def _coadd_maps(self, cmaps, shape, rm):
        """
        """

        if self.projtype == "WCS":
            shape = (self.enumbins, self.npix, self.npix)
            self._ccube = skymap.make_coadd_map(cmaps, self._proj, shape)
            fits_utils.write_fits_image(self._ccube.counts, self._ccube.wcs,
                                        self.files['ccube'])
            rm['counts'] += np.squeeze(
                np.apply_over_axes(np.sum, self._ccube.counts,
                                   axes=[1, 2]))
        elif self.projtype == "HPX":
            self._ccube = skymap.make_coadd_map(cmaps, self._proj, shape)
            fits_utils.write_hpx_image(self._ccube.counts, self._ccube.hpx,
                                       self.files['ccube'])
            rm['counts'] += np.squeeze(
                np.apply_over_axes(np.sum, self._ccube.counts,
                                   axes=[1]))
        else:
            raise Exception(
                "Did not recognize projection type %s", self.projtype)

    def update_source(self, name, paramsonly=False, reoptimize=False, **kwargs):
        """Update the dictionary for this source.

        Parameters
        ----------

        name : str

        paramsonly : bool

        reoptimize : bool
           Re-fit background parameters in likelihood scan.

        """

        npts = self.config['gtlike']['llscan_npts']
        optimizer = kwargs.get('optimizer', self.config['optimizer'])

        sd = self.get_src_model(name, paramsonly, reoptimize, npts,
                                optimizer=optimizer)
        src = self.roi.get_source_by_name(name)
        src.update_data(sd)

        for c in self.components:
            src = c.roi.get_source_by_name(name)
            src.update_data(sd)

    def get_src_model(self, name, paramsonly=False, reoptimize=False,
                      npts=None, **kwargs):
        """Compose a dictionary for a source with the current best-fit
        parameters.

        Parameters
        ----------

        name : str

        paramsonly : bool
           Skip computing TS and likelihood profile.

        reoptimize : bool
           Re-fit background parameters in likelihood scan.

        npts : int
           Number of points for likelihood scan.

        Returns
        -------
        src_dict : dict

        """

        self.logger.debug('Generating source dict for ' + name)

        optimizer = kwargs.get('optimizer', self.config['optimizer'])
        if npts is None:
            npts = self.config['gtlike']['llscan_npts']

        name = self.get_source_name(name)
        source = self.like[name].src
        spectrum = source.spectrum()
        normPar = self.like.normPar(name)

        src_dict = defaults.make_default_dict(defaults.source_flux_output)
        src_dict.update({'name': name,
                         'pivot_energy': 1000.,
                         'ts': np.nan,
                         'loglike': np.nan,
                         'npred': 0.0,
                         'loglike_scan': np.nan * np.ones(npts),
                         'dloglike_scan': np.nan * np.ones(npts),
                         'eflux_scan': np.nan * np.ones(npts),
                         'flux_scan': np.nan * np.ones(npts),
                         'norm_scan': np.nan * np.ones(npts),
                         })

        src_dict.update(gtutils.gtlike_spectrum_to_vectors(spectrum))
        src_dict['spectral_pars'] = gtutils.get_function_pars_dict(spectrum)

        # Get Counts Spectrum
        src_dict['model_counts'] = self.model_counts_spectrum(
            name, summed=True)

        # Get NPred
        src_dict['npred'] = self.like.NpredValue(str(name))

        # Get the Model Fluxes
        try:
            src_dict['flux'] = self.like.flux(name, self.energies[0],
                                              self.energies[-1])
            src_dict['flux100'] = self.like.flux(name, 100., 10 ** 5.5)
            src_dict['flux1000'] = self.like.flux(name, 1000., 10 ** 5.5)
            src_dict['flux10000'] = self.like.flux(name, 10000., 10 ** 5.5)
            src_dict['eflux'] = self.like.energyFlux(name,
                                                     self.energies[0],
                                                     self.energies[-1])
            src_dict['eflux100'] = self.like.energyFlux(name, 100.,
                                                        10 ** 5.5)
            src_dict['eflux1000'] = self.like.energyFlux(name, 1000.,
                                                         10 ** 5.5)
            src_dict['eflux10000'] = self.like.energyFlux(name, 10000.,
                                                          10 ** 5.5)
            src_dict['dnde'] = self.like[name].spectrum()(
                pyLike.dArg(src_dict['pivot_energy']))
            src_dict['dnde100'] = self.like[name].spectrum()(
                pyLike.dArg(100.))
            src_dict['dnde1000'] = self.like[name].spectrum()(
                pyLike.dArg(1000.))
            src_dict['dnde10000'] = self.like[name].spectrum()(
                pyLike.dArg(10000.))

            if normPar.getValue() == 0:
                normPar.setValue(1.0)

                dnde_index = -get_spectral_index(self.like[name],
                                                 src_dict['pivot_energy'])

                dnde100_index = -get_spectral_index(self.like[name],
                                                    100.)
                dnde1000_index = -get_spectral_index(self.like[name],
                                                     1000.)
                dnde10000_index = -get_spectral_index(self.like[name],
                                                      10000.)

                normPar.setValue(0.0)
            else:
                dnde_index = -get_spectral_index(self.like[name],
                                                 src_dict['pivot_energy'])

                dnde100_index = -get_spectral_index(self.like[name],
                                                    100.)
                dnde1000_index = -get_spectral_index(self.like[name],
                                                     1000.)
                dnde10000_index = -get_spectral_index(self.like[name],
                                                      10000.)

            src_dict['dnde_index'] = dnde_index
            src_dict['dnde100_index'] = dnde100_index
            src_dict['dnde1000_index'] = dnde1000_index
            src_dict['dnde10000_index'] = dnde10000_index

        except Exception:
            self.logger.error('Failed to update source parameters.',
                              exc_info=True)

        # Only compute TS, errors, and ULs if the source was free in
        # the fit
        if not self.get_free_source_params(name) or paramsonly:
            return src_dict

        emax = 10 ** 5.5

        try:
            src_dict['flux_err'] = self.like.fluxError(name,
                                                       self.energies[0],
                                                       self.energies[-1])
            src_dict['flux100_err'] = self.like.fluxError(name, 100., emax)
            src_dict['flux1000_err'] = self.like.fluxError(name, 1000., emax)
            src_dict['flux10000_err'] = self.like.fluxError(name, 10000., emax)
            src_dict['eflux_err'] = \
                self.like.energyFluxError(name, self.energies[0],
                                          self.energies[-1])
            src_dict['eflux100_err'] = self.like.energyFluxError(name, 100.,
                                                                 emax)
            src_dict['eflux1000_err'] = self.like.energyFluxError(name, 1000.,
                                                                  emax)
            src_dict['eflux10000_err'] = self.like.energyFluxError(name, 10000.,
                                                                   emax)

        except Exception:
            pass
        # self.logger.error('Failed to update source parameters.',
        #  exc_info=True)
        lnlp = self.profile_norm(name, savestate=True,
                                 reoptimize=reoptimize, npts=npts,
                                 optimizer=optimizer)

        src_dict['loglike_scan'] = lnlp['loglike']
        src_dict['dloglike_scan'] = lnlp['dloglike']
        src_dict['eflux_scan'] = lnlp['eflux']
        src_dict['flux_scan'] = lnlp['flux']
        src_dict['norm_scan'] = lnlp['xvals']
        src_dict['loglike'] = np.max(lnlp['loglike'])

        flux_ul_data = utils.get_parameter_limits(
            lnlp['flux'], lnlp['dloglike'])
        eflux_ul_data = utils.get_parameter_limits(
            lnlp['eflux'], lnlp['dloglike'])

        if normPar.getValue() == 0:
            normPar.setValue(1.0)
            flux = self.like.flux(name, self.energies[0], self.energies[-1])
            flux100 = self.like.flux(name, 100., emax)
            flux1000 = self.like.flux(name, 1000., emax)
            flux10000 = self.like.flux(name, 10000., emax)
            eflux = self.like.energyFlux(name, self.energies[0],
                                         self.energies[-1])
            eflux100 = self.like.energyFlux(name, 100., emax)
            eflux1000 = self.like.energyFlux(name, 1000., emax)
            eflux10000 = self.like.energyFlux(name, 10000., emax)

            flux100_ratio = flux100 / flux
            flux1000_ratio = flux1000 / flux
            flux10000_ratio = flux10000 / flux
            eflux100_ratio = eflux100 / eflux
            eflux1000_ratio = eflux1000 / eflux
            eflux10000_ratio = eflux10000 / eflux
            normPar.setValue(0.0)
        else:
            flux100_ratio = src_dict['flux100'] / src_dict['flux']
            flux1000_ratio = src_dict['flux1000'] / src_dict['flux']
            flux10000_ratio = src_dict['flux10000'] / src_dict['flux']

            eflux100_ratio = src_dict['eflux100'] / src_dict['eflux']
            eflux1000_ratio = src_dict['eflux1000'] / src_dict['eflux']
            eflux10000_ratio = src_dict['eflux10000'] / src_dict['eflux']

        src_dict['flux_ul95'] = flux_ul_data['ul']
        src_dict['flux100_ul95'] = flux_ul_data['ul'] * flux100_ratio
        src_dict['flux1000_ul95'] = flux_ul_data['ul'] * flux1000_ratio
        src_dict['flux10000_ul95'] = flux_ul_data['ul'] * flux10000_ratio

        src_dict['eflux_ul95'] = eflux_ul_data['ul']
        src_dict['eflux100_ul95'] = eflux_ul_data['ul'] * eflux100_ratio
        src_dict['eflux1000_ul95'] = eflux_ul_data['ul'] * eflux1000_ratio
        src_dict['eflux10000_ul95'] = eflux_ul_data['ul'] * eflux10000_ratio

        # Extract covariance matrix
        fd = None
        try:
            fd = FluxDensity.FluxDensity(self.like, name)
            src_dict['covar'] = fd.covar
        except RuntimeError:
            pass
        # if ex.message == 'Covariance matrix has not been
        # computed.':

        # Extract bowtie
        if fd and len(src_dict['covar']) and src_dict['covar'].ndim >= 1:
            loge = np.linspace(self.log_energies[0],
                               self.log_energies[-1], 50)
            src_dict['model_flux'] = self.bowtie(name, fd=fd, loge=loge)
            src_dict['dnde100_err'] = fd.error(100.)
            src_dict['dnde1000_err'] = fd.error(1000.)
            src_dict['dnde10000_err'] = fd.error(10000.)

            src_dict['pivot_energy'] = src_dict['model_flux']['pivot_energy']

            e0 = src_dict['pivot_energy']
            src_dict['dnde'] = self.like[name].spectrum()(pyLike.dArg(e0))
            src_dict['dnde_err'] = fd.error(e0)

        if not reoptimize:
            src_dict['ts'] = self.like.Ts2(name, reoptimize=reoptimize)
        else:
            src_dict['ts'] = -2.0 * lnlp['dloglike'][0]

        return src_dict


class GTBinnedAnalysis(fermipy.config.Configurable):
    defaults = dict(selection=defaults.selection,
                    binning=defaults.binning,
                    ltcube=defaults.ltcube,
                    gtlike=defaults.gtlike,
                    data=defaults.data,
                    model=defaults.model,
                    logging=defaults.logging,
                    fileio=defaults.fileio,
                    name=('00', '', str),
                    file_suffix=('', '', str))

    def __init__(self, config, **kwargs):

        self._loglevel = kwargs.pop('loglevel', logging.INFO)

        super(GTBinnedAnalysis, self).__init__(config, **kwargs)

        self._projtype = self.config['binning']['projtype']

        self.logger = Logger.get(self.__class__.__name__,
                                 self.config['fileio']['logfile'],
                                 log_level(self.config['logging']['verbosity']))

        self._roi = ROIModel.create(self.config['selection'],
                                    self.config['model'],
                                    fileio=self.config['fileio'],
                                    coordsys=self.config['binning']['coordsys'])

        workdir = self.config['fileio']['workdir']
        self._name = self.config['name']

        search_dirs = [workdir]

        self._files = {}
        self._files['ft1'] = 'ft1%s.fits'
        self._files['ft1_filtered'] = 'ft1_filtered%s.fits'
        self._files['ccube'] = 'ccube%s.fits'
        self._files['ccubemc'] = 'ccubemc%s.fits'
        self._files['srcmap'] = 'srcmap%s.fits'
        self._files['bexpmap'] = 'bexpmap%s.fits'
        self._files['bexpmap_roi'] = 'bexpmap_roi%s.fits'
        self._files['srcmdl'] = 'srcmdl%s.xml'

        self._data_files = {}
        self._data_files['evfile'] = self.config['data']['evfile']
        self._data_files['scfile'] = self.config['data']['scfile']

        for k, v in self._data_files.items():
            if v is None:
                continue
            if not os.path.isfile(v):
                v = os.path.join(workdir, v)
            if not os.path.isfile(v):
                continue
            if not utils.is_fits_file(v):
                self._data_files[k] = \
                    utils.resolve_file_path_list(v, workdir,
                                                 prefix=k + '_' + self.name)
            else:
                self._data_files[k] = v

        self._srcmap_cache = {}
        self._srcmap = {}

        # Fill dictionary of exposure corrections
        self._src_expscale = {}
        if self.config['gtlike']['expscale'] is not None:
            for src in self.roi:
                self._src_expscale[src.name] = self.config[
                    'gtlike']['expscale']

        if self.config['gtlike']['src_expscale']:
            for k, v in self.config['gtlike']['src_expscale'].items():
                self._src_expscale[k] = v

        for k, v in self._files.items():
            self._files[k] = os.path.join(workdir,
                                          v % self.config['file_suffix'])

        for k in ['srcmap', 'bexpmap', 'bexpmap_roi']:

            if self.config['gtlike'].get(k, None) is None:
                continue

            self._files[k] = resolve_file_path(self.config['gtlike'][k],
                                               search_dirs=search_dirs,
                                               expand=True)

#        if self.config['gtlike'].get('srcmdl', None) is not None:
#            self._files['srcmdl'] = self.config['gtlike']['srcmdl']

        self._ext_ltcube = resolve_file_path(self.config['data']['ltcube'],
                                             search_dirs=search_dirs,
                                             expand=True)

        if self._ext_ltcube is None or \
                self.config['ltcube']['use_local_ltcube']:
            self.files['ltcube'] = os.path.join(workdir,
                                                'ltcube%s.fits' %
                                                self.config['file_suffix'])
        else:
            self.files['ltcube'] = self._ext_ltcube

        self._files['wmap'] = resolve_file_path(self.config['gtlike']['wmap'],
                                                search_dirs=search_dirs,
                                                expand=True)

        try:
            emin = self.config['selection']['emin']
            emax = self.config['selection']['emax']
            logemin = np.log10(emin)
            logemax = np.log10(emax)
        except AttributeError:
            logemin = self.config['selection']['logemin']
            logemax = self.config['selection']['logemax']
            emin = np.power(10., logemin)
            emax = np.power(10., logemax)

        self.config['selection']['logemin'] = logemin
        self.config['selection']['logemax'] = logemax
        self.config['selection']['emin'] = emin
        self.config['selection']['emax'] = emax

        if self.config['binning']['enumbins'] is not None:
            self._enumbins = int(self.config['binning']['enumbins'])
        else:
            self._enumbins = np.round(self.config['binning']['binsperdec'] *
                                      np.log10(emax / emin))
            self._enumbins = int(self._enumbins)

        self._ebin_edges = np.linspace(logemin, logemax,
                                       self._enumbins + 1)
        self._ebin_center = 0.5 * \
            (self._ebin_edges[1:] + self._ebin_edges[:-1])

        if self.config['binning']['npix'] is None:
            self._npix = int(np.round(self.config['binning']['roiwidth'] /
                                      self.config['binning']['binsz']))
        else:
            self._npix = self.config['binning']['npix']

        if self.config['selection']['radius'] is None:
            self._config['selection']['radius'] = float(
                np.sqrt(2.) * 0.5 * self.npix *
                self.config['binning']['binsz'] + 0.5)
            self.logger.debug(
                'Automatically setting selection radius to %s deg',
                self.config['selection']['radius'])

        if self.config['binning']['coordsys'] == 'CEL':
            self._xref = float(self.roi.skydir.ra.deg)
            self._yref = float(self.roi.skydir.dec.deg)
        elif self.config['binning']['coordsys'] == 'GAL':
            self._xref = float(self.roi.skydir.galactic.l.deg)
            self._yref = float(self.roi.skydir.galactic.b.deg)
        else:
            raise Exception('Unrecognized coord system: ' +
                            self.config['binning']['coordsys'])

        self._like = None
        self._coordsys = self.config['binning']['coordsys']
        self._tmin = self.config['selection']['tmin']
        self._tmax = self.config['selection']['tmax']

        if self.projtype == 'HPX':
            self._hpx_region = create_hpx_disk_region_string(self.roi.skydir,
                                                             self._coordsys,
                                                             0.5 * self.config['binning']['roiwidth'])
            self._proj = HPX.create_hpx(-1,
                                        self.config['binning'][
                                            'hpx_ordering_scheme'] == "NESTED",
                                        self._coordsys,
                                        self.config['binning']['hpx_order'],
                                        self._hpx_region,
                                        self.energies)
        elif self.projtype == "WCS":
            self._skywcs = wcs_utils.create_wcs(self._roi.skydir,
                                                coordsys=self._coordsys,
                                                projection=self.config[
                                                    'binning']['proj'],
                                                cdelt=self.binsz,
                                                crpix=1.0 + 0.5 *
                                                (self._npix - 1),
                                                naxis=2)
            self._proj = wcs_utils.create_wcs(self.roi.skydir,
                                              coordsys=self._coordsys,
                                              projection=self.config[
                                                  'binning']['proj'],
                                              cdelt=self.binsz,
                                              crpix=1.0 + 0.5 *
                                              (self._npix - 1),
                                              naxis=3,
                                              energies=self.energies)

        else:
            raise Exception(
                "Did not recognize projection type %s", self.projtype)

        self.print_config(self.logger, loglevel=logging.DEBUG)

    @property
    def loglevel(self):
        """Return the default loglevel."""
        return self._loglevel

    @property
    def roi(self):
        return self._roi

    @property
    def like(self):
        return self._like

    @property
    def name(self):
        return self._name

    @property
    def energies(self):
        """Return the energy bin edges in MeV."""
        return 10**self._ebin_edges

    @property
    def log_energies(self):
        """Return the energy bin edges in log10(E/MeV)."""
        return self._ebin_edges

    @property
    def enumbins(self):
        return len(self._ebin_edges) - 1

    @property
    def npix(self):
        return self._npix

    @property
    def binsz(self):
        return self.config['binning']['binsz']

    @property
    def roiwidth(self):
        return self._npix * self.config['binning']['binsz']

    @property
    def projtype(self):
        """Return the type of projection to use"""
        return self._projtype

    @property
    def tmin(self):
        """Return the MET time for the start of the observation."""
        return self._tmin

    @property
    def tmax(self):
        """Return the MET time for the end of the observation."""
        return self._tmax

    @property
    def wcs(self):
        if self.projtype == "WCS":
            return self._proj
        return None

    @property
    def hpx(self):
        if self.projtype == "HPX":
            return self._proj
        return None

    @property
    def coordsys(self):
        return self._coordsys

    @property
    def files(self):
        return self._files

    @property
    def data_files(self):
        return self._data_files

    @property
    def src_expscale(self):
        return self._src_expscale

    def reload_source(self, name):
        """Recompute the source map for a single source in the model.
        """

        src = self.roi.get_source_by_name(name)

        if hasattr(self.like.logLike, 'loadSourceMap'):
            self.like.logLike.loadSourceMap(str(name), True, False)
            srcmap_utils.delete_source_map(self.files['srcmap'], name)
            self.like.logLike.saveSourceMaps(str(self.files['srcmap']))
            self.like.logLike.buildFixedModelWts()
        else:
            self.write_xml('tmp')
            src = self.delete_source(name)
            self.add_source(name, src, free=True)
            self.load_xml('tmp')

    def reload_sources(self, names):
        """Recompute the source map for a list of sources in the model.
        """

        try:
            self.like.logLike.loadSourceMaps(names, True, True)
        except:
            for name in names:
                self.reload_source(name)

    def add_source(self, name, src_dict, free=None, save_source_maps=True,
                   use_pylike=True, use_single_psf=False):
        """Add a new source to the model.  Source properties
        (spectrum, spatial model) are set with the src_dict argument.


        Parameters
        ----------

        name : str
            Source name.

        src_dict : dict or `~fermipy.roi_model.Source` object
            Dictionary or Source object defining the properties of the
            source.

        free : bool
            Initialize the source with the normalization parameter free.

        save_source_maps : bool
            Write the source map for this source to the source maps file.

        use_pylike : bool

        use_single_psf : bool

        """

        if self.roi.has_source(name):
            msg = 'Source %s already exists.' % name
            self.logger.error(msg)
            raise Exception(msg)

        srcmap_utils.delete_source_map(self.files['srcmap'], name)

        src = self.roi.create_source(name, src_dict)
        self.make_template(src, self.config['file_suffix'])

        if self.config['gtlike']['expscale'] is not None and \
                name not in self._src_expscale:
            self._src_expscale[name] = self.config['gtlike']['expscale']

        if self._like is None:
            return

        if not use_pylike:
            self._update_srcmap_file([src], True)

        pylike_src = self._create_source(src)

        # Initialize source as free/fixed
        if free is not None:
            pylike_src.spectrum().normPar().setFree(free)

        if hasattr(pyLike, 'PsfIntegConfig') and \
                hasattr(pyLike.PsfIntegConfig, 'set_use_single_psf'):
            config = pyLike.BinnedLikeConfig(self.like.logLike.config())
            config.psf_integ_config().set_use_single_psf(use_single_psf)
            self.like.addSource(pylike_src, config)
        else:
            self.like.addSource(pylike_src)

        self.like.syncSrcParams(str(name))
        self.like.logLike.buildFixedModelWts()
        if save_source_maps and \
                not self.config['gtlike']['use_external_srcmap']:
            self.like.logLike.saveSourceMaps(str(self.files['srcmap']))

        self.set_exposure_scale(name)

    def _create_source(self, src):
        """Create a pyLikelihood Source object from a
        `~fermipy.roi_model.Model` object."""

        if src['SpatialType'] == 'SkyDirFunction':
            pylike_src = pyLike.PointSource(self.like.logLike.observation())
            pylike_src.setDir(src.skydir.ra.deg, src.skydir.dec.deg, False,
                              False)
        elif src['SpatialType'] == 'SpatialMap':
            sm = pyLike.SpatialMap(str(src['Spatial_Filename']))
            pylike_src = pyLike.DiffuseSource(sm,
                                              self.like.logLike.observation(),
                                              False)
        elif src['SpatialType'] == 'RadialGaussian':
            sm = pyLike.RadialGaussian(src.skydir.ra.deg, src.skydir.dec.deg,
                                       src.spatial_pars['Sigma']['value'])
            pylike_src = pyLike.DiffuseSource(sm,
                                              self.like.logLike.observation(),
                                              False)

        elif src['SpatialType'] == 'RadialDisk':
            sm = pyLike.RadialDisk(src.skydir.ra.deg, src.skydir.dec.deg,
                                   src.spatial_pars['Radius']['value'])
            pylike_src = pyLike.DiffuseSource(sm,
                                              self.like.logLike.observation(),
                                              False)

        elif src['SpatialType'] == 'MapCubeFunction':
            mcf = pyLike.MapCubeFunction2(str(src['Spatial_Filename']))
            pylike_src = pyLike.DiffuseSource(mcf,
                                              self.like.logLike.observation(),
                                              False)
        else:
            raise Exception('Unrecognized spatial type: %s',
                            src['SpatialType'])

        if src['SpectrumType'] == 'FileFunction':
            fn = gtutils.create_spectrum_from_dict(src['SpectrumType'],
                                                   src.spectral_pars)

            file_function = pyLike.FileFunction_cast(fn)
            filename = str(os.path.expandvars(src['Spectrum_Filename']))
            file_function.readFunction(filename)
        elif src['SpectrumType'] == 'DMFitFunction':

            fn = pyLike.DMFitFunction()
            fn = gtutils.create_spectrum_from_dict(src['SpectrumType'],
                                                   src.spectral_pars, fn)
            filename = str(os.path.expandvars(src['Spectrum_Filename']))
            fn.readFunction(filename)
        else:
            fn = gtutils.create_spectrum_from_dict(src['SpectrumType'],
                                                   src.spectral_pars)

        pylike_src.setSpectrum(fn)
        pylike_src.setName(str(src.name))

        return pylike_src

    def delete_source(self, name, save_template=True, delete_source_map=False,
                      build_fixed_wts=True):

        src = self.roi.get_source_by_name(name)

        self.logger.debug('Deleting source %s', name)

        if self.like is not None:

            normPar = self.like.normPar(name)
            isFree = normPar.isFree()

            if str(src.name) in self.like.sourceNames():
                self.like.deleteSource(str(src.name))
                self.like.logLike.eraseSourceMap(str(src.name))

            if not isFree and build_fixed_wts:
                self.like.logLike.buildFixedModelWts()

        if (not save_template and
            src['Spatial_Filename'] is not None and
            os.path.isfile(src['Spatial_Filename']) and
            os.path.dirname(src['Spatial_Filename']) == self.config['fileio']['workdir']):
            os.remove(src['Spatial_Filename'])

        self.roi.delete_sources([src])

        if delete_source_map:
            srcmap_utils.delete_source_map(self.files['srcmap'], name)

        return src

    def set_exposure_scale(self, name, scale=None):
        """Set the exposure correction of a source.

        Parameters
        ----------
        name : str
            Source name.

        scale : factor
            Exposure scale factor (1.0 = nominal exposure).

        """
        name = self.roi.get_source_by_name(name).name
        if scale is None and name not in self._src_expscale:
            return
        elif scale is None:
            scale = self._src_expscale.get(name, 1.0)
        else:
            self._src_expscale[name] = scale
        self._scale_srcmap({name: scale})

    def set_edisp_flag(self, name, flag=True):
        """Enable/Disable the energy dispersion correction for a
        source."""
        src = self.roi.get_source_by_name(name)
        name = src.name
        self.like[name].src.set_edisp_flag(flag)

    def set_energy_range(self, logemin, logemax):
        """Set the energy range of the analysis.

        Parameters
        ----------
        logemin: float
           Lower end of energy range in log10(E/MeV).

        logemax : float
           Upper end of energy range in log10(E/MeV).

        """

        if logemin is None:
            logemin = self.log_energies[0]

        if logemax is None:
            logemax = self.log_energies[-1]

        imin = int(utils.val_to_edge(self.log_energies, logemin)[0])
        imax = int(utils.val_to_edge(self.log_energies, logemax)[0])

        if imin - imax == 0:
            imin = int(len(self.log_energies) - 1)
            imax = int(len(self.log_energies) - 1)

        klims = self.like.logLike.klims()
        if imin != klims[0] or imax != klims[1]:
            self.like.selectEbounds(imin, imax)

        return np.array([self.log_energies[imin], self.log_energies[imax]])

    def counts_map(self):
        """Return 3-D counts map for this component as a Map object.

        Returns
        -------
        map : `~fermipy.skymap.MapBase`

        """
        try:
            p_method = self.like.logLike.countsMap().projection().method()
        except Exception:
            p_method = 0

        if p_method == 0:  # WCS
            z = self.like.logLike.countsMap().data()
            z = np.array(z).reshape(self.enumbins, self.npix, self.npix)
            return Map(z, copy.deepcopy(self.wcs))
        elif p_method == 1:  # HPX
            z = self.like.logLike.countsMap().data()
            nhpix = self.hpx.npix
            z = np.array(z).reshape(self.enumbins, nhpix)
            return HpxMap(z, self.hpx)
        else:
            self.logger.error('Did not recognize CountsMap type %i' % p_method,
                              exc_info=True)
        return None

    def model_counts_map(self, name=None, exclude=None):
        """Return the model expectation map for a single source, a set
        of sources, or all sources in the ROI.  The map will be
        computed using the current model parameters.

        Parameters
        ----------
        name : str

           Parameter that defines the sources for which the model map
           will be calculated.  If name=None a model map will be
           generated for all sources in the model.  If name='diffuse'
           a map for all diffuse sources will be generated.

        exclude : list

           Source name or list of source names that will be excluded
           from the model map.

        Returns
        -------
        map : `~fermipy.skymap.Map`

           A map object containing the counts and WCS projection.

        """
        if self.projtype == "WCS":
            v = pyLike.FloatVector(self.npix ** 2 * self.enumbins)
        elif self.projtype == "HPX":
            v = pyLike.FloatVector(self._proj.npix * self.enumbins)
        else:
            raise Exception("Unknown projection type %s", self.projtype)

        exclude = utils.arg_to_list(exclude)
        names = utils.arg_to_list(name)

        excluded_names = []
        for i, t in enumerate(exclude):
            srcs = self.roi.get_sources_by_name(t)
            excluded_names += [s.name for s in srcs]

        if not hasattr(self.like.logLike, 'loadSourceMaps'):
            # Update fixed model
            self.like.logLike.buildFixedModelWts()
            # Populate source map hash
            self.like.logLike.buildFixedModelWts(True)
        elif (name is None or name == 'all') and not exclude:
            self.like.logLike.loadSourceMaps()

        src_names = []
        if (name is None) or (name == 'all'):
            src_names = [src.name for src in self.roi.sources]
        elif name == 'diffuse':
            src_names = [src.name for src in self.roi.sources if
                         src.diffuse]
        else:
            srcs = [self.roi.get_source_by_name(t) for t in names]
            src_names = [src.name for src in srcs]

        # Remove sources in exclude list
        src_names = [str(t) for t in src_names if t not in excluded_names]

        if len(src_names) == len(self.roi.sources):
            self.like.logLike.computeModelMap(v)
        elif not hasattr(self.like.logLike, 'setSourceMapImage'):
            for s in src_names:
                model = self.like.logLike.sourceMap(str(s))
                self.like.logLike.updateModelMap(v, model)
        else:
            try:
                self.like.logLike.computeModelMap(src_names, v)
            except:
                vsum = np.zeros(v.size())
                for s in src_names:
                    vtmp = pyLike.FloatVector(v.size())
                    self.like.logLike.computeModelMap(str(s), vtmp)
                    vsum += vtmp
                v = pyLike.FloatVector(vsum)

        if self.projtype == "WCS":
            z = np.array(v).reshape(self.enumbins, self.npix, self.npix)
            return Map(z, copy.deepcopy(self.wcs))
        elif self.projtype == "HPX":
            z = np.array(v).reshape(self.enumbins, self._proj.npix)
            return HpxMap(z, self.hpx)
        else:
            raise Exception(
                "Did not recognize projection type %s", self.projtype)

    def model_counts_spectrum(self, name, logemin, logemax):
        """Return the model counts spectrum of a source.

        Parameters
        ----------
        name : str
           Source name.

        """

        cs = np.array(self.like.logLike.modelCountsSpectrum(str(name)))
        imin = utils.val_to_edge(self.log_energies, logemin)[0]
        imax = utils.val_to_edge(self.log_energies, logemax)[0]
        if imax <= imin:
            raise Exception('Invalid energy range.')
        return cs[imin:imax]

    def setup(self, overwrite=False, **kwargs):
        """Run pre-processing step for this component.  This will
        generate all of the auxiliary files needed to instantiate a
        likelihood object.  By default this function will skip any
        steps for which the output file already exists.

        Parameters
        ----------
        overwrite : bool

            Run all pre-processing steps even if the output file of
            that step is present in the working directory.
        """

        loglevel = kwargs.get('loglevel', self.loglevel)

        self.logger.log(loglevel, 'Running setup for component %s',
                        self.name)

        use_external_srcmap = self.config['gtlike']['use_external_srcmap']

        # Run data selection
        if not use_external_srcmap:
            self._select_data(overwrite=overwrite, **kwargs)

        # Create LT Cube
        if self._ext_ltcube is not None:
            self.logger.log(loglevel, 'Using external LT cube.')
        else:
            self._create_ltcube(overwrite=overwrite, **kwargs)

        self.logger.debug('Loading LT Cube %s', self.files['ltcube'])
        self._ltc = LTCube.create(self.files['ltcube'])

        # Extract tmin, tmax from LT cube
        self._tmin = self._ltc.tstart
        self._tmax = self._ltc.tstop

        self.logger.debug('Creating PSF model')
        self._psf = irfs.PSFModel.create(self.roi.skydir, self._ltc,
                                         self.config['gtlike']['irfs'],
                                         self.config['selection']['evtype'],
                                         self.energies)

        # Bin data and create exposure cube
        if not use_external_srcmap:
            self._bin_data(overwrite=overwrite, **kwargs)
            self._create_expcube(overwrite=overwrite, **kwargs)

        # Make spatial maps for extended sources
        for s in self.roi.sources:
            if s.diffuse:
                continue
            if not s.extended:
                continue
            self.make_template(s, self.config['file_suffix'])

        # Write ROI XML
        self.roi.write_xml(self.files['srcmdl'])

        # Create source maps file
        if not use_external_srcmap:
            self._create_srcmaps(overwrite=overwrite)

        if not self.config['data']['cacheft1'] and os.path.isfile(self.files['ft1']):
            self.logger.debug('Deleting FT1 file.')
            os.remove(self.files['ft1'])

        self.logger.log(loglevel, 'Finished setup for component %s',
                        self.name)

    def _select_data(self, overwrite=False, **kwargs):

        loglevel = kwargs.get('loglevel', self.loglevel)

        if (os.path.isfile(self.files['ft1']) and
            os.path.isfile(self.files['ccube']) and
                not overwrite):
            self.logger.log(loglevel, 'Skipping data selection.')
            return

        # Run gtselect and gtmktime
        kw_gtselect = dict(infile=self.data_files['evfile'],
                           outfile=self.files['ft1'],
                           ra=self.roi.skydir.ra.deg,
                           dec=self.roi.skydir.dec.deg,
                           rad=self.config['selection']['radius'],
                           convtype=self.config['selection']['convtype'],
                           phasemin=self.config['selection']['phasemin'],
                           phasemax=self.config['selection']['phasemax'],
                           evtype=self.config['selection']['evtype'],
                           evclass=self.config['selection']['evclass'],
                           tmin=self.config['selection']['tmin'],
                           tmax=self.config['selection']['tmax'],
                           emin=self.config['selection']['emin'],
                           emax=self.config['selection']['emax'],
                           zmax=self.config['selection']['zmax'],
                           chatter=self.config['logging']['chatter'])

        kw_gtmktime = dict(evfile=self.files['ft1'],
                           outfile=self.files['ft1_filtered'],
                           scfile=self.data_files['scfile'],
                           roicut=self.config['selection']['roicut'],
                           filter=self.config['selection']['filter'])

        run_gtapp('gtselect', self.logger, kw_gtselect,
                  loglevel=loglevel)
        if self.config['selection']['roicut'] == 'yes' or \
                self.config['selection']['filter'] is not None:
            run_gtapp('gtmktime', self.logger, kw_gtmktime,
                      loglevel=loglevel)
            os.system('mv %s %s' % (self.files['ft1_filtered'],
                                    self.files['ft1']))

    def _bin_data(self, overwrite=False, **kwargs):

        loglevel = kwargs.get('loglevel', self.loglevel)

        # Run gtbin
        if self.projtype == "WCS":
            kw = dict(algorithm='ccube',
                      nxpix=self.npix, nypix=self.npix,
                      binsz=self.config['binning']['binsz'],
                      evfile=self.files['ft1'],
                      outfile=self.files['ccube'],
                      scfile=self.data_files['scfile'],
                      xref=self._xref,
                      yref=self._yref,
                      axisrot=0,
                      proj=self.config['binning']['proj'],
                      ebinalg='LOG',
                      emin=self.config['selection']['emin'],
                      emax=self.config['selection']['emax'],
                      enumbins=self._enumbins,
                      coordsys=self.config['binning']['coordsys'],
                      chatter=self.config['logging']['chatter'])
        elif self.projtype == "HPX":
            hpx_region = "DISK(%.3f,%.3f,%.3f)" % (
                self._xref, self._yref, 0.5 * self.config['binning']['roiwidth'])
            kw = dict(algorithm='healpix',
                      evfile=self.files['ft1'],
                      outfile=self.files['ccube'],
                      scfile=self.data_files['scfile'],
                      xref=self._xref,
                      yref=self._yref,
                      proj=self.config['binning']['proj'],
                      hpx_ordering_scheme=self.config['binning'][
                          'hpx_ordering_scheme'],
                      hpx_order=self.config['binning']['hpx_order'],
                      hpx_ebin=self.config['binning']['hpx_ebin'],
                      hpx_region=hpx_region,
                      ebinalg='LOG',
                      emin=self.config['selection']['emin'],
                      emax=self.config['selection']['emax'],
                      enumbins=self._enumbins,
                      coordsys=self.config['binning']['coordsys'],
                      chatter=self.config['logging']['chatter'])
        else:
            self.logger.error(
                'Unknown projection type, %s. Choices are WCS or HPX',
                self.projtype)

        if not os.path.isfile(self.files['ccube']) or overwrite:
            run_gtapp('gtbin', self.logger, kw, loglevel=loglevel)
        else:
            self.logger.debug('Skipping gtbin.')

    def _create_ltcube(self, overwrite=False, **kwargs):

        loglevel = kwargs.get('loglevel', self.loglevel)

        if os.path.isfile(self.files['ltcube']) and not overwrite:
            self.logger.log(loglevel, 'Skipping LT Cube.')
            return

        # Run gtltcube
        kw = dict(evfile=self.files['ft1'],
                  scfile=self.data_files['scfile'],
                  outfile=self.files['ltcube'],
                  binsz=self.config['ltcube']['binsz'],
                  dcostheta=self.config['ltcube']['dcostheta'],
                  zmax=self.config['selection']['zmax'])

        if self.config['ltcube']['use_local_ltcube']:
            self.logger.info('Generating local LT cube.')
            colnames = ['START', 'STOP', 'LIVETIME',
                        'RA_SCZ', 'DEC_SCZ',
                        'RA_ZENITH', 'DEC_ZENITH']
            tab_sc = create_sc_table(self.data_files['scfile'],
                                     colnames=colnames)
            tab_gti = Table.read(self.files['ft1'], 'GTI')
            radius = self.config['selection']['radius'] + 10.0
            ltc_new = LTCube.create_from_gti(self.roi.skydir, tab_sc, tab_gti,
                                             self.config['selection']['zmax'],
                                             radius=radius)
            ltc_new.write(self.files['ltcube'])
        else:
            run_gtapp('gtltcube', self.logger, kw, loglevel=loglevel)

    def _create_expcube(self, overwrite=False, **kwargs):

        loglevel = kwargs.get('loglevel', self.loglevel)

        if os.path.isfile(self.files['bexpmap']) and not overwrite:
            self.logger.log(loglevel, 'Skipping gtexpcube.')
            return

        if self.config['gtlike']['irfs'] == 'CALDB':
            if self.projtype == "HPX":
                cmap = None
            else:
                cmap = self.files['ccube']
        else:
            cmap = 'none'

        # Run gtexpcube2
        kw = dict(infile=self.files['ltcube'], cmap=cmap,
                  ebinalg='LOG',
                  emin=self.config['selection']['emin'],
                  emax=self.config['selection']['emax'],
                  enumbins=self._enumbins,
                  outfile=self.files['bexpmap'], proj='CAR',
                  nxpix=360, nypix=180, binsz=1,
                  xref=0.0, yref=0.0,
                  evtype=self.config['selection']['evtype'],
                  irfs=self.config['gtlike']['irfs'],
                  coordsys=self.config['binning']['coordsys'],
                  chatter=self.config['logging']['chatter'])

        run_gtapp('gtexpcube2', self.logger, kw, loglevel=loglevel)

        if self.projtype == "WCS":
            kw = dict(infile=self.files['ltcube'], cmap='none',
                      ebinalg='LOG',
                      emin=self.config['selection']['emin'],
                      emax=self.config['selection']['emax'],
                      enumbins=self._enumbins,
                      outfile=self.files['bexpmap_roi'], proj='CAR',
                      nxpix=self.npix, nypix=self.npix,
                      binsz=self.config['binning']['binsz'],
                      xref=self._xref, yref=self._yref,
                      evtype=self.config['selection']['evtype'],
                      irfs=self.config['gtlike']['irfs'],
                      coordsys=self.config['binning']['coordsys'],
                      chatter=self.config['logging']['chatter'])

            run_gtapp('gtexpcube2', self.logger, kw, loglevel=loglevel)
        elif self.projtype == "HPX":
            self.logger.debug('Skipping local gtexpcube for HEALPix')
        else:
            raise Exception(
                "Did not recognize projection type %s", self.projtype)

    def _create_srcmaps(self, overwrite=False, **kwargs):

        loglevel = kwargs.get('loglevel', self.loglevel)

        # Run gtsrcmaps
        kw = dict(scfile=self.data_files['scfile'],
                  expcube=self.files['ltcube'],
                  cmap=self.files['ccube'],
                  srcmdl=self.files['srcmdl'],
                  bexpmap=self.files['bexpmap'],
                  outfile=self.files['srcmap'],
                  irfs=self.config['gtlike']['irfs'],
                  wmap=self.files['wmap'],
                  evtype=self.config['selection']['evtype'],
                  rfactor=self.config['gtlike']['rfactor'],
                  #                   resample=self.config['resample'],
                  minbinsz=self.config['gtlike']['minbinsz'],
                  chatter=self.config['logging']['chatter'],
                  emapbnds='no')

        if not os.path.isfile(self.files['srcmap']) or overwrite:
            run_gtapp('gtsrcmaps', self.logger, kw, loglevel=loglevel)
        else:
            self.logger.log(loglevel, 'Skipping gtsrcmaps.')

    def _create_binned_analysis(self, xmlfile=None, **kwargs):

        loglevel = kwargs.get('loglevel', self.loglevel)

        self.logger.log(loglevel, 'Creating BinnedAnalysis for component %s.',
                        self.name)

        srcmdl_file = self.files['srcmdl']
        if xmlfile is not None:
            srcmdl_file = self.get_model_path(xmlfile)

        # Create BinnedObs
        self.logger.debug('Creating BinnedObs')
        kw = dict(srcMaps=self.files['srcmap'], expCube=self.files['ltcube'],
                  binnedExpMap=self.files['bexpmap'],
                  irfs=self.config['gtlike']['irfs'])
        self.logger.debug(kw)

        self._obs = ba.BinnedObs(**utils.unicode_to_str(kw))

        # Create BinnedAnalysis
        self.logger.debug('Creating BinnedAnalysis')
        kw = dict(srcModel=srcmdl_file,
                  optimizer='MINUIT',
                  wmap=self._files['wmap'],
                  convolve=self.config['gtlike']['convolve'],
                  resample=self.config['gtlike']['resample'],
                  minbinsz=self.config['gtlike']['minbinsz'],
                  resamp_fact=self.config['gtlike']['rfactor'])
        self.logger.debug(kw)

        self._like = BinnedAnalysis(binnedData=self._obs,
                                    **utils.unicode_to_str(kw))

#        print self.like.logLike.use_single_fixed_map()
#        self.like.logLike.set_use_single_fixed_map(False)
#        print self.like.logLike.use_single_fixed_map()

        if self.config['gtlike']['edisp']:
            self.logger.debug('Enabling energy dispersion')
            self.like.logLike.set_edisp_flag(True)

        for s in self.config['gtlike']['edisp_disable']:

            if not self.roi.has_source(s):
                continue

            self.logger.debug('Disabling energy dispersion for %s', s)
            self.set_edisp_flag(s, False)

        # Recompute fixed model weights
        self.logger.debug('Computing fixed weights')
        self.like.logLike.buildFixedModelWts()
        self.logger.debug('Updating source maps')
        if not self.config['gtlike']['use_external_srcmap']:
            self.like.logLike.saveSourceMaps(str(self.files['srcmap']))

        # Apply exposure corrections
        self._scale_srcmap(self._src_expscale)

    def _scale_srcmap(self, scale_map):
        srcmap = fits.open(self.files['srcmap'])

        for hdu in srcmap[1:]:
            if hdu.name not in scale_map:
                continue

            scale = scale_map[hdu.name]
            if scale < 1e-20:
                self.logger.warning(
                    "The expscale parameter was zero, setting it to 1e-8")
                scale = 1e-8
            if 'EXPSCALE' in hdu.header:
                old_scale = hdu.header['EXPSCALE']
            else:
                old_scale = 1.0
            hdu.data *= scale / old_scale
            hdu.header['EXPSCALE'] = scale

        srcmap.writeto(self.files['srcmap'], clobber=True)

        for name in scale_map.keys():
            self.like.logLike.eraseSourceMap(str(name))
        self.like.logLike.buildFixedModelWts()

    def _make_scaled_srcmap(self):
        """Make an exposure cube with the same binning as the counts map."""

        self.logger.info('Computing scaled source map.')

        bexp0 = fits.open(self.files['bexpmap_roi'])
        bexp1 = fits.open(self.config['gtlike']['bexpmap'])
        srcmap = fits.open(self.config['gtlike']['srcmap'])

        if bexp0[0].data.shape != bexp1[0].data.shape:
            raise Exception('Wrong shape for input exposure map file.')

        bexp_ratio = bexp0[0].data / bexp1[0].data

        self.logger.info(
            'Min/Med/Max exposure correction: %f %f %f' % (np.min(bexp_ratio),
                                                           np.median(
                                                               bexp_ratio),
                                                           np.max(bexp_ratio)))

        for hdu in srcmap[1:]:

            if hdu.name == 'GTI':
                continue
            if hdu.name == 'EBOUNDS':
                continue
            hdu.data *= bexp_ratio

        srcmap.writeto(self.files['srcmap'], clobber=True)

    def restore_counts_maps(self):

        cmap = Map.create_from_fits(self.files['ccube'])

        if hasattr(self.like.logLike, 'setCountsMap'):
            self.like.logLike.setCountsMap(np.ravel(cmap.counts.astype(float)))

        srcmap_utils.update_source_maps(self.files['srcmap'],
                                        {'PRIMARY': cmap.counts},
                                        logger=self.logger)

    def simulate_roi(self, name=None, clear=True, randomize=True):
        """Simulate the whole ROI or inject a simulation of one or
        more model components into the data.

        Parameters
        ----------
        name : str
           Name of the model component to be simulated.  If None then
           the whole ROI will be simulated.

        clear : bool
           Zero the current counts map before injecting the simulation.

        randomize : bool
           Fill with each pixel with random values drawn from a
           poisson distribution.  If false then fill each pixel with
           the counts expectation value.

        """

        data = self.counts_map().counts
        m = self.model_counts_map(name)

        if clear:
            data.fill(0.0)

        if randomize:
            data += np.random.poisson(m.counts).astype(float)
        else:
            data += m.counts

        if hasattr(self.like.logLike, 'setCountsMap'):
            self.like.logLike.setCountsMap(np.ravel(data))

        srcmap_utils.update_source_maps(self.files['srcmap'],
                                        {'PRIMARY': data},
                                        logger=self.logger)
        fits_utils.write_fits_image(data, self.wcs, self.files['ccubemc'])

    def write_model_map(self, model_name=None, name=None):
        """Save counts model map to a FITS file.

        """

        if model_name is None:
            suffix = self.config['file_suffix']
        else:
            suffix = '_%s%s' % (model_name, self.config['file_suffix'])

        self.logger.info('Generating model map for component %s.', self.name)

        outfile = os.path.join(self.config['fileio']['workdir'],
                               'mcube%s.fits' % (suffix))

        cmap = self.model_counts_map(name)

        if self.projtype == "HPX":
            fits_utils.write_hpx_image(cmap.counts, cmap.hpx, outfile)
        elif self.projtype == "WCS":
            fits_utils.write_fits_image(cmap.counts, cmap.wcs, outfile)
        else:
            raise Exception(
                "Did not recognize projection type %s", self.projtype)
        return cmap

    def make_template(self, src, suffix):

        if src['SpatialType'] != 'SpatialMap':
            return
        if src['Spatial_Filename'] is not None:
            return

        if src['SpatialModel'] in ['RadialGaussian']:
            template_file = os.path.join(self.config['fileio']['workdir'],
                                         '%s_template_gauss_%05.3f%s.fits' % (
                                             src.name, src['SpatialWidth'],
                                             suffix))

            sigma = src['SpatialWidth'] / 1.5095921854516636
            srcmap_utils.make_gaussian_spatial_map(src.skydir, sigma,
                                                   template_file)
            src['Spatial_Filename'] = template_file
        elif src['SpatialModel'] in ['RadialDisk']:
            template_file = os.path.join(self.config['fileio']['workdir'],
                                         '%s_template_disk_%05.3f%s.fits' % (
                                             src.name, src['SpatialWidth'],
                                             suffix))
            radius = src['SpatialWidth'] / 0.8246211251235321
            srcmap_utils.make_disk_spatial_map(src.skydir, radius,
                                               template_file)
            src['Spatial_Filename'] = template_file

    def _update_srcmap_file(self, sources, overwrite=True):
        """Check the contents of the source map file and generate
        source maps for any components that are not present."""

        if not os.path.isfile(self.files['srcmap']):
            return

        hdulist = fits.open(self.files['srcmap'])
        hdunames = [hdu.name.upper() for hdu in hdulist]

        srcmaps = {}

        for src in sources:

            if src.name.upper() in hdunames and not overwrite:
                continue
            self.logger.debug('Creating source map for %s', src.name)
            srcmaps[src.name] = self._create_srcmap(src.name, src)

        if srcmaps:
            self.logger.debug(
                'Updating source map file for component %s.', self.name)
            srcmap_utils.update_source_maps(self.files['srcmap'], srcmaps,
                                            logger=self.logger)

    def _create_srcmap_cache(self, name, src, **kwargs):

        from fermipy.srcmap_utils import SourceMapCache

        skydir = src.skydir
        spatial_model = src['SpatialModel']
        spatial_width = src['SpatialWidth']
        xpix, ypix = wcs_utils.skydir_to_pix(skydir, self._skywcs)
        rebin = min(int(np.ceil(self.binsz / 0.01)), 8)
        shape_out = (self.enumbins + 1, self.npix, self.npix)
        cache = SourceMapCache.create(self._psf, spatial_model,
                                      spatial_width, shape_out,
                                      self.config['binning']['binsz'],
                                      rebin=rebin)
        self._srcmap_cache[name] = cache

    def _create_srcmap(self, name, src, **kwargs):
        """Generate the source map for a source."""

        psf_scale_fn = kwargs.get('psf_scale_fn', None)
        skydir = src.skydir
        spatial_model = src['SpatialModel']
        spatial_width = src['SpatialWidth']
        xpix, ypix = wcs_utils.skydir_to_pix(skydir, self._skywcs)
        rebin = min(int(np.ceil(self.binsz / 0.01)), 8)
        cache = self._srcmap_cache.get(name, None)
        if cache is not None:
            k = cache.create_map([ypix, xpix])
        else:
            k = srcmap_utils.make_srcmap(self._psf, spatial_model,
                                         spatial_width,
                                         npix=self.npix, xpix=xpix, ypix=ypix,
                                         cdelt=self.config['binning']['binsz'],
                                         rebin=rebin,
                                         psf_scale_fn=psf_scale_fn)

        return k

    def _update_srcmap(self, name, src, **kwargs):
        """Update the source map for an existing source in memory."""

        k = self._create_srcmap(name, src, **kwargs)
        scale = self._src_expscale.get(name, 1.0)
        k *= scale

        # Force the source map to be cached
        self.like.logLike.sourceMap(str(name)).model()
        self.like.logLike.setSourceMapImage(str(name), np.ravel(k))

        normPar = self.like.normPar(name)
        if not normPar.isFree():
            self.like.logLike.buildFixedModelWts()

    def generate_model(self, model_name=None, outfile=None):
        """Generate a counts model map from an XML model file using
        gtmodel.

        Parameters
        ----------
        model_name : str
            Name of the model.  If no name is given it will use the
            baseline model.

        outfile : str
            Override the name of the output model file.
        """

        if model_name is not None:
            model_name = os.path.splitext(model_name)[0]

        if model_name is None or model_name == '':
            srcmdl = self.files['srcmdl']
        else:
            srcmdl = self.get_model_path(model_name)

        if not os.path.isfile(srcmdl):
            raise Exception("Model file does not exist: %s", srcmdl)

        if model_name is None:
            suffix = self.config['file_suffix']
        else:
            suffix = '_%s%s' % (model_name, self.config['file_suffix'])

        outfile = os.path.join(self.config['fileio']['workdir'],
                               'mcube%s.fits' % (suffix))

        # May consider generating a custom source model file
        if not os.path.isfile(outfile):

            kw = dict(srcmaps=self.files['srcmap'],
                      srcmdl=srcmdl,
                      bexpmap=self.files['bexpmap'],
                      outfile=outfile,
                      expcube=self.files['ltcube'],
                      irfs=self.config['gtlike']['irfs'],
                      evtype=self.config['selection']['evtype'],
                      edisp=bool(self.config['gtlike']['edisp']),
                      outtype='ccube',
                      chatter=self.config['logging']['chatter'])

            run_gtapp('gtmodel', self.logger, kw)
        else:
            self.logger.info('Skipping gtmodel')

    def load_xml(self, xmlfile):

        xmlfile = self.get_model_path(xmlfile)
        self.logger.info('Loading %s' % xmlfile)
        self.like.logLike.reReadXml(str(xmlfile))
        if not self.like.logLike.fixedModelUpdated():
            self.like.logLike.buildFixedModelWts()

    def write_xml(self, xmlfile):
        """Write the XML model for this analysis component."""

        xmlfile = self.get_model_path(xmlfile)
        self.logger.info('Writing %s...', xmlfile)
        self.like.writeXml(str(xmlfile))

    def get_model_path(self, name):
        """Infer the path to the XML model name."""

        name, ext = os.path.splitext(name)
        ext = '.xml'
        xmlfile = name + self.config['file_suffix'] + ext
        xmlfile = utils.resolve_path(xmlfile,
                                     workdir=self.config['fileio']['workdir'])

        return xmlfile

    def _tscube_app(self, xmlfile):
        """Run gttscube as an application."""

        xmlfile = self.get_model_path(xmlfile)

        outfile = os.path.join(self.config['fileio']['workdir'],
                               'tscube%s.fits' % (self.config['file_suffix']))

        kw = dict(cmap=self.files['ccube'],
                  expcube=self.files['ltcube'],
                  bexpmap=self.files['bexpmap'],
                  irfs=self.config['gtlike']['irfs'],
                  evtype=self.config['selection']['evtype'],
                  srcmdl=xmlfile,
                  nxpix=self.npix, nypix=self.npix,
                  binsz=self.config['binning']['binsz'],
                  xref=float(self.roi.skydir.ra.deg),
                  yref=float(self.roi.skydir.dec.deg),
                  proj=self.config['binning']['proj'],
                  stlevel=0,
                  coordsys=self.config['binning']['coordsys'],
                  outfile=outfile)

        run_gtapp('gttscube', self.logger, kw)
