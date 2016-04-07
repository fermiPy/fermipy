import os
import copy
import shutil
import collections
import yaml
import numpy as np
import tempfile
import logging
import scipy
import scipy.optimize
import scipy.ndimage
from scipy.interpolate import UnivariateSpline
# pyLikelihood needs to be imported before astropy to avoid CFITSIO
# header error
import pyLikelihood as pyLike
import astropy.io.fits as pyfits
from astropy.coordinates import SkyCoord
import fermipy
import fermipy.defaults as defaults
import fermipy.utils as utils
import fermipy.fits_utils as fits_utils
import fermipy.plotting as plotting
import fermipy.irfs as irfs
import fermipy.sed as sed
import fermipy.sed as sourcefind
from fermipy.residmap import ResidMapGenerator
from fermipy.tsmap import TSMapGenerator, TSCubeGenerator
from fermipy.sourcefind import SourceFinder
from fermipy.utils import merge_dict, tolist, create_wcs
from fermipy.utils import Map
from fermipy.utils import create_hpx_disk_region_string
from fermipy.hpx_utils import HpxMap, HPX
from fermipy.roi_model import ROIModel, Model
from fermipy.logger import Logger
from fermipy.logger import logLevel as ll
# pylikelihood

import GtApp
import FluxDensity
from LikelihoodState import LikelihoodState
from gtutils import BinnedAnalysis, SummedLikelihood
import BinnedAnalysis as ba

norm_parameters = {
    'ConstantValue': ['Value'],
    'PowerLaw': ['Prefactor'],
    'PowerLaw2': ['Integral'],
    'BrokenPowerLaw': ['Prefactor'],
    'LogParabola': ['norm'],
    'PLSuperExpCutoff': ['Prefactor'],
    'ExpCutoff': ['Prefactor'],
    'FileFunction': ['Normalization'],
}

shape_parameters = {
    'ConstantValue': [],
    'PowerLaw': ['Index'],
    'PowerLaw2': ['Index'],
    'BrokenPowerLaw': ['Index1', 'Index2', 'BreakValue'],
    'LogParabola': ['alpha', 'beta'],
    'PLSuperExpCutoff': ['Index1', 'Cutoff'],
    'ExpCutoff': ['Index1', 'Cutoff'],
    'FileFunction': [],
}

index_parameters = {
    'ConstantValue': [],
    'PowerLaw': ['Index'],
    'PowerLaw2': ['Index'],
    'BrokenPowerLaw': ['Index1', 'Index2'],
    'LogParabola': ['alpha', 'beta'],
    'PLSuperExpCutoff': ['Index1', 'Index2'],
    'ExpCutoff': ['Index1'],
    'FileFunction': [],
}

def interpolate_function_min(x, y):
    sp = scipy.interpolate.splrep(x, y, k=2, s=0)

    def fn(t):
        return scipy.interpolate.splev(t, sp, der=1)

    if np.sign(fn(x[0])) == np.sign(fn(x[-1])):

        if np.sign(fn(x[0])) == -1:
            return x[-1]
        else:
            return x[0]

    x0 = scipy.optimize.brentq(fn,
                               x[0], x[-1],
                               xtol=1e-10 * np.median(x))

    return x0


def get_spectral_index(src,egy):
    """Compute the local spectral index of a source."""
    delta = 1E-5
    f0 = src.spectrum()(pyLike.dArg(egy*(1-delta)))
    f1 = src.spectrum()(pyLike.dArg(egy*(1+delta)))

    if f0 > 0 and f1 > 0:    
        gamma = np.log10(f0 / f1) / np.log10((1-delta)/(1+delta))
    else:
        gamma = np.nan

    return gamma


def run_gtapp(appname, logger, kw):
    logger.info('Running %s' % appname)
    filter_dict(kw, None)
    gtapp = GtApp.GtApp(appname)

    for k, v in kw.items():
        gtapp[k] = v

    logger.info(gtapp.command())
    stdin, stdout = gtapp.runWithOutput(print_command=False)

    for line in stdout:
        logger.info(line.strip())

        # Capture return code?


def filter_dict(d, val):
    for k, v in d.items():
        if v == val:
            del d[k]


def gtlike_spectrum_to_dict(spectrum):
    """ Convert a pyLikelihood object to a python dictionary which can
        be easily saved to a file."""
    parameters = pyLike.ParameterVector()
    spectrum.getParams(parameters)
    d = dict(spectrum_type=spectrum.genericName())
    for p in parameters:

        pname = p.getName()
        pval = p.getTrueValue()
        perr = abs(p.error() * p.getScale()) if p.isFree() else np.nan
        d[pname] = np.array([pval, perr])

        if d['spectrum_type'] == 'FileFunction':
            ff = pyLike.FileFunction_cast(spectrum)
            d['file'] = ff.filename()
    return d


def resolve_path(path, workdir=None):
    if os.path.isabs(path):
        return path
    elif workdir is None:
        return os.path.abspath(path)
    else:
        return os.path.join(workdir, path)


def load_roi_data(infile, workdir=None):
    infile = resolve_path(infile, workdir=workdir)
    infile, ext = os.path.splitext(infile)

    if os.path.isfile(infile + '.npy'):
        infile += '.npy'
    elif os.path.isfile(infile + '.yaml'):
        infile += '.yaml'
    else:
        raise Exception('Input file does not exist.')

    ext = os.path.splitext(infile)[1]

    if ext == '.npy':
        return infile, load_npy(infile)
    elif ext == '.yaml':
        return infile, load_yaml(infile)
    else:
        raise Exception('Unrecognized extension.')


def load_yaml(infile):
    return yaml.load(open(infile))


def load_npy(infile):
    return np.load(infile).flat[0]


class GTAnalysis(fermipy.config.Configurable,sed.SEDGenerator):
    """High-level analysis interface that internally manages a set of
    analysis component objects.  Most of the functionality of the
    fermiPy package is provided through the methods of this class.
    The class constructor accepts a dictionary that defines the
    configuration for the analysis.  Keyword arguments provided in
    **kwargs can be used to override parameter values in the
    configuration dictionary."""

    defaults = {'logging': defaults.logging,
                'fileio': defaults.fileio,
                'optimizer': defaults.optimizer,
                'binning': defaults.binning,
                'selection': defaults.selection,
                'model': defaults.model,
                'data': defaults.data,
                'gtlike': defaults.gtlike,
                'mc': defaults.mc,
                'residmap': defaults.residmap,
                'tsmap': defaults.tsmap,
                'tscube': defaults.tscube,
                'sourcefind': defaults.sourcefind,
                'sed': defaults.sed,
                'extension': defaults.extension,
                'localize': defaults.localize,
                'roiopt': defaults.roiopt,
                'plotting': defaults.plotting,
                'components': (None, '', list)}

    def __init__(self, config, **kwargs):

        super(GTAnalysis, self).__init__(config, validate=True,**kwargs)

        self._projtype = self.config['binning']['projtype']

        # Set random seed
        np.random.seed(self.config['mc']['seed'])
        
        # Setup directories
        self._rootdir = os.getcwd()

        # Destination directory for output data products
        if self.config['fileio']['outdir'] is not None:
            self._savedir = os.path.join(self._rootdir,
                                         self.config['fileio']['outdir'])
            utils.mkdir(self._savedir)
        else:
            raise Exception('Save directory not defined.')

        # put pfiles into savedir
        os.environ['PFILES'] = \
            self._savedir + ';' + os.environ['PFILES'].split(';')[-1]

        if self.config['fileio']['logfile'] is None:
            self._config['fileio']['logfile'] = os.path.join(self._savedir,
                                                             'fermipy')

        self.logger = Logger.get(self.__class__.__name__,
                                 self.config['fileio']['logfile'],
                                 ll(self.config['logging']['verbosity']))

        self.logger.info('\n' + '-' * 80 + '\n' + "This is fermipy version {}.".
                         format(fermipy.__version__))
        self.print_config(self.logger, loglevel=logging.DEBUG)

        # Working directory (can be the same as savedir)
        if self.config['fileio']['usescratch']:
            self._config['fileio']['workdir'] = tempfile.mkdtemp(
                prefix=os.environ['USER'] + '.',
                dir=self.config['fileio']['scratchdir'])
            self.logger.info(
                'Created working directory: %s' % self.config['fileio'][
                    'workdir'])
            self.stage_input()
        else:
            self._config['fileio']['workdir'] = self._savedir

        if 'FERMIPY_WORKDIR' not in os.environ:
            os.environ['FERMIPY_WORKDIR'] = self.config['fileio']['workdir']

        # Create Plotter
        self._plotter = plotting.AnalysisPlotter(self.config['plotting'],
                                                 fileio=self.config['fileio'],
                                                 logging=self.config['logging'])
            
        # Setup the ROI definition
        self._roi = ROIModel.create(self.config['selection'],
                                    self.config['model'],
                                    fileio=self.config['fileio'],
                                    logfile=self.config['fileio']['logfile'],
                                    logging=self.config['logging'],
                                    coordsys=self.config['binning']['coordsys'])

        self._like = None
        self._components = []
        configs = self._create_component_configs()

        for cfg in configs:
            comp = self._create_component(cfg)
            self._components.append(comp)

        for c in self.components:
            for s in c.roi.sources:
                if s.name not in self.roi:
                    self.roi.load_source(s)
            
        energies = np.zeros(0)
        roiwidths = np.zeros(0)
        binsz = np.zeros(0)
        for c in self.components:
            energies = np.concatenate((energies, c.energies))
            roiwidths = np.insert(roiwidths, 0, c.roiwidth)
            binsz = np.insert(binsz, 0, c.binsz)

        self._ebin_edges = np.sort(np.unique(energies.round(5)))
        self._enumbins = len(self._ebin_edges) - 1
        self._erange = np.array([self._ebin_edges[0],
                                 self._ebin_edges[-1]])
        
        self._roi_model = {
            'logLike': np.nan,
            'Npred': 0.0,
            'counts': np.zeros(self.enumbins),
            'model_counts': np.zeros(self.enumbins),
            'energies': np.copy(self.energies),
            'erange': np.copy(self.erange),
            'components': []
        }

        for c in self._components:
            comp_model = [{'logLike': np.nan,
                           'Npred': 0.0,
                           'counts': np.zeros(c.enumbins),
                           'model_counts': np.zeros(c.enumbins),
                           'energies': np.copy(c.energies)}]

            self._roi_model['components'] += comp_model

        self._roiwidth = max(roiwidths)
        self._binsz = min(binsz)
        self._npix = int(np.round(self._roiwidth / self._binsz))

        if self.projtype == 'HPX':
            self._hpx_region = create_hpx_disk_region_string(self._roi.skydir,
                                                             coordsys=
                                                             self.config[
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
                                    self._ebin_edges)

        else:
            self._skywcs = create_wcs(self._roi.skydir,
                                      coordsys=self.config['binning'][
                                          'coordsys'],
                                      projection=self.config['binning']['proj'],
                                      cdelt=self._binsz,
                                      crpix=1.0 + 0.5 * (self._npix - 1),
                                      naxis=2)

            self._proj = create_wcs(self._roi.skydir,
                                    coordsys=self.config['binning']['coordsys'],
                                    projection=self.config['binning']['proj'],
                                    cdelt=self._binsz,
                                    crpix=1.0 + 0.5 * (self._npix - 1),
                                    naxis=3,
                                    energies=self.energies)
            """
            self._proj.wcs.crpix[2] = 1
            self._proj.wcs.crval[2] = 10 ** self.energies[0]
            self._proj.wcs.cdelt[2] = 10 ** self.energies[1] - 10 ** \
                                                               self.energies[0]
            self._proj.wcs.ctype[2] = 'Energy'
            """

    def __del__(self):
        self.stage_output()
        self.cleanup()

    @property
    def workdir(self):
        """Return the analysis working directory."""
        return self.config['fileio']['workdir']
        
    @property
    def roi(self):
        """Return the ROI object."""
        return self._roi

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
        """Return the energy bin edges."""
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
    def erange(self):
        return self._erange
    
    @property
    def projtype(self):
        """Return the type of projection to use"""
        return self._projtype

    @staticmethod
    def create(infile, config=None):
        """Create a new instance of GTAnalysis from an analysis output file
        generated with `~fermipy.GTAnalysis.write_roi`.  By default
        the new instance will inherit the configuration of the
        saved analysis instance.  The configuration may be overriden
        by passing a config file path with the ``config`` argument.

        Parameters
        ----------

        infile : str
            Path to the ROI results file.

        config : str
            Path to a configuration file.  This will override the
            configuration in the ROI results file.

        """

        infile = os.path.abspath(infile)
        roi_file, roi_data = load_roi_data(infile)

        if config is None:
            config = roi_data['config']

        gta = GTAnalysis(config)
        gta.setup(init_sources=False)
        gta.load_roi(infile)
        return gta

    def set_log_level(self, level):
        self.logger.handlers[1].setLevel(level)
        for c in self.components:
            c.logger.handlers[1].setLevel(level)

    def _update_roi(self):

        rm = self._roi_model

        rm['model_counts'].fill(0)
        rm['Npred'] = 0
        for i, c in enumerate(self.components):
            rm['components'][i]['model_counts'].fill(0)
            rm['components'][i]['Npred'] = 0

        for name in self.like.sourceNames():

            src = self.roi.get_source_by_name(name, True)
            rm['model_counts'] += src['model_counts']
            rm['Npred'] += np.sum(src['model_counts'])
            mc = self.model_counts_spectrum(name)

            for i, c in enumerate(self.components):
                rm['components'][i]['model_counts'] += mc[i]
                rm['components'][i]['Npred'] += np.sum(mc[i])

    def _update_srcmap(self, name, skydir, spatial_model, spatial_width):

        for c in self.components:
            c._update_srcmap(name, skydir, spatial_model, spatial_width)

    def reload_source(self, name):
        """Delete and reload a source in the model.  This will refresh
        the spatial model of this source to the one defined in the XML
        model."""
        
        for c in self.components:
            c.reload_source(name)

        self._init_source(name)
            
        self.like.model = self.like.components[0].model
        
    def add_source(self, name, src_dict, free=False, init_source=True,
                   save_source_maps=True, **kwargs):
        """Add a source to the ROI model.  This function may be called
        either before or after setup().

        Parameters
        ----------

        name : str
            Source name.

        src_dict : dict or `~fermipy.roi_model.Source` object
            Dictionary or source object defining the source properties
            (coordinates, spectral parameters, etc.).

        free : bool
            Initialize the source with a free normalization parameter.

        """

        if self.roi.has_source(name):
            msg = 'Source %s already exists.' % name
            self.logger.error(msg)
            raise Exception(msg)

        loglevel=kwargs.pop('loglevel',logging.INFO)
        
        self.logger.log(loglevel,'Adding source ' + name)

        src = self.roi.create_source(name,src_dict)

        for c in self.components:
            c.add_source(name, src_dict, free=free,
                         save_source_maps=save_source_maps)

        if self._like is None:
            return

        if self.config['gtlike']['edisp'] and src.name not in \
                self.config['gtlike']['edisp_disable']:
            self.set_edisp_flag(src.name, True)

        self.like.syncSrcParams(str(name))
        self.like.model = self.like.components[0].model

        if init_source:
            self._init_source(name)

    def delete_source(self, name, save_template=True, delete_source_map=False,
                      **kwargs):
        """Delete a source from the model.

        Parameters
        ----------

        name : str
            Source name.

        Returns
        -------    
        src : `~fermipy.roi_model.Model`
            The deleted source object.

        """

        if not self.roi.has_source(name):
            self.logger.error('No source with name: %s' % name)
            return

        loglevel=kwargs.pop('loglevel',logging.INFO)
        
        self.logger.log(loglevel,'Deleting source %s' % name)

        # STs require a source to be freed before deletion
        normPar = self.like.normPar(name)
        if not normPar.isFree():
            self.free_norm(name)

        for c in self.components:
            c.delete_source(name, save_template=save_template,
                            delete_source_map=delete_source_map)

        src = self.roi.get_source_by_name(name, True)
        self.roi.delete_sources([src])
        self.like.model = self.like.components[0].model

        return src

    def _create_component_configs(self):
        configs = []

        components = self.config['components']

        common_config = GTBinnedAnalysis.get_config()
        common_config = merge_dict(common_config, self.config)

        if components is None or len(components) == 0:
            cfg = copy.copy(common_config)
            cfg['file_suffix'] = '_00'
            cfg['name'] = '00'
            configs.append(cfg)
        elif isinstance(components, dict):
            for i, k in enumerate(sorted(components.keys())):
                cfg = copy.copy(common_config)
                cfg = merge_dict(cfg, components[k])
                cfg['file_suffix'] = '_' + k
                cfg['name'] = k
                configs.append(cfg)
        elif isinstance(components, list):
            for i, c in enumerate(components):
                cfg = copy.copy(common_config)
                cfg = merge_dict(cfg, c)
                cfg['file_suffix'] = '_%02i' % i
                cfg['name'] = '%02i' % i
                configs.append(cfg)
        else:
            raise Exception('Invalid type for component block.')

        return configs

    def _create_component(self, cfg):

        self.logger.debug("Creating Analysis Component: " + cfg['name'])

        cfg['fileio']['workdir'] = self.config['fileio']['workdir']

        comp = GTBinnedAnalysis(cfg, logging=self.config['logging'])

        return comp

    def stage_output(self):
        """Copy data products to final output directory."""

        extensions = ['.xml', '.par', '.yaml', '.png', '.pdf', '.npy']
        if self.config['fileio']['savefits']:
            extensions += ['.fits', '.fit']

        if self.config['fileio']['workdir'] == self._savedir:
            return
        elif os.path.isdir(self.config['fileio']['workdir']):
            self.logger.info('Staging files to %s' % self._savedir)
            for f in os.listdir(self.config['fileio']['workdir']):

                if not os.path.splitext(f)[1] in extensions: continue

                self.logger.info('Copying ' + f)
                shutil.copy(os.path.join(self.config['fileio']['workdir'], f),
                            self._savedir)

        else:
            self.logger.error('Working directory does not exist.')

    def stage_input(self):
        """Copy data products to intermediate working directory."""

        extensions = ['.fits', '.fit', '.xml', '.npy']

        if self.config['fileio']['workdir'] == self._savedir:
            return
        elif os.path.isdir(self.config['fileio']['workdir']):
            self.logger.info('Staging files to %s' %
                             self.config['fileio']['workdir'])
            #            for f in glob.glob(os.path.join(self._savedir,'*')):
            for f in os.listdir(self._savedir):
                if not os.path.splitext(f)[1] in extensions: continue
                self.logger.debug('Copying ' + f)
                shutil.copy(os.path.join(self._savedir, f),
                            self.config['fileio']['workdir'])
        else:
            self.logger.error('Working directory does not exist.')

    def setup(self, init_sources=True, overwrite=False):
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

        self.logger.info('Running setup')

        # Run data selection step

        self._like = SummedLikelihood()
        for i, c in enumerate(self._components):
            c.setup(overwrite=overwrite)
            self._like.addComponent(c.like)

        self._ccube_file = os.path.join(self.config['fileio']['workdir'],
                                        'ccube.fits')

        self._init_roi_model()

        if init_sources:

            self.logger.info('Initializing source properties')
            for name in self.like.sourceNames():
                self._init_source(name)

            self._update_roi()

        self.logger.info('Finished setup')

    def _create_likelihood(self, srcmdl):
        self._like = SummedLikelihood()
        for c in self.components:
            c._create_binned_analysis(srcmdl)
            self._like.addComponent(c.like)
            
        self.like.model = self.like.components[0].model
        self._init_roi_model()
        
    def _init_roi_model(self):

        rm = self._roi_model

        rm['counts'] = np.zeros(self.enumbins)
        rm['logLike'] = -self.like()

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
            rm['components'][i]['logLike'] = c.like()

        if proj_type == 0:
            shape = (self.enumbins, self.npix, self.npix)
        elif proj_type == 1:
            shape = (self.enumbins, self._proj.npix)

        self._coadd_maps(cmaps, shape, rm)

    def _init_source(self, name):
        
        src = self.roi.get_source_by_name(name, True)
        src.update_data({'sed': None, 'extension': None,
                         'localize': None,
                         'class': None})

        if 'CLASS1' in src['catalog']:
            src['class'] = src['catalog']['CLASS1'].strip()

        src.update_data(self.get_src_model(name, False))
        return src

    def cleanup(self):

        if self.config['fileio']['workdir'] == self._savedir:
            return
        elif os.path.isdir(self.config['fileio']['workdir']):
            self.logger.info('Deleting working directory: ' +
                             self.config['fileio']['workdir'])
            shutil.rmtree(self.config['fileio']['workdir'])

    def generate_model(self, model_name=None):
        """Generate model maps for all components.  model_name should
        be a unique identifier for the model.  If model_name is None
        then the model maps will be generated using the current
        parameters of the ROI."""

        for i, c in enumerate(self._components):
            c.generate_model(model_name=model_name)

            # If all model maps have the same spatial/energy binning we
            # could generate a co-added model map here

    def setEnergyRange(self, emin, emax):
        """Set the energy bounds of the analysis.  This restricts the
        evaluation of the likelihood to the data that falls in this
        range.  Input values will be rounded to the closest bin edge
        value.  If either argument is None then the lower or upper
        bound of the analysis instance will be used.

        Parameters
        ----------

        emin : float
           Lower energy bound in log10(E/MeV).

        emax : float
           Upper energy bound in log10(E/MeV).

        Returns
        -------

        eminmax : array
           Minimum and maximum energy.

        """

        if emin is None:
            emin = self.energies[0]
        else:
            imin = int(utils.val_to_edge(self.energies, emin)[0])
            emin = self.energies[imin]
            
        if emax is None:
            emax = self.energies[-1]
        else:
            imax = int(utils.val_to_edge(self.energies, emax)[0])
            emax = self.energies[imax]

        erange = np.array([emin,emax])
        
        if np.allclose(erange,self._erange):
            return self._erange
        
        self._erange = np.array([emin,emax])
        self._roi_model['erange'] = np.copy(self.erange)
        for c in self.components:
            c.setEnergyRange(emin, emax)

        return self._erange

    def counts_map(self):
        """Return a `~fermipy.utils.Map` representation of the counts map.

        Returns
        -------

        map : `~fermipy.utils.Map`

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

        map : `~fermipy.utils.Map`
        """

        maps = [c.model_counts_map(name, exclude) for c in self.components]
       
        if self.projtype == "HPX":
            shape = (self.enumbins, self._proj.npix)
            cmap = fits_utils.make_coadd_map(maps, self._proj, shape)
        elif self.projtype == "WCS":
            shape = (self.enumbins, self.npix, self.npix)
            cmap = fits_utils.make_coadd_map(maps, self._proj, shape)
        else:
            raise Exception(
                "Did not recognize projection type %s" % self.projtype)
        return cmap

    def model_counts_spectrum(self, name, emin=None, emax=None, summed=False):
        """Return the predicted number of model counts versus energy
        for a given source and energy range.  If summed=True return
        the counts spectrum summed over all components otherwise
        return a list of model spectra."""

        if emin is None:
            emin = self.energies[0]
        if emax is None:
            emax = self.energies[-1]

        if summed:
            cs = np.zeros(self.enumbins)
            imin = utils.val_to_bin_bounded(self.energies, emin + 1E-7)[0]
            imax = utils.val_to_bin_bounded(self.energies, emax - 1E-7)[0] + 1

            for c in self.components:
                ecenter = 0.5 * (c.energies[:-1] + c.energies[1:])
                counts = c.model_counts_spectrum(name, self.energies[0],
                                                 self.energies[-1])

                cs += np.histogram(ecenter,
                                   weights=counts,
                                   bins=self.energies)[0]

            return cs[imin:imax]
        else:
            cs = []
            for c in self.components:
                cs += [c.model_counts_spectrum(name, emin, emax)]
            return cs

    def get_sources(self, cuts=None, distance=None, minmax_ts=None, minmax_npred=None,
                    square=False):
        """Retrieve list of sources in the ROI model satisfying the
        given selections.

        Returns
        -------

        srcs : list 
            A list of `~fermipy.roi_model.Model` objects.
        """

        return self.roi.get_sources(cuts,distance,
                                    minmax_ts,minmax_npred,square)

    def delete_sources(self, cuts=None, distance=None,
                       minmax_ts=None, minmax_npred=None,
                       square=False):
        """Delete sources in the ROI model satisfying the given
        selection criteria.

        Returns
        -------

        srcs : list 
            A list of `~fermipy.roi_model.Model` objects.

        """

        srcs = self.get_sources(cuts, distance,
                                minmax_ts,minmax_npred,
                                square=square)
        self._roi.delete_sources(srcs)
        for c in self.components:
            c.delete_sources(srcs)

        return srcs

    def free_sources(self, free=True, pars=None, cuts=None,
                     distance=None, minmax_ts=None, minmax_npred=None, 
                     square=False, exclude_diffuse=False):
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
            Set a list of parameters to be freed/fixed for this
            source.  If none then all source parameters will be
            freed/fixed.  If pars='norm' then only normalization
            parameters will be freed.

        cuts : dict
            Dictionary of [min,max] selections on source properties.

        distance : float        
            Distance out to which sources should be freed or fixed.
            If this parameter is none no selection will be applied.

        minmax_ts : list
            Free sources that have TS in the range [min,max].  If
            either min or max are None then only a lower (upper) bound
            will be applied.  If this parameter is none no selection
            will be applied.

        minmax_npred : list        
            Free sources that have Npred in the range [min,max].  If
            either min or max are None then only a lower (upper) bound
            will be applied.  If this parameter is none no selection
            will be applied.
                    
        square : bool
            Switch between applying a circular or square (ROI-like)
            selection on the maximum projected distance from the ROI
            center.

        exclude_diffuse : bool
            Exclude diffuse sources.
            
        Returns
        -------

        srcs : list 
            A list of `~fermipy.roi_model.Model` objects.

        
        """

        srcs = self.roi.get_sources(cuts, distance,
                                    minmax_ts,minmax_npred,
                                    square=square,exclude_diffuse=exclude_diffuse,
                                    coordsys=self.config['binning']['coordsys'])

        for s in srcs:
            self.free_source(s.name, free=free, pars=pars)

        return srcs

    def free_sources_by_position(self, free=True, pars=None,
                                 distance=None, square=False):
        """Free/Fix all sources within a certain distance of the given sky
        coordinate.  By default it will use the ROI center.

        Parameters
        ----------

        free : bool        
            Choose whether to free (free=True) or fix (free=False)
            source parameters.

        pars : list        
            Set a list of parameters to be freed/fixed for this
            source.  If none then all source parameters will be
            freed/fixed.  If pars='norm' then only normalization
            parameters will be freed.

        distance : float        
            Distance in degrees out to which sources should be freed
            or fixed.  If none then all sources will be selected.

        square : bool        
            Apply a square (ROI-like) selection on the maximum distance in
            either X or Y in projected cartesian coordinates.   

        Returns
        -------

        srcs : list 
            A list of `~fermipy.roi_model.Source` objects.
     
        """

        return self.free_sources(free, pars, cuts=None, distance=distance,
                                 square=square)

    def set_edisp_flag(self, name, flag=True):
        """Enable or disable the energy dispersion correction for the
        given source."""
        src = self.roi.get_source_by_name(name, True)
        name = src.name

        for c in self.components:
            c.like[name].src.set_edisp_flag(flag)

    def scale_parameter(self, name, par, scale):

        idx = self.like.par_index(name, par)
        self.like[idx].setScale(self.like[idx].getScale() * scale)

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
        name = self.roi.get_source_by_name(name, True).name
        idx = self.like.par_index(name, par)
        current_bounds = list(self.like.model[idx].getBounds())

        if scale is not None:
            self.like[idx].setScale(scale)
        else:
            scale = self.like.model[idx].getScale()
        
        if true_value:
            current_bounds[0] = min(current_bounds[0],value/scale)
            current_bounds[1] = max(current_bounds[1],value/scale)
        else:
            current_bounds[0] = min(current_bounds[0],value)
            current_bounds[1] = max(current_bounds[1],value)

        # update current bounds to encompass new value
        self.like[idx].setBounds(*current_bounds)
        
        if true_value:
            for p in self.like[idx].pars:
                p.setTrueValue(value)
        else:
            self.like[idx].setValue(value)

        if bounds is not None:
            self.like[idx].setBounds(*bounds)

        self.like.syncSrcParams(str(name))
            
        if update_source:
            self.update_source(name)

    def set_parameter_scale(self,name,par,scale):
        """Update the scale of a parameter while keeping its value constant."""
        name = self.roi.get_source_by_name(name, True).name
        idx = self.like.par_index(name, par)
        current_bounds = list(self.like.model[idx].getBounds())
        current_scale = self.like.model[idx].getScale()
        current_value = self.like[idx].getValue()

        self.like[idx].setScale(scale)
        self.like[idx].setValue(current_value*current_scale/scale)
        self.like[idx].setBounds(current_bounds[0]*current_scale/scale,
                                 current_bounds[1]*current_scale/scale)
        
            
    def set_parameter_bounds(self,name,par,bounds):
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

    def free_parameter(self, name, par, free=True):
        idx = self.like.par_index(name, par)
        self.like[idx].setFree(free)
        self.like.syncSrcParams()

    def free_source(self, name, free=True, pars=None):
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

        # Find the source
        src = self.roi.get_source_by_name(name, True)
        name = src.name

        if pars is None:
            pars = []
            pars += norm_parameters[src['SpectrumType']]
            pars += shape_parameters[src['SpectrumType']]
        elif pars == 'norm':
            pars = []
            pars += norm_parameters[src['SpectrumType']]
        elif pars == 'shape':
            pars = []
            pars += shape_parameters[src['SpectrumType']]
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
            self.logger.debug('Freeing parameters for %-22s: %s'
                              % (name, par_names))
        else:
            self.logger.debug('Fixing parameters for %-22s: %s'
                              % (name, par_names))

        for (idx, par_name) in zip(par_indices, par_names):
            self.like[idx].setFree(free)
        self.like.syncSrcParams(str(name))

    #        freePars = self.like.freePars(name)
    #        if not free:
    #            self.like.setFreeFlag(name, freePars, False)
    #        else:
    #            self.like[idx].setFree(True)

    def set_norm_scale(self, name, value):
        name = self.get_source_name(name)
        normPar = self.like.normPar(name)
        normPar.setScale(value)
        self.like.syncSrcParams(str(name))
    
    def set_norm(self, name, value):
        name = self.get_source_name(name)
        normPar = self.like.normPar(name)
        normPar.setValue(value)
        self.like.syncSrcParams(str(name))

    def free_norm(self, name, free=True):
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
        self.free_source(name, pars=[normPar], free=free)

    def free_index(self, name, free=True):
        """Free/Fix index of a source.

        Parameters
        ----------

        name : str
            Source name.

        free : bool        
            Choose whether to free (free=True) or fix (free=False).

        """
        src = self.roi.get_source_by_name(name, True)
        self.free_source(name, free=free,
                         pars=index_parameters[src['SpectrumType']])

    def free_shape(self, name, free=True):
        """Free/Fix shape parameters of a source.

        Parameters
        ----------

        name : str
            Source name.

        free : bool        
            Choose whether to free (free=True) or fix (free=False).
        """
        src = self.roi.get_source_by_name(name, True)
        self.free_source(name, free=free,
                         pars=shape_parameters[src['SpectrumType']])

    def get_free_params(self):

        params = {}
        for srcName in self.like.sourceNames():
            for parName in self.get_free_source_params(srcName):
                idx = self.like.par_index(srcName, parName)
                par = self.like.model[idx]
                bounds = par.getBounds()
                
                is_norm = parName == self.like.normPar(srcName).getName()
                
                params[idx] = {'src_name' : srcName,
                               'par_name' : parName,
                               'value' : par.getValue(),
                               'error' : par.error(),
                               'scale' : par.getScale(),
                               'idx' : idx,
                               'is_free' : par.isFree(),
                               'is_norm' : is_norm,
                               'bounds' : bounds }

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

    def _latch_free_params(self):
        self._free_params = self.get_free_param_vector()

    def _restore_free_params(self):
        self.set_free_param_vector(self._free_params)
                
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
            name = self.roi.get_source_by_name(name, True).name
        return name

    def zero_source(self, name):
        normPar = self.like.normPar(name).getName()
        self.scale_parameter(name, normPar, 1E-10)
        self.free_source(name, free=False)
        self.like.syncSrcParams(str(name))

    def unzero_source(self, name):
        normPar = self.like.normPar(name).getName()
        self.scale_parameter(name, normPar, 1E10)
        self.like.syncSrcParams(str(name))

    def residmap(self, prefix='', **kwargs):
        """Generate 2-D spatial residual maps using the current ROI
        model and the convolution kernel defined with the `model`
        argument.

        Parameters
        ----------

        prefix : str
            String that will be prefixed to the output residual map files.

        model : dict
           Dictionary defining the properties of the convolution kernel.

        exclude : str or list of str
            Source or sources that will be removed from the model when
            computing the residual map.

        erange : list
           Restrict the analysis to an energy range (emin,emax) in
           log10(E/MeV) that is a subset of the analysis energy range.
           By default the full analysis energy range will be used.  If
           either emin/emax are None then only an upper/lower bound on
           the energy range wil be applied.    

        make_plots : bool        
            Write image files.

        make_fits : bool
            Write FITS files.

        Returns
        -------

        maps : dict
           A dictionary containing the `~fermipy.utils.Map` objects
           for the residual significance and amplitude.    

        """

        self.logger.info('Generating residual maps')

        make_plots = kwargs.get('make_plots', True)
        rmg = ResidMapGenerator(self.config['residmap'], 
                                fileio=self.config['fileio'],
                                logging=self.config['logging'])

        model = kwargs.get('model',self.config['residmap']['model'])

        maps = rmg.make_residual_map(self,prefix,copy.deepcopy(model),**kwargs)

        for m in maps if isinstance(maps,list) else [maps]:
            if make_plots:
                plotter = plotting.AnalysisPlotter(self.config['plotting'],
                                                   fileio=self.config['fileio'],
                                                   logging=self.config[
                                                       'logging'])

                plotter.make_residual_plots(self, m)

        self.logger.info('Finished residual maps')
                
        return maps

    def optimize(self, **kwargs):
        """Iteratively optimize the ROI model.  The optimization is
        performed in three sequential steps:
        
        * Free the normalization of the N largest components (as
          determined from NPred) that contain a fraction ``npred_frac``
          of the total predicted counts in the model and perform a
          simultaneous fit of the normalization parameters of these
          components.

        * Individually fit the normalizations of all sources that were
          not included in the first step in order of their Npred
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

        """

        self.logger.info('Starting')

        logLike0 = -self.like()
        self.logger.debug('LogLike: %f' % logLike0)

        # Extract options from kwargs
        npred_frac_threshold = kwargs.get('npred_frac',
                                          self.config['roiopt']['npred_frac'])
        npred_threshold = kwargs.get('npred_threshold',
                                     self.config['roiopt']['npred_threshold'])
        shape_ts_threshold = kwargs.get('shape_ts_threshold',
                                        self.config['roiopt'][
                                            'shape_ts_threshold'])

        # preserve free parameters
        free = self.get_free_param_vector()

        # Fix all parameters
        self.free_sources(free=False)

        # Free norms of sources for which the sum of Npred is a
        # fraction > npred_frac of the total model counts in the ROI
        npred_sum = 0
        skip_sources = []
        for s in sorted(self.roi.sources, key=lambda t: t['Npred'],
                        reverse=True):

            npred_sum += s['Npred']
            npred_frac = npred_sum / self._roi_model['Npred']
            self.free_norm(s.name)
            skip_sources.append(s.name)

            if npred_frac > npred_frac_threshold: break

        self.fit()
        self.free_sources(free=False)

        # Step through remaining sources and re-fit normalizations
        for s in sorted(self.roi.sources, key=lambda t: t['Npred'],
                        reverse=True):

            if s.name in skip_sources: continue

            if s['Npred'] < npred_threshold:
                self.logger.debug(
                    'Skipping %s with Npred %10.3f' % (s.name, s['Npred']))
                continue

            self.logger.debug('Fitting %s Npred: %10.3f TS: %10.3f' % (
                s.name, s['Npred'], s['ts']))
            self.free_norm(s.name)
            self.fit()
            self.logger.debug('Post-fit Results Npred: %10.3f TS: %10.3f' % (
                s['Npred'], s['ts']))
            self.free_norm(s.name, free=False)

            # Refit spectral shape parameters for sources with TS >
            # shape_ts_threshold
        for s in sorted(self.roi.sources,
                        key=lambda t: t['ts'] if np.isfinite(t['ts']) else 0,
                        reverse=True):

            if s['ts'] < shape_ts_threshold \
                    or not np.isfinite(s['ts']): continue

            self.logger.debug('Fitting shape %s TS: %10.3f' % (s.name, s['ts']))
            self.free_source(s.name)
            self.fit()
            self.free_source(s.name, free=False)

        self.set_free_param_vector(free)

        logLike1 = -self.like()
        self.logger.info('Finished')
        self.logger.info(
            'LogLike: %f Delta-LogLike: %f' % (logLike1, logLike1 - logLike0))

    def localize(self, name, **kwargs):
        """Find the best-fit position of a source.  Localization is
        performed in two steps.  First a TS map is computed centered
        on the source with half-width set by ``dtheta_max``.  A fit is
        then performed to the maximum TS peak in this map.  The source
        position is then further refined by scanning the likelihood in
        the vicinity of the peak found in the first step.  The size of
        the scan region is set to encompass the 99% positional
        uncertainty contour as determined from the peak fit.

        Parameters
        ----------

        name : str
            Source name.

        dtheta_max : float
            Maximum offset in RA/DEC in deg from the nominal source
            position that will be used to define the boundaries of the
            TS map search region.

        nstep : int        
            Number of steps in longitude/latitude that will be taken
            when refining the source position.  The bounds of the scan
            range are set to the 99% positional uncertainty as
            determined from the TS map peak fit.  The total number of
            sampling points will be nstep**2.

        fix_background : bool
            Fix background parameters when fitting the source position.

        update : bool
            Update the model for this source with the best-fit
            position.  If newname=None this will overwrite the
            existing source map of this source with one corresponding
            to its new location.

        newname : str
            Name that will be assigned to the relocalized source 
            when update=True.  If newname is None then the existing
            source name will be used.

        Returns
        -------

        localize : dict
            Dictionary containing results of the localization
            analysis.  This dictionary is also saved to the
            dictionary of this source in 'localize'.

        """

        name = self.roi.get_source_by_name(name, True).name

        # Extract options from kwargs
        config = copy.deepcopy(self.config['localize'])
        config.update(kwargs)
        config.setdefault('newname', name)

        nstep = config['nstep']
        dtheta_max = config['dtheta_max']
        update = config['update']
        newname = config['newname']
        prefix = kwargs.pop('prefix','')
        
        self.logger.info('Running localization for %s' % name)

        saved_state = LikelihoodState(self.like)

        src = self.roi.get_source_by_name(name, True)
        skydir = src.skydir
        skywcs = self._skywcs
        src_pix = skydir.to_pixel(skywcs)

        tsmap = self.tsmap(utils.join_strings([prefix,name.lower().replace(' ','_')]),
                           model=src.data,multithread=True,
                           map_skydir=skydir,
                           map_size=2.0*dtheta_max,
                           exclude=[name],make_plots=False)

        ix, iy = np.unravel_index(np.argmax(0.5*tsmap['ts'].counts),tsmap['ts'].counts.shape)        
        tsmap_fit = utils.fit_parabola(tsmap['ts'].counts, ix, iy, dpix=2)
                                     
        peak_skydir = SkyCoord.from_pixel(tsmap_fit['y0'],tsmap_fit['x0'],tsmap['ts'].wcs)
        peak_sigmax = 2.0**0.5*tsmap_fit['sigmax']*np.abs(tsmap['ts'].wcs.wcs.cdelt[0])
        peak_sigmay = 2.0**0.5*tsmap_fit['sigmay']*np.abs(tsmap['ts'].wcs.wcs.cdelt[1])
        peak_sigma = (peak_sigmax*peak_sigmay)**0.5
        peak_pix = peak_skydir.to_pixel(skywcs)
        peak_r68 = 2.30**0.5*peak_sigma
        peak_r95 = 5.99**0.5*peak_sigma
        peak_r99 = 9.21**0.5*peak_sigma        
        
        # Fit baseline (point-source) model
        self.free_norm(name)
        self.fit(update=False)

        # Save likelihood value for baseline fit
        logLike0 = -self.like()

        self.zero_source(name)

        o = {'config': config,
             'fit_success': True,
             'logLike_base': logLike0 }

        cdelt0 = np.abs(skywcs.wcs.cdelt[0])
        cdelt1 = np.abs(skywcs.wcs.cdelt[1])
        delta_pix = np.linspace(-peak_r99,peak_r99,nstep)/cdelt0
        scan_step = 2.0*peak_r99/(nstep-1.0)
        
        scan_xpix = delta_pix+peak_pix[0]
        scan_ypix = delta_pix+peak_pix[1]
              
        scan_skydir = SkyCoord.from_pixel(np.ravel(np.ones((nstep, nstep)) * scan_xpix[:,np.newaxis]),
                                          np.ravel(np.ones((nstep, nstep)) * scan_ypix[np.newaxis,:]),
                                          skywcs)
                
        lnlscan = dict(xpix=scan_xpix,
                       ypix=scan_ypix,
                       logLike=np.zeros((nstep, nstep)),
                       dlogLike=np.zeros((nstep, nstep)),
                       dlogLike_fit=np.zeros((nstep, nstep)))

        for i, t in enumerate(scan_skydir):
            # make a copy
            s = self.roi.copy_source(name)

            model_name = '%s_localize' % (name.replace(' ', '').lower())
            s.set_name(model_name)
            s.set_position(t)
            #            s.set_spatial_model(spatial_model,w)

            self.add_source(model_name, s, free=True,
                            init_source=False, save_source_maps=False,
                            loglevel=logging.DEBUG)
            #self.fit(update=False)
            self.like.optimize(0)
            
            logLike1 = -self.like()
            lnlscan['logLike'].flat[i] = logLike1
            self.delete_source(model_name,loglevel=logging.DEBUG)

        lnlscan['dlogLike'] = lnlscan['logLike'] - np.max(lnlscan['logLike'])

        saved_state.restore()
        
        ix, iy = np.unravel_index(np.argmax(lnlscan['dlogLike']),(nstep,nstep))
        
        scan_fit = utils.fit_parabola(lnlscan['dlogLike'], ix, iy, dpix=3)

        sigmax = 2.**0.5*scan_fit['sigmax']*scan_step
        sigmay = 2.**0.5*scan_fit['sigmay']*scan_step
                
        lnlscan['dlogLike_fit'] = \
            utils.parabola((np.linspace(0,nstep-1.0,nstep)[:,np.newaxis],
                            np.linspace(0,nstep-1.0,nstep)[np.newaxis,:]),
                           *scan_fit['popt']).reshape((nstep,nstep))
            
        o['lnlscan'] = lnlscan

        # Best fit position and uncertainty from fit to TS map
        o['peak_theta'] = tsmap_fit['theta']
        o['peak_sigmax'] = peak_sigmax
        o['peak_sigmay'] = peak_sigmay
        o['peak_sigma'] = peak_sigma
        o['peak_r68'] = peak_r68
        o['peak_r95'] = peak_r95
        o['peak_r99'] = peak_r99
        o['peak_ra'] = peak_skydir.icrs.ra.deg
        o['peak_dec'] = peak_skydir.icrs.dec.deg
        o['peak_glon'] = peak_skydir.galactic.l.deg
        o['peak_glat'] = peak_skydir.galactic.b.deg
        o['tsmap_fit'] = tsmap_fit
        o['scan_fit'] = scan_fit
        
        # Best fit position and uncertainty from likelihood scan
        o['xpix'] = scan_fit['x0']*scan_step/cdelt0 + scan_xpix[0]
        o['ypix'] = scan_fit['y0']*scan_step/cdelt1 + scan_ypix[0]
        o['deltax'] = (o['xpix']-src_pix[0])*cdelt0
        o['deltay'] = (o['ypix']-src_pix[1])*cdelt1
        o['theta'] = scan_fit['theta']
        o['sigmax'] = sigmax
        o['sigmay'] = sigmay
        o['sigma'] = (o['sigmax']*o['sigmay'])**0.5
        o['r68'] = 2.30**0.5*o['sigma']
        o['r95'] = 5.99**0.5*o['sigma']
        o['r99'] = 9.21**0.5*o['sigma']

        new_skydir = SkyCoord.from_pixel(o['xpix'],o['ypix'],skywcs)

        o['offset'] = skydir.separation(new_skydir).deg
        o['ra'] = new_skydir.icrs.ra.deg
        o['dec'] = new_skydir.icrs.dec.deg
        o['glon'] = new_skydir.galactic.l.deg
        o['glat'] = new_skydir.galactic.b.deg
        
        if o['fit_success'] and o['offset'] > dtheta_max:
            o['fit_success'] = False
            self.logger.error('Best-fit position outside search region:\n '
                              'offset = %.3f deltax = %.3f deltay = %.3f '%(o['offset'],
                                                                            o['deltax'],o['deltay']) +
                              'dtheta_max = %.3f'%(dtheta_max))

        self.roi[name]['localize'] = copy.deepcopy(o)

        try:
            self._plotter.make_localization_plot(self, name, tsmap, prefix=prefix,
                                                 **kwargs)
        except Exception:
            self.logger.error('Plot failed.', exc_info=True)
            
        if update and o['fit_success']:

            self.logger.info(
                'Updating position to: '
                'RA %8.3f DEC %8.3f (offset = %8.3f)' % (o['ra'], o['dec'],
                                                         o['offset']))
            s = self.roi.copy_source(name)
            self.delete_source(name)
            s.set_position(new_skydir)
            s.set_name(newname, names=s.names)

            self.add_source(newname, s, free=True)
            self.fit()
            src = self.roi.get_source_by_name(newname, True)
            self.roi[name]['localize'] = copy.deepcopy(o)
            
        self.logger.info('Finished localization.')
        return o

    def extension(self, name, **kwargs):
        """Test this source for spatial extension with the likelihood
        ratio method (TS_ext).  This method will substitute an
        extended spatial model for the given source and perform a
        one-dimensional scan of the spatial extension parameter over
        the range specified with the width parameters.  The 1-D
        profile likelihood is then used to compute the best-fit value,
        upper limit, and TS for extension.  Any background parameters
        that are free will also be simultaneously profiled in the
        likelihood scan.

        Parameters
        ----------

        name : str
            Source name.

        spatial_model : str
            Spatial model that will be used when testing extension
            (e.g. DiskSource, GaussianSource).

        width_min : float
            Minimum value in degrees for the spatial extension scan.
        
        width_max : float
            Maximum value in degrees for the spatial extension scan.

        width_nstep : int
            Number of scan points between width_min and width_max.
            Scan points will be spaced evenly on a logarithmic scale
            between log(width_min) and log(width_max).

        width : array-like        
            Sequence of values in degrees for the spatial extension
            scan.  If this argument is None then the scan points will
            be determined from width_min/width_max/width_nstep.
            
        fix_background : bool
            Fix all background sources when performing the extension fit.

        update : bool        
            Update this source with the best-fit model for spatial
            extension.
            
        save_model_map : bool
            Save model maps for all steps in the likelihood scan.
            
        Returns
        -------

        extension : dict
            Dictionary containing results of the extension analysis.  The same
            dictionary is also saved to the dictionary of this source under
            'extension'.
        """

        name = self.roi.get_source_by_name(name, True).name

        # Extract options from kwargs
        config = copy.deepcopy(self.config['extension'])
        fermipy.config.validate_config(kwargs, config)
        config.update(kwargs)

        spatial_model = config['spatial_model']
        width_min = config['width_min']
        width_max = config['width_max']
        width_nstep = config['width_nstep']
        width = config['width']
        fix_background = config['fix_background']
        save_model_map = config['save_model_map']
        update = config['update']

        self.logger.info('Starting')
        self.logger.info('Running analysis for %s' % name)

        ext_model_name = '%s_ext' % (name.lower().replace(' ', '_'))
        null_model_name = '%s_noext' % (name.lower().replace(' ', '_'))

        saved_state = LikelihoodState(self.like)

        if fix_background:
            self.free_sources(free=False)

        # Fit baseline model
        self.free_norm(name)
        self.fit(update=False)

        # Save likelihood value for baseline fit
        logLike0 = -self.like()

        #self.write_model_map(model_name=null_model_name, name=name)

        #        src = self.like.deleteSource(name)
        normPar = self.like.normPar(name).getName()
        self.scale_parameter(name, normPar, 1E-10)
        self.free_source(name, free=False)
        self.like.syncSrcParams(name)

        if save_model_map:
            self.write_model_map(model_name=ext_model_name + '_bkg')

        if width is None:
            width = np.logspace(np.log10(width_min), np.log10(width_max),
                                width_nstep)

        width = np.array(width)
        width = np.delete(width,0.0)            
        width = np.concatenate(([0.0],np.array(width)))
            
        o = defaults.make_default_dict(defaults.extension_output)
        o['width'] = width
        o['dlogLike'] = np.zeros(len(width)+1)
        o['logLike'] = np.zeros(len(width)+1)
        o['logLike_base'] = logLike0
        o['config'] = config

        # Fit a point-source
        s = self.roi.copy_source(name)
        model_name = '%s_ptsrc' % (name)
        s.set_name(model_name)
        s.set_spatial_model('PSFSource')
        #s.set_spatial_model('PointSource')

        self.logger.debug('Testing point-source model.')
        self.add_source(model_name, s, free=True, init_source=False,
                        loglevel=logging.DEBUG)
        self.fit(update=False)
        o['logLike_ptsrc'] = -self.like()
        
        self.delete_source(model_name, save_template=False,
                           loglevel=logging.DEBUG)
        
        # Perform scan over width parameter
        self.logger.debug('Width scan vector:\n %s' % width)

        if not hasattr(self.components[0].like.logLike, 'setSourceMapImage'):
            o['logLike'] = self._scan_extension_pylike(name, spatial_model, width[1:])
        else:
            o['logLike'] = self._scan_extension(name, spatial_model, width[1:])
        o['logLike'] = np.concatenate(([o['logLike_ptsrc']],o['logLike']))
        o['dlogLike'] = o['logLike'] - o['logLike_ptsrc']
        
        try:
            
            ul_data = utils.get_parameter_limits(o['width'], o['dlogLike'])

            o['ext'] = ul_data['x0']
            o['ext_ul95'] = ul_data['ul']
            o['ext_err_lo'] = ul_data['err_lo']
            o['ext_err_hi'] = ul_data['err_hi']
            o['ts_ext'] = 2 * ul_data['lnlmax']
            o['ext_err'] = ul_data['err']
        except Exception:
            self.logger.error('Upper limit failed.', exc_info=True)

        self.logger.info('Best-fit extension: %6.4f + %6.4f - %6.4f'
                         % (o['ext'], o['ext_err_lo'], o['ext_err_hi']))
        self.logger.info('TS_ext:        %.3f' % o['ts_ext'])
        self.logger.info('Extension UL: %6.4f' % o['ext_ul95'])

            
        if np.isfinite(o['ext']):

            # Fit with the best-fit extension model
            s = self.roi.copy_source(name)
            model_name = ext_model_name
            s.set_name(model_name)
            s.set_spatial_model(spatial_model, max(o['ext'],10**-2.5))

            self.logger.info('Refitting extended model')
            self.add_source(model_name, s, free=True)
            self.fit()

            o['source_fit'] = self.get_src_model(model_name)
            o['logLike_ext'] = -self.like()
            
#            self.write_model_map(model_name=model_name,
#                                    name=model_name)

            src_ext = self.delete_source(model_name, save_template=False)

        # Restore ROI parameters to previous state
        self.scale_parameter(name, normPar, 1E10)
        self.like.syncSrcParams(name)
        saved_state.restore()
                
        if update and src_ext is not None:
            src = self.delete_source(name)
            src.set_spectral_pars(src_ext.spectral_pars)
            src.set_spatial_model(src_ext['SpatialModel'],
                                  src_ext['SpatialWidth'])
            self.add_source(name,src,free=True)
            self.fit()
        
        src = self.roi.get_source_by_name(name, True)
        src['extension'] = copy.deepcopy(o)

        self.logger.info('Finished')

        return o

    def _scan_extension(self, name, spatial_model, width):

        ext_model_name = '%s_ext' % (name.lower().replace(' ', '_'))
        
        s = self.roi.copy_source(name)
        s.set_name(ext_model_name)
        s.set_spatial_model('PSFSource', width[-1])
        self.add_source(ext_model_name, s, free=True, init_source=False)

        par = self.like.normPar(ext_model_name)
        
        logLike = []
        for i, w in enumerate(width):

            self._update_srcmap(ext_model_name, self.roi[name].skydir,
                                spatial_model, w)
            self.like.optimize(0)            
            logLike += [-self.like()]

        self.delete_source(ext_model_name, save_template=False)
            
        return np.array(logLike)

    def _scan_extension_pylike(self, name, spatial_model, width):

        ext_model_name = '%s_ext' % (name.lower().replace(' ', '_'))

        logLike = []
        for i, w in enumerate(width):

            # make a copy
            s = self.roi.copy_source(name)
            s.set_name(ext_model_name)
            s.set_spatial_model(spatial_model, w)
            
            self.logger.debug('Adding test source with width: %10.3f deg' % w)
            self.add_source(ext_model_name, s, free=True, init_source=False,
                            loglevel=logging.DEBUG)
            
            self.like.optimize(0)
            logLike += [-self.like()]
            
#            if save_model_map:
#                self.write_model_map(model_name=model_name + '%02i' % i,
#                                        name=model_name)
                
            self.delete_source(ext_model_name, save_template=False,
                               loglevel=logging.DEBUG)

        return np.array(logLike)
    
    def profile_norm(self, name, emin=None, emax=None, reoptimize=False,
                     xvals=None, npts=20, fix_shape=True, savestate=True):
        """Profile the normalization of a source.

        Parameters
        ----------

        name : str
            Source name.

        reoptimize : bool
            Re-optimize free parameters in the model at each point
            in the profile likelihood scan.

        """

        if savestate:
            saved_state = LikelihoodState(self.like)

        if fix_shape:
            self.free_sources(False,pars='shape')
            
        # Find the source
        name = self.roi.get_source_by_name(name, True).name
        parName = self.like.normPar(name).getName()

        erange = self.erange
        if emin is not None or emax is not None:
            self.setEnergyRange(emin,emax)

        self.like.optimize(0)
            
        # Find a sequence of values for the normalization scan
        if xvals is None:

            if reoptimize:
                xvals = self._find_scan_pts_reopt(name,npts=npts)
            else:
                xvals = self._find_scan_pts(name,npts=npts)
                
        o = self.profile(name, parName, 
                         reoptimize=reoptimize, xvals=xvals,
                         savestate=savestate)

        if savestate:
            saved_state.restore() 
        
        if emin is not None or emax is not None:
            self.setEnergyRange(*erange)
        
        return o

    def _find_scan_pts(self,name,emin=None,emax=None,npts=20):

        par = self.like.normPar(name)
        
        eminmax = [self.erange[0] if emin is None else emin,
                   self.erange[1] if emax is None else emax]
        
        val = par.getValue()
        
        if val == 0:
            par.setValue(1.0)
            self.like.syncSrcParams(name)
            cs = self.model_counts_spectrum(name,
                                            eminmax[0],
                                            eminmax[1],
                                            summed=True)
            npred = np.sum(cs)
            val = 1./npred
            npred = 1.0
            par.setValue(0.0)
            self.like.syncSrcParams(name)
        else:
            cs = self.model_counts_spectrum(name,
                                            eminmax[0],
                                            eminmax[1],
                                            summed=True)
            npred = np.sum(cs)

        if npred < 10:
            val *= 1. / min(1.0, npred)
            xvals = val * 10 ** np.linspace(-1.0, 3.0, 2 * npts + 1)
            xvals = np.insert(xvals, 0, 0.0)
        else:
            xvals = np.linspace(0, 1, 1 + npts)
            xvals = np.concatenate((-1.0 * xvals[1:][::-1], xvals))
            xvals = val * 10 ** xvals
            xvals = np.insert(xvals, 0, 0.0)
            
        return xvals
    
    def _find_scan_pts_reopt(self,name,emin=None,emax=None,npts=11):
        
        parName = self.like.normPar(name).getName()
        
        npts = max(npts,5)
        xvals = self._find_scan_pts(name,emin=emin, emax=emax, npts=20)
        lnlp0 = self.profile(name, parName, emin=emin, emax=emax, 
                             reoptimize=False,xvals=xvals)
        xval0 = self.like.normPar(name).getValue()
        lims0 = utils.get_parameter_limits(lnlp0['xvals'], lnlp0['dlogLike'],
                                           ul_confidence=0.99)

        if not np.isfinite(lims0['ll']):
            xvals = np.array([0.0,xval0,xval0+lims0['err_hi'],lims0['ul']])
        else:
            xvals = np.array([lims0['ll'],
                              xval0-lims0['err_lo'],xval0,
                              xval0+lims0['err_hi'],
                              lims0['ul']])
            
        lnlp1 = self.profile(name, parName, emin=emin, emax=emax, 
                             reoptimize=True,xvals=xvals)

        dlogLike = copy.deepcopy(lnlp1['dlogLike'])
        dlogLike0 = dlogLike[-1]
        xup = xvals[-1]
        
        for i in range(20):
            
            lims1 = utils.get_parameter_limits(xvals, dlogLike,
                                               ul_confidence=0.99)
                
            if np.abs(np.abs(dlogLike0) - utils.cl_to_dlnl(0.99)) < 0.1:
                break
                
            if not np.isfinite(lims1['ul']) or np.abs(dlogLike[-1]) < 1.0:
                xup = 2.0*xvals[-1]
            else:
                xup = lims1['ul']
                                                    
            lnlp = self.profile(name, parName, emin=emin, emax=emax,
                                reoptimize=True,xvals=[xup])
            dlogLike0 = lnlp['dlogLike']
                
            dlogLike = np.concatenate((dlogLike,dlogLike0))
            xvals = np.concatenate((xvals,[xup]))
            isort = np.argsort(xvals)
            dlogLike = dlogLike[isort]
            xvals = xvals[isort]

#        from scipy.interpolate import UnivariateSpline
#        s = UnivariateSpline(xvals,dlogLike,k=2,s=1E-4)        
#        import matplotlib.pyplot as plt
#        plt.figure()
#        plt.plot(xvals,dlogLike,marker='o')
#        plt.plot(np.linspace(xvals[0],xvals[-1],100),s(np.linspace(xvals[0],xvals[-1],100)))
#        plt.gca().set_ylim(-5,1)
#        plt.gca().axhline(-utils.cl_to_dlnl(0.99))
        
        if np.isfinite(lims1['ll']):
            xlo = np.concatenate(([0.0],np.linspace(lims1['ll'],xval0,(npts+1)//2-1)))            
        elif np.abs(dlogLike[0]) > 0.1:
            xlo = np.linspace(0.0,xval0,(npts+1)//2)
        else:
            xlo = np.array([0.0,xval0])
            
        if np.isfinite(lims1['ul']):
            xhi = np.linspace(xval0,lims1['ul'],npts+1-len(xlo))[1:]
        else:
            xhi = np.linspace(xval0,lims0['ul'],npts+1-len(xlo))[1:]

        xvals = np.concatenate((xlo,xhi))
        return xvals
#        xvals = np.concatenate((xlo[1:-1],xhi))
#        lnlp = self.profile_norm(name, savestate=True,
#                                 reoptimize=True,xvals=xvals)

#        isort = np.argsort(np.concatenate((lnlp1['xvals'][:2],lnlp['xvals'])))
#        for k, v in lnlp.items():
#            lnlp[k] = np.concatenate((lnlp1[k][:2],lnlp[k]))[isort]
    
    def profile(self, name, parName, emin=None, emax=None, reoptimize=False,
                xvals=None, npts=None, savestate=True):
        """Profile the likelihood for the given source and parameter.

        Parameters
        ----------

        name : str
           Source name.

        parName : str
           Parameter name.
        
        reoptimize : bool
           Re-fit nuisance parameters at each step in the scan.  Note
           that this will only re-fit parameters that were free when
           the method was executed.

        Returns
        -------

        lnlprofile : dict
           Dictionary containing results of likelihood scan.

        """
        
        # Find the source
        name = self.roi.get_source_by_name(name, True).name

        par = self.like.normPar(name)
        parName = self.like.normPar(name).getName()
        idx = self.like.par_index(name, parName)
        bounds = self.like.model[idx].getBounds()
        value = self.like.model[idx].getValue()
        erange = self.erange
        
        if savestate:
            saved_state = LikelihoodState(self.like)
            
        # If parameter is fixed temporarily free it
        par.setFree(True)

        if emin is not None or emax is not None:
            eminmax = self.setEnergyRange(emin, emax)
        else:
            eminmax = self.erange
            
        logLike0 = -self.like()

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
        self.like[idx].setBounds(min(xvals[0],value),
                                 max(xvals[-1],value))

        o = {'xvals': xvals,
             'Npred': np.zeros(len(xvals)),
             'dfde': np.zeros(len(xvals)),
             'flux': np.zeros(len(xvals)),
             'eflux': np.zeros(len(xvals)),
             'dlogLike': np.zeros(len(xvals)),
             'logLike': np.zeros(len(xvals))
             }

        if reoptimize and hasattr(self.like.components[0].logLike,
                                  'setUpdateFixedWeights'):

            for c in self.components:
                c.like.logLike.setUpdateFixedWeights(False)
        
        for i, x in enumerate(xvals):
            
            self.like[idx] = x
            self.like.syncSrcParams(name)

            if self.like.logLike.getNumFreeParams() > 1 and reoptimize:
                # Only reoptimize if not all frozen
                self.like.freeze(idx)
                self.like.optimize(0)
                logLike1 = -self.like()
                self.like.thaw(idx)
            else:
                logLike1 = -self.like()
            
            flux = self.like[name].flux(10 ** eminmax[0], 10 ** eminmax[1])
            eflux = self.like[name].energyFlux(10 ** eminmax[0], 10 ** eminmax[1])
            prefactor = self.like[idx]

            o['dlogLike'][i] = logLike1 - logLike0
            o['logLike'][i] = logLike1
            o['dfde'][i] = prefactor.getTrueValue()
            o['flux'][i] = flux
            o['eflux'][i] = eflux

            cs = self.model_counts_spectrum(name,
                                            eminmax[0],
                                            eminmax[1], summed=True)
            o['Npred'][i] += np.sum(cs)

        if reoptimize and hasattr(self.like.components[0].logLike,
                                  'setUpdateFixedWeights'):

            for c in self.components:
                c.like.logLike.setUpdateFixedWeights(True)
                
        # Restore model parameters to original values
        if savestate:
            saved_state.restore()        
            
        self.like[idx].setBounds(*bounds)
        if emin is not None or emax is not None:
            self.setEnergyRange(*erange)
        
        return o

    def tsmap(self, prefix='', **kwargs):
        """Generate a spatial TS map for a source component with
        properties defined by the `model` argument.  The TS map will
        have the same geometry as the ROI.  The output of this method
        is a dictionary containing `~fermipy.utils.Map` objects with
        the TS and amplitude of the best-fit test source.  By default
        this method will also save maps to FITS files and render them
        as image files.

        This method uses a simplified likelihood fitting
        implementation that only fits for the normalization of the
        test source.  Before running this method it is recommended to
        first optimize the ROI model (e.g. by running
        :py:meth:`~fermipy.gtanalysis.GTAnalysis.optimize`).

        Parameters
        ----------

        prefix : str
           Optional string that will be prepended to all output files
           (FITS and rendered images).

        model : dict
           Dictionary defining the properties of the test source.

        exclude : str or list of str
            Source or sources that will be removed from the model when
            computing the TS map.

        erange : list
           Restrict the analysis to an energy range (emin,emax) in
           log10(E/MeV) that is a subset of the analysis energy range.
           By default the full analysis energy range will be used.  If
           either emin/emax are None then only an upper/lower bound on
           the energy range wil be applied.

        max_kernel_radius : float
           Set the maximum radius of the test source kernel.  Using a
           smaller value will speed up the TS calculation at the loss of
           accuracy.  The default value is 3 degrees.

        make_plots : bool
           Write image files.

        make_fits : bool
           Write FITS files.

        Returns
        -------

        maps : dict
           A dictionary containing the `~fermipy.utils.Map` objects
           for TS and source amplitude.

        """

        self.logger.info('Generating TS maps')

        make_plots = kwargs.get('make_plots', True)

        maps = self._tsmap_fast(prefix, **kwargs)

        for m in maps if isinstance(maps, list) else [maps]:
            if make_plots:
                plotter = plotting.AnalysisPlotter(self.config['plotting'],
                                                   fileio=self.config['fileio'],
                                                   logging=self.config['logging'])

                plotter.make_tsmap_plots(self, m)

        self.logger.info('Finished TS maps')

        return maps


    def find_sources(self, prefix='', **kwargs):
        """An iterative source-finding algorithm.

        Parameters
        ----------

        model : dict        
           Dictionary defining the properties of the test source.
           This is the model that will be used for generating TS maps.
        
        sqrt_ts_threshold : float
           Source threshold in sqrt(TS).  Only peaks with sqrt(TS)
           exceeding this threshold will be used as seeds for new
           sources.

        min_separation : float
           Minimum separation in degrees of sources detected in each
           iteration. The source finder will look for the maximum peak
           in the TS map within a circular region of this radius.

        max_iter : int
           Maximum number of source finding iterations.  The source
           finder will continue adding sources until no additional
           peaks are found or the number of iterations exceeds this
           number.

        sources_per_iter : int
           Maximum number of sources that will be added in each
           iteration.  If the number of detected peaks in a given
           iteration is larger than this number, only the N peaks with
           the largest TS will be used as seeds for the current
           iteration.

        tsmap_fitter : str        
           Set the method used internally for generating TS maps.
           Valid options:

           * tsmap 
           * tscube

        tsmap : dict
           Keyword arguments dictionary for tsmap method.

        tscube : dict
           Keyword arguments dictionary for tscube method.
           
           
        """

        self.logger.info('Running source finding.')

        sf = SourceFinder(self.config['sourcefind'],
                          fileio=self.config['fileio'],
                          logging=self.config['logging'])
        
        maps = sf.find_sources(self, prefix, **kwargs)

        self.logger.info('Finished source finding.')
        
        return maps
        
    def _init_optimizer(self):
        pass

    def _create_optObject(self):
        """ Make MINUIT or NewMinuit type optimizer object """

        optimizer = self.config['optimizer']['optimizer']
        if optimizer.upper() == 'MINUIT':
            optObject = pyLike.Minuit(self.like.logLike)
        elif optimizer.upper == 'NEWMINUIT':
            optObject = pyLike.NewMinuit(self.like.logLike)
        else:
            optFactory = pyLike.OptimizerFactory_instance()
            optObject = optFactory.create(optimizer, self.like.logLike)
        return optObject

    def _run_fit(self, **kwargs):

        try:
            self.like.fit(**kwargs)
        except Exception:
            self.logger.error('Likelihood optimization failed.', exc_info=True)

        if isinstance(self.like.optObject, pyLike.Minuit) or \
                isinstance(self.like.optObject, pyLike.NewMinuit):
            quality = self.like.optObject.getQuality()
        else:
            quality = 3

        return quality

    def constrain_norms(self, srcNames, cov_scale=1.0):

        # Get the covariance matrix

        for name in srcNames:
            par = self.like.normPar(name)

            err = par.error()
            val = par.getValue()
            
            if par.error() == 0.0 or not par.isFree():
                continue
            
            self.add_gauss_prior(name, par.getName(),
                                 val,err*cov_scale)

    def add_gauss_prior(self, name, parName, mean, sigma):
        
        par = self.like[name].funcs["Spectrum"].params[parName]
        par.addGaussianPrior(mean,sigma)

    def remove_prior(self,name, parName):

        par = self.like[name].funcs["Spectrum"].params[parName]
        par.removePrior()

    def remove_priors(self):
        """Clear all priors."""

        for src in self.roi.sources:

            for par in self.like[src.name].funcs["Spectrum"].params.values():
                par.removePrior()
        
    def fit(self, update=True, **kwargs):
        """Run the likelihood optimization.  This will execute a fit
        of all parameters that are currently free in the ROI model and
        update the charateristics (TS, Npred, etc.) of the
        corresponding model components.  The fit will be repeated N
        times (set with the retries parameter) until a fit quality of
        3 is obtained.

        Parameters
        ----------

        update : bool
           Do not update the ROI model.

        tol : float
           Set the optimizer tolerance.

        verbosity : int
           Set the optimizer output level.

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
           Dictionary containing diagnostic information for the fit
           (fit quality, parameter covariances, etc.).

        """

        if not self.like.logLike.getNumFreeParams():
            self.logger.debug("Skipping fit.  No free parameters.")
            return

        verbosity = kwargs.get('verbosity',
                               self.config['optimizer']['verbosity'])
        covar = kwargs.get('covar', True)
        tol = kwargs.get('tol', self.config['optimizer']['tol'])
        retries = kwargs.get('retries', self.config['optimizer']['retries'])
        min_fit_quality = kwargs.get('min_fit_quality',
                                     self.config['optimizer'][
                                         'min_fit_quality'])
        reoptimize = kwargs.get('reoptimize', False)
        
        saved_state = LikelihoodState(self.like)
        kw = dict(optObject=self._create_optObject(),
                  covar=covar, verbosity=verbosity, tol=tol)

        num_free = self.like.logLike.getNumFreeParams()
        o = {'fit_quality' : 0,
             'covariance' : None,
             'correlation' : None,
             'logLike' : None, 'dlogLike' : None,
             'values' : np.ones(num_free)*np.nan,
             'errors' : np.ones(num_free)*np.nan,
             'indices': np.zeros(num_free,dtype=int),
             'is_norm' : np.empty(num_free,dtype=bool),
             'src_names' : num_free*[None],
             'par_names' : num_free*[None],
             }

        logLike0 = -self.like()        
        quality = 0
        niter = 0
        max_niter = retries
        while niter < max_niter:
            self.logger.debug("Fit iteration: %i" % niter)
            niter += 1
            quality = self._run_fit(**kw)
            if quality > 2:
                break
            
        o['fit_quality'] = quality
        o['covariance'] = np.array(self.like.covariance)
        o['errors'] = np.diag(o['covariance'])**0.5
        
        errinv = 1./o['errors']

        o['correlation'] = \
            o['covariance']*errinv[:,np.newaxis]*errinv[np.newaxis,:]

        free_params = self.get_free_params()
        for i, p in enumerate(free_params):
            o['values'][i] = p['value']
            o['errors'][i] = p['error']
            o['indices'][i] = p['idx']
            o['src_names'][i] = p['src_name']
            o['par_names'][i] = p['par_name']
            o['is_norm'][i] = p['is_norm']
        
        o['niter'] = niter
        
        # except Exception, message:
        #            print self.like.optObject.getQuality()
        #            self.logger.error('Likelihood optimization failed.',
        # exc_info=True)
        #            saved_state.restore()
        #            return quality

        o['logLike'] = -self.like()
        o['dlogLike'] = o['logLike'] - logLike0

        if o['fit_quality'] < min_fit_quality:
            self.logger.error(
                "Failed to converge with %s" % self.like.optimizer)
            saved_state.restore()
            return o

        if update:

            self._extract_correlation(o,free_params)
            for name in self.like.sourceNames():
                freePars = self.get_free_source_params(name)                
                if len(freePars) == 0:
                    continue
                self.update_source(name, reoptimize=reoptimize)
                
            self._roi_model['logLike'] = o['logLike']
            self._roi_model['fit_quality'] = o['fit_quality']

            for i, c in enumerate(self.components):
                self._roi_model['components'][i]['logLike'] = -c.like()

            # Update roi model counts
            self._update_roi()

        self.logger.debug("Fit returned successfully.")
        self.logger.debug("Fit Quality: %i "%o['fit_quality'] + 
                          "LogLike: %12.3f "%o['logLike'] + 
                          "DeltaLogLike: %12.3f"%o['dlogLike'])
        return o

    def fit_correlation(self):

        saved_state = LikelihoodState(self.like)
        self.free_sources(False)
        self.free_sources(pars='norm')
        fit_results = self.fit(min_fit_quality=2)
        free_params = self.get_free_params()
        self._extract_correlation(fit_results,free_params)        
        saved_state.restore()
    
    def _extract_correlation(self,fit_results,free_params):
        
        for i, p0 in enumerate(free_params):
            if not p0['is_norm']:
                continue

            src = self.roi[p0['src_name']]
            for j, p1 in enumerate(free_params):

                if not p1['is_norm']:
                    continue
                
                src['correlation'][p1['src_name']] = fit_results['correlation'][i,j]
        
    
    def load_xml(self, xmlfile):
        """Load model definition from XML."""

        self.logger.info('Loading XML')
        
        for c in self.components:
            c.load_xml(xmlfile)

        for name in self.like.sourceNames():
            self.update_source(name)

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

        Returns
        -------

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
        Inject a simulated source into the 

        Parameters
        ----------
        src_dict : dict

        Returns
        -------

        """

        if src_dict is None:
            src_dict = {}
        else:
            src_dict = copy.deepcopy(src_dict)

        skydir = utils.get_target_skydir(src_dict, self.roi.skydir)

        src_dict.setdefault('ra', skydir.ra.deg)
        src_dict.setdefault('dec', skydir.dec.deg)
        src_dict.setdefault('SpatialModel', 'PointSource')
        src_dict.setdefault('SpatialWidth', 0.3)
        src_dict.setdefault('Index', 2.0)
        src_dict.setdefault('Prefactor', 1E-13)

        self.add_source('mcsource', src_dict, free=True,
                        init_source=False)
        for c in self.components:
            c.simulate_roi('mcsource',clear=False)

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
        """
        Generate a simulation of the ROI using the current best-fit
        model and replace the data counts cube with this simulation.
        The simulation is created by generating an array of Poisson
        random numbers with expectation values drawn from the model
        cube of the binned analysis instance.  This function will
        update the counts cube both in memory and in the source map
        file.  The counts cube can be restored to its original state
        by calling this method with `restore`=True.

        Parameters
        ----------

        name : str
           Name of the model component to be simulated.  If None then
           the whole ROI will be simulated.

        restore : bool
           Restore the data counts cube to its original state.
           
        """

        if restore:
            self._restore_counts_maps()
            return
        
        for c in self.components:
            c.simulate_roi(name=name,clear=True,randomize=randomize)

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

        outfile = os.path.join(self.config['fileio']['workdir'],
                               'mcube_%s.fits' % (model_name))

        if self.projtype == "HPX":
            shape = (self.enumbins, self._proj.npix)
            model_counts = fits_utils.make_coadd_map(maps, self._proj, shape)
            utils.write_hpx_image(model_counts.counts, self._proj, outfile)
        elif self.projtype == "WCS":
            shape = (self.enumbins, self.npix, self.npix)
            model_counts = fits_utils.make_coadd_map(maps, self._proj, shape)
            utils.write_fits_image(model_counts.counts, self._proj, outfile)
        else:
            raise Exception(
                "Did not recognize projection type %s" % self.projtype)
        return [model_counts] + maps

    def print_roi(self):
        self.logger.info(str(self.roi))

    def print_params(self):

        pars = self.get_free_params()

        o = '\n'
        o += '%4s %-20s%10s%10s%10s%10s%10s%5s\n' % (
        'idx','parname', 'value','error',
        'min', 'max', 'scale', 'free')
        
        o += '-' * 80 + '\n'

        src_pars = collections.OrderedDict()
        for p in pars:
            src_pars.setdefault(p['src_name'],[])
            src_pars[p['src_name']] += [p]

        for k, v in src_pars.items():

            o += '%s\n'%k
            for p in v:

                o += '%4i %-20.19s' % (p['idx'], p['par_name'])  
                o += '%10.3g%10.3g' % (p['value'],p['error'])
                o += '%10.3g%10.3g%10.3g' % (p['bounds'][0],p['bounds'][1],
                                          p['scale'])
            
                if p['is_free']:
                    o += '    *'
                else:
                    o += '     '

                o += '\n'
            
        self.logger.info(o)
            
    def print_model(self):

        o = '\n'
        o += '%-20s%8s%8s%7s%10s%10s%12s%5s\n' % (
        'sourcename', 'offset','norm','eflux','index',
        'ts', 'Npred', 'free')
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
                index = s['dfde1000_index'][0]
            else:
                index = 0.5*(s['dfde1000_index'][0]+s['dfde10000_index'][0])
                
            o += '%-20.19s%8.3f%8.3f%10.3g%7.2f%10.2f%12.1f%5s\n' % (
            s['name'], s['offset'], normVal, s['eflux'][0],index,
            s['ts'], s['Npred'],free_str)

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
                index = s['dfde1000_index'][0]
            else:
                index = 0.5*(s['dfde1000_index'][0]+s['dfde10000_index'][0])
                
            o += '%-20.19s%8s%8.3f%10.3g%7.2f%10.2f%12.1f%5s\n' % (
            s['name'], 
            '---', normVal, s['eflux'][0], index, s['ts'], s['Npred'],free_str)
                        
        self.logger.info(o)
                    
    def load_roi(self, infile):
        """This function reloads the analysis state from a previously
        saved instance generated with
        `~fermipy.gtanalysis.GTAnalysis.write_roi`."""
        
        infile = resolve_path(infile, workdir=self.config['fileio']['workdir'])
        roi_file, roi_data = load_roi_data(infile,
                                           workdir=self.config['fileio']['workdir'])

        self.logger.info('Loading ROI file: %s'%roi_file)
        
        self._roi_model = roi_data['roi']
        self._erange = self._roi_model.setdefault('erange',self.erange)
                
        sources = roi_data.pop('sources')

        self.roi.load_sources(sources.values())
        for c in self.components:
            c.roi.load_sources(sources.values())

        self._create_likelihood(infile)
        self.setEnergyRange(self.erange[0], self.erange[1])
        
        # Load XML
#        self.load_xml(infile)

        self.logger.info('Finished Loading ROI')

    def write_roi(self, outfile=None, make_residuals=False, make_tsmap=False,
                  save_model_map=True, format=None, **kwargs):
        """Write current model to a file.  This function will write an
        XML model file and an ROI dictionary in both YAML and npy
        formats.

        Parameters
        ----------

        outfile : str
            Name of the output file.  The extension of this string
            will be stripped when generating the XML, YAML and
            Numpy filenames.

        make_residuals : bool
            Run residual analysis.

        save_model_map : bool
            Save the current counts model to a FITS file.

        format : str
            Set the output file format (yaml or npy).

        """
        # extract the results in a convenient format

        make_plots = kwargs.get('make_plots',True)

        if outfile is None:
            outfile = os.path.join(self.config['fileio']['workdir'],
                                   'results')
            prefix = ''
        else:
            outfile, ext = os.path.splitext(outfile)
            prefix = outfile
            if not os.path.isabs(outfile):
                outfile = os.path.join(self.config['fileio']['workdir'],
                                       outfile)

        self.write_xml(prefix)

        for c in self.components:
            c.like.logLike.saveSourceMaps(c._srcmap_file)
        
        mcube_maps = None
        if save_model_map:
            mcube_maps = self.write_model_map(prefix)

        if make_residuals:
            resid_maps = self.residmap(prefix, make_plots=make_plots)

        if make_tsmap:
            ts_maps = self.tsmap(prefix, make_plots=make_plots)

        o = {}
        o['roi'] = copy.deepcopy(self._roi_model)
        o['config'] = copy.deepcopy(self.config)
        o['version'] = fermipy.__version__
        o['sources'] = {}

        for s in self.roi.sources:
            o['sources'][s.name] = copy.deepcopy(s.data)

        if format is None:
            format = ['npy','yaml']
        elif not isinstance(format,list):
            format = [format]
            
        for fmt in format:

            if fmt == 'yaml':
                self.logger.info('Writing %s...' % (outfile + '.yaml'))
                yaml.dump(tolist(o), open(outfile + '.yaml', 'w'))
            elif fmt == 'npy':                
                self.logger.info('Writing %s...' % (outfile + '.npy'))
                np.save(outfile + '.npy', o)
            else:
                raise Exception('Unrecognized format.')

        if make_plots:
            self.make_plots(prefix, mcube_maps[0],
                            **kwargs.get('plotting',{}))

    def make_plots(self, prefix, mcube_map=None, **kwargs):

        #mcube_maps = kwargs.pop('mcube_maps', None)
        if mcube_map is None:
            mcube_map = self.model_counts_map()

        plotter = plotting.AnalysisPlotter(self.config['plotting'],
                                           fileio=self.config['fileio'],
                                           logging=self.config['logging'])
        plotter.run(self, mcube_map, prefix=prefix, **kwargs)

    def tscube(self,  prefix='', **kwargs):
        """Generate a spatial TS map for a source component with
        properties defined by the `model` argument.  This method uses
        the `gttscube` ST application for source fitting and will
        simultaneously fit the test source normalization as well as
        the normalizations of any background components that are
        currently free.  The output of this method is a dictionary
        containing `~fermipy.utils.Map` objects with the TS and
        amplitude of the best-fit test source.  By default this method
        will also save maps to FITS files and render them as image
        files.

        Parameters
        ----------

        prefix : str
           Optional string that will be prepended to all output files
           (FITS and rendered images).

        model : dict
           Dictionary defining the properties of the test source.

        do_sed : bool
           Compute the energy bin-by-bin fits.
        
        nnorm : int
           Number of points in the likelihood v. normalization scan.

        norm_sigma : float
           Number of sigma to use for the scan range.
        
        tol : float        
           Critetia for fit convergence (estimated vertical distance
           to min < tol ).
        
        tol_type : int
           Absoulte (0) or relative (1) criteria for convergence.
        
        max_iter : int
           Maximum number of iterations for the Newton's method fitter
        
        remake_test_source : bool
           If true, recomputes the test source image (otherwise just shifts it)
        
        st_scan_level : int
           
        make_plots : bool
           Write image files.

        make_fits : bool
           Write FITS files.       

        Returns
        -------
        
        maps : dict
           A dictionary containing the `~fermipy.utils.Map` objects
           for TS and source amplitude.

        """
        
        make_plots = kwargs.get('make_plots', True)
        
        tsg = TSCubeGenerator(self.config['tscube'],
                              fileio=self.config['fileio'],
                              logging=self.config['logging'])

        model = kwargs.get('model', self.config['tscube']['model'])
        maps = tsg.make_ts_cube(self, prefix, copy.deepcopy(model), **kwargs)

        if make_plots:
            plotter = plotting.AnalysisPlotter(self.config['plotting'],
                                               fileio=self.config['fileio'],
                                               logging=self.config['logging'])
            
            plotter.make_tsmap_plots(self, maps, suffix='tscube')

        self.logger.info("Finished running TSCube.")
        return maps            

    def _tsmap_fast(self, prefix, **kwargs):
        """Evaluate the TS for an additional source component at each point
        in the ROI.  This is a simplified implementation optimized for speed
        that only fits for the source normalization (all background components
        are kept fixed)."""

        tsg = TSMapGenerator(self.config['tsmap'],
                             fileio=self.config['fileio'],
                             logging=self.config['logging'])

        model = kwargs.get('model', self.config['tsmap']['model'])
        return tsg.make_ts_map(self, prefix, copy.deepcopy(model), **kwargs)

    def _tsmap_pylike(self, prefix, **kwargs):
        """Evaluate the TS for an additional source component at each point
        in the ROI.  This is the brute force implementation of TS map
        generation that runs a full pyLikelihood fit
        at each point in the ROI."""

        logLike0 = -self.like()
        self.logger.info('LogLike: %f' % logLike0)

        saved_state = LikelihoodState(self.like)

        # Get the ROI geometry

        # Loop over pixels
        w = copy.deepcopy(self._skywcs)
        #        w = create_wcs(self._roi.skydir,cdelt=self._binsz,crpix=50.5)

        data = np.zeros((self.npix, self.npix))

        #        hdu_image = pyfits.PrimaryHDU(np.zeros((self.npix,self.npix)),
        #                                      header=w.to_header())
        #        for i in range(100):
        #            for j in range(100):
        #                print w.wcs_pix2world(i,j,0)

        #        self.free_sources(free=False)

        xpix = np.linspace(0, self.npix - 1, self.npix)[:,
               np.newaxis] * np.ones(data.shape)
        ypix = np.linspace(0, self.npix - 1, self.npix)[np.newaxis,
               :] * np.ones(data.shape)

        radec = utils.pix_to_skydir(xpix, ypix, w)
        radec = (np.ravel(radec.ra.deg), np.ravel(radec.dec.deg))

        testsource_dict = {
            'ra': radec[0][0],
            'dec': radec[1][0],
            'SpectrumType': 'PowerLaw',
            'Index': 2.0,
            'Scale': 1000,
            'Prefactor': {'value': 0.0, 'scale': 1e-13},
            'SpatialModel': 'PSFSource',
        }

        #        src = self.roi.get_source_by_name('tsmap_testsource',True)

        for i, (ra, dec) in enumerate(zip(radec[0], radec[1])):
            testsource_dict['ra'] = ra
            testsource_dict['dec'] = dec
            #                        src.set_position([ra,dec])
            self.add_source('tsmap_testsource', testsource_dict, free=True,
                            init_source=False,save_source_maps=False)

            #            for c in self.components:
            #                c.update_srcmap_file([src],True)

            self.set_parameter('tsmap_testsource', 'Prefactor', 0.0,
                               update_source=False)
            self.fit(update=False)

            logLike1 = -self.like()
            ts = max(0, 2 * (logLike1 - logLike0))

            data.flat[i] = ts

            #            print i, ra, dec, ts
            #            print self.like()
            #            print self.components[0].like.model['tsmap_testsource']

            self.delete_source('tsmap_testsource')

        saved_state.restore()

        outfile = os.path.join(self.config['fileio']['workdir'], 'tsmap.fits')
        utils.write_fits_image(data, w, outfile)

    def _bowtie(self, fd, energies=None):
        """Generate a spectral uncertainty band for the given source.
        This will create a band as a function of energy by propagating
        the errors on the global fit parameters.  Note that this band
        only reflects the uncertainty for parameters that were left
        free in the fit."""

        if energies is None:
            emin = self.energies[0]
            emax = self.energies[-1]
            energies = np.linspace(emin, emax, 50)

        dfde = [fd.value(10 ** x) for x in energies]
        dfde_err = [fd.error(10 ** x) for x in energies]

        dfde = np.array(dfde)
        dfde_err = np.array(dfde_err)
        fhi = dfde * (1.0 + dfde_err / dfde)
        flo = dfde / (1.0 + dfde_err / dfde)

        return {'ecenter': energies, 'dfde': dfde,
                'dfde_lo': flo, 'dfde_hi': fhi}

    def _coadd_maps(self, cmaps, shape, rm):
        """
        """
        
        if self.projtype == "WCS":
            shape = (self.enumbins, self.npix, self.npix)
            self._ccube = fits_utils.make_coadd_map(cmaps, self._proj, shape)
            utils.write_fits_image(self._ccube.counts, self._ccube.wcs,
                                   self._ccube_file)
            rm['counts'] += np.squeeze(
                np.apply_over_axes(np.sum, self._ccube.counts,
                                   axes=[1, 2]))
        elif self.projtype == "HPX":
            self._ccube = fits_utils.make_coadd_map(cmaps, self._proj, shape)
            utils.write_hpx_image(self._ccube.counts, self._ccube.hpx,
                                  self._ccube_file)
            rm['counts'] += np.squeeze(
                np.apply_over_axes(np.sum, self._ccube.counts,
                                   axes=[1]))
        else:
            raise Exception(
                "Did not recognize projection type %s" % self.projtype)

    def update_source(self, name, paramsonly=False, reoptimize=False,
                      npts=20):
        """Update the dictionary for this source.

        Parameters
        ----------

        name : str

        paramsonly : bool

        reoptimize : bool
           Re-fit background parameters in likelihood scan.

        npts : int
           Number of points for likelihood scan.
        """
        
        sd = self.get_src_model(name, paramsonly, reoptimize, npts)
        src = self.roi.get_source_by_name(name, True)
        src.update_data(sd)

        for c in self.components:
            src = c.roi.get_source_by_name(name, True)            
            src.update_data(sd)
            

    def get_src_model(self, name, paramsonly=False, reoptimize=False,
                      npts=20):
        """Compose a dictionary for the given source with the current
        best-fit parameters.

        Parameters
        ----------

        name : str

        paramsonly : bool

        reoptimize : bool
           Re-fit background parameters in likelihood scan.

        npts : int
           Number of points for likelihood scan.
        """

        self.logger.debug('Generating source dict for ' + name)

        name = self.get_source_name(name)
        source = self.like[name].src
        spectrum = source.spectrum()
        normPar = self.like.normPar(name)

        src_dict = {'name': name,
                    'flux': np.ones(2) * np.nan,
                    'flux100': np.ones(2) * np.nan,
                    'flux1000': np.ones(2) * np.nan,
                    'flux10000': np.ones(2) * np.nan,
                    'eflux': np.ones(2) * np.nan,
                    'eflux100': np.ones(2) * np.nan,
                    'eflux1000': np.ones(2) * np.nan,
                    'eflux10000': np.ones(2) * np.nan,
                    'dfde': np.ones(2) * np.nan,
                    'dfde100': np.ones(2) * np.nan,
                    'dfde1000': np.ones(2) * np.nan,
                    'dfde10000': np.ones(2) * np.nan,
                    'dfde100_index': np.ones(2) * np.nan,
                    'dfde1000_index': np.ones(2) * np.nan,
                    'dfde10000_index': np.ones(2) * np.nan,
                    'flux_ul95': np.nan,
                    'flux100_ul95': np.nan,
                    'flux1000_ul95': np.nan,
                    'flux10000_ul95': np.nan,
                    'eflux_ul95': np.nan,
                    'eflux100_ul95': np.nan,
                    'eflux1000_ul95': np.nan,
                    'eflux10000_ul95': np.nan,
                    'pivot_energy': 3.,
                    'ts': np.nan,
                    'Npred': 0.0,
                    'lnlprofile': None
                    }

        src_dict['params'] = gtlike_spectrum_to_dict(spectrum)
        
        # Get Counts Spectrum
        src_dict['model_counts'] = self.model_counts_spectrum(name, summed=True)

        # Get NPred
        src_dict['Npred'] = self.like.NpredValue(name)
        
        # Get the Model Fluxes
        try:
            src_dict['flux'][0] = self.like.flux(name, 10 ** self.energies[0],
                                                 10 ** self.energies[-1])
            src_dict['flux100'][0] = self.like.flux(name, 100., 10 ** 5.5)
            src_dict['flux1000'][0] = self.like.flux(name, 1000., 10 ** 5.5)
            src_dict['flux10000'][0] = self.like.flux(name, 10000., 10 ** 5.5)
            src_dict['eflux'][0] = self.like.energyFlux(name,
                                                        10 ** self.energies[0],
                                                        10 ** self.energies[-1])
            src_dict['eflux100'][0] = self.like.energyFlux(name, 100.,
                                                           10 ** 5.5)
            src_dict['eflux1000'][0] = self.like.energyFlux(name, 1000.,
                                                            10 ** 5.5)
            src_dict['eflux10000'][0] = self.like.energyFlux(name, 10000.,
                                                             10 ** 5.5)
            src_dict['dfde'][0] = self.like[name].spectrum()(
                pyLike.dArg(10 ** src_dict['pivot_energy']))
            src_dict['dfde100'][0] = self.like[name].spectrum()(
                pyLike.dArg(100.))
            src_dict['dfde1000'][0] = self.like[name].spectrum()(
                pyLike.dArg(1000.))
            src_dict['dfde10000'][0] = self.like[name].spectrum()(
                pyLike.dArg(10000.))

            if normPar.getValue() == 0:
                normPar.setValue(1.0)
                dfde100_index = -get_spectral_index(self.like[name],
                                                    100.)
                dfde1000_index = -get_spectral_index(self.like[name],
                                                     1000.)
                dfde10000_index = -get_spectral_index(self.like[name],
                                                      10000.)
                
                normPar.setValue(0.0)
            else:
                dfde100_index = -get_spectral_index(self.like[name],
                                                    100.)
                dfde1000_index = -get_spectral_index(self.like[name],
                                                     1000.)
                dfde10000_index = -get_spectral_index(self.like[name],
                                                      10000.)
            
            src_dict['dfde100_index'][0] = dfde100_index 
            src_dict['dfde1000_index'][0] = dfde1000_index
            src_dict['dfde10000_index'][0] = dfde10000_index
            
        except Exception:
            self.logger.error('Failed to update source parameters.',
                              exc_info=True)

        # Only compute TS, errors, and ULs if the source was free in
        # the fit
        if not self.get_free_source_params(name) or paramsonly:
            return src_dict

        try:
            src_dict['flux'][1] = self.like.fluxError(name,
                                                      10 ** self.energies[0],
                                                      10 ** self.energies[-1])
            src_dict['flux100'][1] = self.like.fluxError(name, 100., 10 ** 5.5)
            src_dict['flux1000'][1] = self.like.fluxError(name, 1000.,
                                                          10 ** 5.5)
            src_dict['flux10000'][1] = self.like.fluxError(name, 10000.,
                                                           10 ** 5.5)
            src_dict['eflux'][1] = self.like.energyFluxError(name, 10 **
                                                             self.energies[0],
                                                             10 **
                                                             self.energies[-1])
            src_dict['eflux100'][1] = self.like.energyFluxError(name, 100.,
                                                                10 ** 5.5)
            src_dict['eflux1000'][1] = self.like.energyFluxError(name, 1000.,
                                                                 10 ** 5.5)
            src_dict['eflux10000'][1] = self.like.energyFluxError(name, 10000.,
                                                                  10 ** 5.5)

        except Exception:
            pass
        # self.logger.error('Failed to update source parameters.',
        #  exc_info=True)

        lnlp = self.profile_norm(name, savestate=True,
                                 reoptimize=reoptimize,npts=npts)
            
        src_dict['lnlprofile'] = lnlp
        
        flux_ul_data = utils.get_parameter_limits(lnlp['flux'], lnlp['dlogLike'])
        eflux_ul_data = utils.get_parameter_limits(lnlp['eflux'], lnlp['dlogLike'])

        if normPar.getValue() == 0:
            normPar.setValue(1.0)
            flux = self.like.flux(name, 10 ** self.energies[0], 10 ** self.energies[-1])
            flux100 = self.like.flux(name, 100., 10 ** 5.5)
            flux1000 = self.like.flux(name, 1000., 10 ** 5.5)
            flux10000 = self.like.flux(name, 10000., 10 ** 5.5)
            eflux = self.like.energyFlux(name, 10 ** self.energies[0], 10 ** self.energies[-1])
            eflux100 = self.like.energyFlux(name, 100., 10 ** 5.5)
            eflux1000 = self.like.energyFlux(name, 1000., 10 ** 5.5)
            eflux10000 = self.like.energyFlux(name, 10000., 10 ** 5.5)

            flux100_ratio = flux100/flux
            flux1000_ratio = flux1000/flux
            flux10000_ratio = flux10000/flux
            eflux100_ratio = eflux100/flux
            eflux1000_ratio = eflux1000/flux
            eflux10000_ratio = eflux10000/flux
            normPar.setValue(0.0)
        else:
            flux100_ratio = src_dict['flux100'][0]/src_dict['flux'][0]
            flux1000_ratio = src_dict['flux1000'][0]/src_dict['flux'][0]
            flux10000_ratio = src_dict['flux10000'][0]/src_dict['flux'][0]
            
            eflux100_ratio = src_dict['eflux100'][0]/src_dict['eflux'][0]
            eflux1000_ratio = src_dict['eflux1000'][0]/src_dict['eflux'][0]
            eflux10000_ratio = src_dict['eflux10000'][0]/src_dict['eflux'][0]
        
        
        src_dict['flux_ul95'] = flux_ul_data['ul']
        src_dict['flux100_ul95'] = flux_ul_data['ul']*flux100_ratio
        src_dict['flux1000_ul95'] = flux_ul_data['ul']*flux1000_ratio
        src_dict['flux10000_ul95'] = flux_ul_data['ul']*flux10000_ratio

        src_dict['eflux_ul95'] = eflux_ul_data['ul']
        src_dict['eflux100_ul95'] = eflux_ul_data['ul']*eflux100_ratio
        src_dict['eflux1000_ul95'] = eflux_ul_data['ul']*eflux1000_ratio
        src_dict['eflux10000_ul95'] = eflux_ul_data['ul']*eflux10000_ratio

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
            energies = np.linspace(self.energies[0], self.energies[-1], 50)
            src_dict['model_flux'] = self._bowtie(fd, energies)
            src_dict['dfde100'][1] = fd.error(100.)
            src_dict['dfde1000'][1] = fd.error(1000.)
            src_dict['dfde10000'][1] = fd.error(10000.)

            ferr = (src_dict['model_flux']['dfde_hi'] -
                    src_dict['model_flux']['dfde_lo']) / src_dict['model_flux'][
                       'dfde']

            # Extract pivot energy
            try:
                src_dict['pivot_energy'] = interpolate_function_min(energies,
                                                                    ferr)
            except Exception:
                self.logger.error('Failed to compute pivot energy',
                                  exc_info=True)

            e0 = src_dict['pivot_energy']
            src_dict['dfde'][0] = self.like[name].spectrum()(
                pyLike.dArg(10 ** e0))
            src_dict['dfde'][1] = fd.error(10 ** e0)

        if not reoptimize:
            src_dict['ts'] = self.like.Ts2(name, reoptimize=reoptimize)
        else:
            src_dict['ts'] = -2.0*lnlp['dlogLike'][0]
            
        return src_dict


class GTBinnedAnalysis(fermipy.config.Configurable):
    defaults = dict(selection=defaults.selection,
                    binning=defaults.binning,
                    gtlike=defaults.gtlike,
                    data=defaults.data,
                    model=defaults.model,
                    logging=defaults.logging,
                    fileio=defaults.fileio,
                    name=('00', '', str),
                    file_suffix=('', '', str))

    def __init__(self, config, **kwargs):
        super(GTBinnedAnalysis, self).__init__(config, **kwargs)

        self._projtype = self.config['binning']['projtype']

        self.logger = Logger.get(self.__class__.__name__,
                                 self.config['fileio']['logfile'],
                                 ll(self.config['logging']['verbosity']))

        self._roi = ROIModel.create(self.config['selection'],
                                    self.config['model'],
                                    fileio=self.config['fileio'],
                                    logfile=self.config['fileio']['logfile'],
                                    logging=self.config['logging'],
                                    coordsys=self.config['binning']['coordsys'])

        workdir = self.config['fileio']['workdir']
        self._name = self.config['name']

        from os.path import join

        self._ft1_file = join(workdir,
                              'ft1%s.fits' % self.config['file_suffix'])
        self._ft1_filtered_file = join(workdir,
                                       'ft1_filtered%s.fits' % self.config[
                                           'file_suffix'])
        self._ltcube = join(workdir,
                            'ltcube%s.fits' % self.config['file_suffix'])
        self._ccube_file = join(workdir,
                                'ccube%s.fits' % self.config['file_suffix'])
        self._ccubemc_file = join(workdir,
                                  'ccubemc%s.fits' % self.config['file_suffix'])
        self._mcube_file = join(workdir,
                                'mcube%s.fits' % self.config['file_suffix'])
        self._srcmap_file = join(workdir,
                                 'srcmap%s.fits' % self.config['file_suffix'])
        self._bexpmap_file = join(workdir,
                                  'bexpmap%s.fits' % self.config['file_suffix'])
        self._bexpmap_roi_file = join(workdir,
                                      'bexpmap_roi%s.fits' % self.config[
                                          'file_suffix'])
        self._srcmdl_file = join(workdir,
                                 'srcmdl%s.xml' % self.config['file_suffix'])

        if self.config['binning']['enumbins'] is not None:
            self._enumbins = int(self.config['binning']['enumbins'])
        else:
            self._enumbins = np.round(self.config['binning']['binsperdec'] *
                                      np.log10(
                                          self.config['selection']['emax'] /
                                          self.config['selection']['emin']))
            self._enumbins = int(self._enumbins)

        self._ebin_edges = np.linspace(
            np.log10(self.config['selection']['emin']),
            np.log10(self.config['selection']['emax']),
            self._enumbins + 1)
        self._ebin_center = 0.5 * (self._ebin_edges[1:] + self._ebin_edges[:-1])

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
                'Automatically setting selection radius to %s deg' %
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

        if self.projtype == 'HPX':
            self._hpx_region = create_hpx_disk_region_string(self.roi.skydir,
                                                             self._coordsys,
                                                             0.5*self.config['binning']['roiwidth'])
            self._proj = HPX.create_hpx(-1,
                                     self.config['binning']['hpx_ordering_scheme'] == "NESTED",
                                     self._coordsys,
                                     self.config['binning']['hpx_order'],
                                     self._hpx_region,
                                     self._ebin_edges)
        elif self.projtype == "WCS":
            self._skywcs = create_wcs(self._roi.skydir,
                                      coordsys=self._coordsys,
                                      projection=self.config['binning']['proj'],
                                      cdelt=self.binsz,
                                      crpix=1.0 + 0.5 * (self._npix - 1),
                                      naxis=2)
            self._proj = create_wcs(self.roi.skydir,
                                    coordsys=self._coordsys,
                                    projection=self.config['binning']['proj'],
                                    cdelt=self.binsz,
                                    crpix=1.0 + 0.5 * (self._npix - 1),
                                    naxis=3,
                                    energies=self.energies)

        else:
            raise Exception(
                "Did not recognize projection type %s" % self.projtype)


        self.print_config(self.logger, loglevel=logging.DEBUG)

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

    def reload_source(self, name):
        """Delete and reload a source in the model."""

        src = self.roi.get_source_by_name(name, True)
        
        if hasattr(self.like.logLike, 'loadSourceMap'):

            if src['SpatialModel'] in ['PSFSource','GaussianSource',
                                       'DiskSource']:
                self._update_srcmap_file([src], True)
                self.like.logLike.loadSourceMap(name, False, False)
            else:
                self.like.logLike.loadSourceMap(name, True, False)

            utils.delete_source_map(self._srcmap_file,name)
            self.like.logLike.saveSourceMaps(self._srcmap_file)
            self.like.logLike.buildFixedModelWts()
        else:
            self.write_xml('tmp')
            src = self.delete_source(name)
            self.add_source(name, src, free=True)
            self.load_xml('tmp')
    
    def add_source(self, name, src_dict, free=False, save_source_maps=True):
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

        """

        if self.roi.has_source(name):
            msg = 'Source %s already exists.' % name
            self.logger.error(msg)
            raise Exception(msg)

        utils.delete_source_map(self._srcmap_file,name)

        src = self.roi.create_source(name,src_dict)
        self.make_template(src, self.config['file_suffix'])
        
        if self._like is None:
            return

        self._update_srcmap_file([src], True)

        pylike_src = self._create_source(src,free=free)        
        self.like.addSource(pylike_src)
        self.like.syncSrcParams(str(name))
        self.like.logLike.buildFixedModelWts()
        if save_source_maps:
            self.like.logLike.saveSourceMaps(self._srcmap_file)

    def _create_source(self, src, free=False):
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
        elif src['SpatialType'] == 'SpatialGaussian':
            sm = pyLike.SpatialGaussian(src.skydir.ra.deg, src.skydir.dec.deg,
                                        src['SpatialWidth'])
            pylike_src = pyLike.DiffuseSource(sm,
                                              self.like.logLike.observation(),
                                              False)
            
        elif src['SpatialType'] == 'MapCubeFunction':
            mcf = pyLike.MapCubeFunction2(str(src['Spatial_Filename']))
            pylike_src = pyLike.DiffuseSource(mcf,
                                              self.like.logLike.observation(),
                                              False)
        else:
            raise Exception(
                'Unrecognized spatial type: %s' % src['SpatialType'])

        pl = pyLike.SourceFactory_funcFactory().create(src['SpectrumType'])

        for k, v in src.spectral_pars.items():

            par = pl.getParam(k)

            vmin = min(float(v['value']), float(v['min']))
            vmax = max(float(v['value']), float(v['max']))

            par.setValue(float(v['value']))
            par.setBounds(vmin, vmax)
            par.setScale(float(v['scale']))

            if 'free' in v and int(v['free']) != 0:
                par.setFree(True)
            else:
                par.setFree(False)
            pl.setParam(par)

        pylike_src.setSpectrum(pl)
        pylike_src.setName(str(src.name))

        # Initialize source as free/fixed
        pylike_src.spectrum().normPar().setFree(free)

        return pylike_src
        
            
    def delete_source(self, name, save_template=True, delete_source_map=False):

        src = self.roi.get_source_by_name(name, True)

        self.logger.debug('Deleting source %s' % (name))

        if self.like is not None:

            normPar = self.like.normPar(name)
            isFree = normPar.isFree()
            
            if str(src.name) in self.like.sourceNames():
                self.like.deleteSource(str(src.name))
                self.like.logLike.eraseSourceMap(str(src.name))
                
            if not isFree:            
                self.like.logLike.buildFixedModelWts()
                    
        if not save_template and 'Spatial_Filename' in src and \
                src['Spatial_Filename'] is not None and \
                os.path.isfile(src['Spatial_Filename']):
            os.remove(src['Spatial_Filename'])

        self.roi.delete_sources([src])

        if delete_source_map:
            utils.delete_source_map(self._srcmap_file,name)
            
        return src

    def delete_sources(self, srcs):
        for s in srcs:
            if self.like is not None:
                self.like.deleteSource(str(s.name))
                self.like.logLike.eraseSourceMap(str(s.name))
        self._roi.delete_sources(srcs)

    def set_edisp_flag(self, name, flag=True):
        src = self.roi.get_source_by_name(name, True)
        name = src.name
        self.like[name].src.set_edisp_flag(flag)

    def setEnergyRange(self, emin, emax):
        """Set the energy range of the analysis.
        
        """
        
        if emin is None:
            emin = self.energies[0]

        if emax is None:
            emax = self.energies[-1]
            
        imin = int(utils.val_to_edge(self.energies, emin)[0])
        imax = int(utils.val_to_edge(self.energies, emax)[0])

        if imin - imax == 0:
            imin = len(self.energies) - 1
            imax = len(self.energies) - 1

        self.like.selectEbounds(int(imin), int(imax))

        return np.array([self.energies[imin], self.energies[imax]])

    def counts_map(self):
        """Return 3-D counts map for this component as a Map object.

        Returns
        -------
        map : `~fermipy.utils.MapBase`

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
        """Return the model counts map for a single source, a list of
        sources, or for the sum of all sources in the ROI.
        
        Parameters
        ----------
        name : str

           Parameter controlling the set of sources for which the
           model counts map will be calculated.  If name=None a
           model map will be generated for all sources in the ROI.

        exclude : str or list of str

           Source or list of sources that will be excluded when
           calculating the model map.

        Returns
        -------

        map : `~fermipy.utils.Map`

           A map object containing the counts and WCS projection.
                   
        """
        if self.projtype == "WCS":
            v = pyLike.FloatVector(self.npix ** 2 * self.enumbins)
        elif self.projtype == "HPX":
            v = pyLike.FloatVector(self._proj.npix * self.enumbins)
        else:
            raise Exception("Unknown projection type %s"%self.projtype)

        if exclude is None:
            exclude = []
        elif not isinstance(exclude, list):
            exclude = [exclude]

        excluded_srcnames = []
        for i, t in enumerate(exclude):
            srcs = self.roi.get_source_by_name(t)
            excluded_srcnames += [s.name for s in srcs]
            
        if not hasattr(self.like.logLike, 'loadSourceMaps'):
            # Update fixed model
            self.like.logLike.buildFixedModelWts()
            # Populate source map hash
            self.like.logLike.buildFixedModelWts(True)
        elif (name is None or name =='all') and not exclude:
            self.like.logLike.loadSourceMaps()

        src_names = []
        if ((name is None) or (name == 'all')) and exclude:
            for name in self.like.sourceNames():
                if name in excluded_srcnames:
                    continue
                src_names += [name]
        elif name == 'diffuse':           
            for src in self.roi.sources:
                if not src.diffuse:
                    continue
                if src.name in excluded_srcnames:
                    continue
                src_names += [src.name]
        elif isinstance(name, list):
            for n in name:
                src = self.roi.get_source_by_name(n, True)
                if src.name in excluded_srcnames:
                    continue
                src_names += [src.name]
        elif name is not None:
            src = self.roi.get_source_by_name(name, True)
            src_names += [src.name]

        if len(src_names) == 0:
            self.like.logLike.computeModelMap(v)
        elif not hasattr(self.like.logLike, 'setSourceMapImage'):            
            for s in src_names:
                model = self.like.logLike.sourceMap(str(s))
                self.like.logLike.updateModelMap(v, model)
        else:
            vsum = np.zeros(v.size())
            for s in src_names:
                vtmp = pyLike.FloatVector(v.size())
                self.like.logLike.computeModelMap(str(s),vtmp)
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
                "Did not recognize projection type %s" % self.projtype)

    def model_counts_spectrum(self, name, emin, emax):

        cs = np.array(self.like.logLike.modelCountsSpectrum(name))
        imin = utils.val_to_edge(self.energies, emin)[0]
        imax = utils.val_to_edge(self.energies, emax)[0]
        if imax <= imin: raise Exception('Invalid energy range.')
        return cs[imin:imax]

    def setup(self, overwrite=False):
        """Run pre-processing step for this component.

        Parameters
        ----------

        overwrite : bool

           Run all pre-processing steps even if the output file of
           that step is present in the working directory.  By default
           this function will skip any steps for which the output file
           already exists.
        """

        self.logger.info("Running setup for Analysis Component: " +
                         self.name)

        srcmdl_file = self._srcmdl_file
        roi_center = self.roi.skydir

        # Run gtselect and gtmktime
        kw_gtselect = dict(infile=self.config['data']['evfile'],
                           outfile=self._ft1_file,
                           ra=roi_center.ra.deg, dec=roi_center.dec.deg,
                           rad=self.config['selection']['radius'],
                           convtype=self.config['selection']['convtype'],
                           evtype=self.config['selection']['evtype'],
                           evclass=self.config['selection']['evclass'],
                           tmin=self.config['selection']['tmin'],
                           tmax=self.config['selection']['tmax'],
                           emin=self.config['selection']['emin'],
                           emax=self.config['selection']['emax'],
                           zmax=self.config['selection']['zmax'],
                           chatter=self.config['logging']['chatter'])

        kw_gtmktime = dict(evfile=self._ft1_file,
                           outfile=self._ft1_filtered_file,
                           scfile=self.config['data']['scfile'],
                           roicut=self.config['selection']['roicut'],
                           filter=self.config['selection']['filter'])

        if not os.path.isfile(self._ft1_file) or overwrite:
            run_gtapp('gtselect', self.logger, kw_gtselect)
            if self.config['selection']['roicut'] == 'yes' or \
                            self.config['selection']['filter'] is not None:
                run_gtapp('gtmktime', self.logger, kw_gtmktime)
                os.system(
                    'mv %s %s' % (self._ft1_filtered_file, self._ft1_file))
        else:
            self.logger.debug('Skipping gtselect')

        # Run gtltcube
        kw = dict(evfile=self._ft1_file,
                  scfile=self.config['data']['scfile'],
                  outfile=self._ltcube,
                  zmax=self.config['selection']['zmax'])

        if self.config['data']['ltcube'] is not None:

            self._ltcube = os.path.expandvars(self.config['data']['ltcube'])
            if not os.path.isfile(self._ltcube):
                self._ltcube = os.path.join(self.config['fileio']['workdir'],
                                            self._ltcube)

            if not os.path.isfile(self._ltcube):
                raise Exception('Invalid livetime cube: %s' % self._ltcube)

        elif not os.path.isfile(self._ltcube) or overwrite:
            run_gtapp('gtltcube', self.logger, kw)
        else:
            self.logger.debug('Skipping gtltcube')

        self.logger.debug('Loading LT Cube %s' % self._ltcube)
        self._ltc = irfs.LTCube.create(self._ltcube)

        self.logger.debug('Creating PSF model')
        self._psf = irfs.PSFModel(self.roi.skydir, self._ltc,
                                  self.config['gtlike']['irfs'],
                                  self.config['selection']['evtype'],
                                  self.energies)

        # Run gtbin
        if self.projtype == "WCS":
            kw = dict(algorithm='ccube',
                      nxpix=self.npix, nypix=self.npix,
                      binsz=self.config['binning']['binsz'],
                      evfile=self._ft1_file,
                      outfile=self._ccube_file,
                      scfile=self.config['data']['scfile'],
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
                      evfile=self._ft1_file,
                      outfile=self._ccube_file,
                      scfile=self.config['data']['scfile'],
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
                'Unknown projection type, %s. Choices are WCS or HPX' %
                self.projtype)

        if not os.path.isfile(self._ccube_file) or overwrite:
            run_gtapp('gtbin', self.logger, kw)
        else:
            self.logger.debug('Skipping gtbin')

        evtype = self.config['selection']['evtype']

        if self.config['gtlike']['irfs'] == 'CALDB':
            if self.projtype == "HPX":
                cmap = None
            else:
                cmap = self._ccube_file
        else:
            cmap = 'none'

        # Run gtexpcube2
        kw = dict(infile=self._ltcube, cmap=cmap,
                  ebinalg='LOG',
                  emin=self.config['selection']['emin'],
                  emax=self.config['selection']['emax'],
                  enumbins=self._enumbins,
                  outfile=self._bexpmap_file, proj='CAR',
                  nxpix=360, nypix=180, binsz=1,
                  xref=0.0, yref=0.0,
                  evtype=evtype,
                  irfs=self.config['gtlike']['irfs'],
                  coordsys=self.config['binning']['coordsys'],
                  chatter=self.config['logging']['chatter'])
        
        if not os.path.isfile(self._bexpmap_file) or overwrite:
            run_gtapp('gtexpcube2', self.logger, kw)
        else:
            self.logger.debug('Skipping gtexpcube')

        if self.projtype == "WCS":
            kw = dict(infile=self._ltcube, cmap='none',
                      ebinalg='LOG',
                      emin=self.config['selection']['emin'],
                      emax=self.config['selection']['emax'],
                      enumbins=self._enumbins,
                      outfile=self._bexpmap_roi_file, proj='CAR',
                      nxpix=self.npix, nypix=self.npix,
                      binsz=self.config['binning']['binsz'],
                      xref=self._xref, yref=self._yref,
                      evtype=self.config['selection']['evtype'],
                      irfs=self.config['gtlike']['irfs'],
                      coordsys=self.config['binning']['coordsys'],
                      chatter=self.config['logging']['chatter'])
            if not os.path.isfile(self._bexpmap_roi_file) or overwrite:
                run_gtapp('gtexpcube2', self.logger, kw)
            else:
                self.logger.debug('Skipping local gtexpcube')
        elif self.projtype == "HPX":
            self.logger.debug('Skipping local gtexpcube for HEALPix')
        else:
            raise Exception(
                "Did not recognize projection type %s" % self.projtype)

        # Make spatial templates for extended sources
        for s in self.roi.sources:
            if s.diffuse:
                continue
            if not s.extended:
                continue
            self.make_template(s, self.config['file_suffix'])

        # Write ROI XML
        #if not os.path.isfile(srcmdl_file):
        self.roi.write_xml(srcmdl_file)

        # Run gtsrcmaps
        kw = dict(scfile=self.config['data']['scfile'],
                  expcube=self._ltcube,
                  cmap=self._ccube_file,
                  srcmdl=srcmdl_file,
                  bexpmap=self._bexpmap_file,
                  outfile=self._srcmap_file,
                  irfs=self.config['gtlike']['irfs'],
                  evtype=evtype,
                  rfactor=self.config['gtlike']['rfactor'],
                  #                   resample=self.config['resample'],
                  minbinsz=self.config['gtlike']['minbinsz'],
                  chatter=self.config['logging']['chatter'],
                  emapbnds='no')

        if not os.path.isfile(self._srcmap_file) or overwrite:
            if self.config['gtlike']['srcmap'] and self.config['gtlike']['bexpmap']:
                self.make_scaled_srcmap()
            else:
                run_gtapp('gtsrcmaps', self.logger, kw)
        else:
            self.logger.debug('Skipping gtsrcmaps')

        # Create templates for extended sources
        self._update_srcmap_file(None, True)

        self._create_binned_analysis()
        
        self.logger.info(
            'Finished setup for Analysis Component: %s' % self.name)

    def _create_binned_analysis(self, xmlfile=None):

        srcmdl_file = self._srcmdl_file
        if xmlfile is not None:
            srcmdl_file = self.get_model_path(xmlfile)
            
        # Create BinnedObs
        self.logger.debug('Creating BinnedObs')
        kw = dict(srcMaps=self._srcmap_file, expCube=self._ltcube,
                  binnedExpMap=self._bexpmap_file,
                  irfs=self.config['gtlike']['irfs'])
        self.logger.debug(kw)

        self._obs = ba.BinnedObs(**kw)

        # Create BinnedAnalysis
        self.logger.debug('Creating BinnedAnalysis')
        kw = dict(srcModel=srcmdl_file,
                  optimizer='MINUIT',
                  convolve=self.config['gtlike']['convolve'],
                  resample=self.config['gtlike']['resample'],
                  minbinsz=self.config['gtlike']['minbinsz'],
                  resamp_fact=self.config['gtlike']['rfactor'])
        self.logger.debug(kw)

        self._like = BinnedAnalysis(binnedData=self._obs, **kw)

#        print self.like.logLike.use_single_fixed_map()
#        self.like.logLike.set_use_single_fixed_map(False)
#        print self.like.logLike.use_single_fixed_map()

        if self.config['gtlike']['edisp']:
            self.logger.debug('Enabling energy dispersion')
            self.like.logLike.set_edisp_flag(True)

        for s in self.config['gtlike']['edisp_disable']:

            if not self.roi.has_source(s):
                continue
            
            self.logger.debug('Disabling energy dispersion for %s' % s)
            self.set_edisp_flag(s, False)

        # Recompute fixed model weights
        self.like.logLike.buildFixedModelWts()
        self.like.logLike.saveSourceMaps(self._srcmap_file)

    def make_scaled_srcmap(self):
        """Make an exposure cube with the same binning as the counts map."""

        self.logger.info('Computing scaled source map.')

        bexp0 = pyfits.open(self._bexpmap_roi_file)
        bexp1 = pyfits.open(self.config['gtlike']['bexpmap'])
        srcmap = pyfits.open(self.config['gtlike']['srcmap'])

        if bexp0[0].data.shape != bexp1[0].data.shape:
            raise Exception('Wrong shape for input exposure map file.')

        bexp_ratio = bexp0[0].data / bexp1[0].data

        self.logger.info(
            'Min/Med/Max exposure correction: %f %f %f' % (np.min(bexp_ratio),
                                                           np.median(
                                                               bexp_ratio),
                                                           np.max(bexp_ratio)))

        for hdu in srcmap[1:]:

            if hdu.name == 'GTI': continue
            if hdu.name == 'EBOUNDS': continue
            hdu.data *= bexp_ratio

        srcmap.writeto(self._srcmap_file, clobber=True)

    def restore_counts_maps(self):

        cmap = Map.create_from_fits(self._ccube_file)

        if hasattr(self.like.logLike, 'setCountsMap'):
            self.like.logLike.setCountsMap(np.ravel(cmap.counts.astype(float)))

        utils.update_source_maps(self._srcmap_file, {'PRIMARY': cmap.counts},
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

        utils.update_source_maps(self._srcmap_file, {'PRIMARY': data},
                                 logger=self.logger)
        utils.write_fits_image(data, self.wcs, self._ccubemc_file)
        
    def write_model_map(self, model_name=None, name=None):
        """Save counts model map to a FITS file.

        """

        if model_name is None:
            suffix = self.config['file_suffix']
        else:
            suffix = '_%s%s' % (model_name, self.config['file_suffix'])

        self.logger.info('Generating model map for component %s.' % self.name)

        outfile = os.path.join(self.config['fileio']['workdir'],
                               'mcube%s.fits' % (suffix))
        
        cmap = self.model_counts_map(name)

        if self.projtype == "HPX":
            utils.write_hpx_image(cmap.counts, cmap.hpx, outfile)
        elif self.projtype == "WCS":
            utils.write_fits_image(cmap.counts, cmap.wcs, outfile)
        else:
            raise Exception(
                "Did not recognize projection type %s" % self.projtype)
        return cmap

    def make_template(self, src, suffix):

        if 'SpatialModel' not in src:
            return
        elif src['SpatialModel'] == 'GaussianSource':
            template_file = os.path.join(self.config['fileio']['workdir'],
                                         '%s_template_gauss_%05.3f%s.fits' % (
                                             src.name, src['SpatialWidth'],
                                             suffix))
            utils.make_gaussian_spatial_map(src.skydir, src['SpatialWidth'],
                                            template_file, npix=500)
            src['Spatial_Filename'] = template_file
        elif src['SpatialModel'] == 'DiskSource':
            template_file = os.path.join(self.config['fileio']['workdir'],
                                         '%s_template_disk_%05.3f%s.fits' % (
                                             src.name, src['SpatialWidth'],
                                             suffix))
            utils.make_disk_spatial_map(src.skydir, src['SpatialWidth'],
                                        template_file, npix=500)
            src['Spatial_Filename'] = template_file        

    def _update_srcmap_file(self, sources=None, overwrite=False):
        """Check the contents of the source map file and generate
        source maps for any components that are not present."""

        if not os.path.isfile(self._srcmap_file):
            return

        hdulist = pyfits.open(self._srcmap_file)
        hdunames = [hdu.name.upper() for hdu in hdulist]

        srcmaps = {}

        if sources is None:
            sources = self.roi.sources

        for s in sources:

            if s.diffuse:
                continue
            if 'SpatialModel' not in s:
                continue
            if s['SpatialModel'] in ['PointSource', 'Gaussian',
                                     'SpatialMap','SpatialGaussian',
                                     'SpatialDisk']:
                continue
            if s.name.upper() in hdunames and not overwrite:
                continue

            self.logger.debug('Creating source map for %s' % s.name)

            xpix, ypix = utils.skydir_to_pix(s.skydir, self._skywcs)
            xpix -= (self.npix-1.0)/2.
            ypix -= (self.npix-1.0)/2.
            
            k = utils.make_srcmap(s.skydir, self._psf, s['SpatialModel'],
                                  s['SpatialWidth'],
                                  npix=self.npix, xpix=xpix, ypix=ypix,
                                  cdelt=self.config['binning']['binsz'],
                                  rebin=8)
            
            srcmaps[s.name] = k

        if srcmaps:
            self.logger.debug(
                'Updating source map file for component %s.' % self.name)
            utils.update_source_maps(self._srcmap_file, srcmaps,
                                     logger=self.logger)

    def _update_srcmap(self, name, skydir, spatial_model, spatial_width):

        xpix, ypix = utils.skydir_to_pix(skydir, self._skywcs)
        xpix -= (self.npix-1.0)/2.
        ypix -= (self.npix-1.0)/2.

        k = utils.make_srcmap(self.roi.skydir, self._psf, spatial_model,
                              spatial_width,
                              npix=self.npix, xpix=xpix, ypix=ypix,
                              cdelt=self.config['binning']['binsz'],
                              rebin=8)

        self.like.logLike.setSourceMapImage(str(name),np.ravel(k))        
        #src_map = self.like.logLike.sourceMap(str(name))
        #src_map.setImage(np.ravel(k))

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
            srcmdl = self._srcmdl_file
        else:
            srcmdl = self.get_model_path(model_name)

        if not os.path.isfile(srcmdl):
            raise Exception("Model file does not exist: %s" % srcmdl)

        if model_name is None:
            suffix = self.config['file_suffix']
        else:
            suffix = '_%s%s' % (model_name, self.config['file_suffix'])

        outfile = os.path.join(self.config['fileio']['workdir'],
                               'mcube%s.fits' % (suffix))

        # May consider generating a custom source model file
        if not os.path.isfile(outfile):

            kw = dict(srcmaps=self._srcmap_file,
                      srcmdl=srcmdl,
                      bexpmap=self._bexpmap_file,
                      outfile=outfile,
                      expcube=self._ltcube,
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
        self.like.logLike.reReadXml(xmlfile)
        if not self.like.logLike.fixedModelUpdated():
            self.like.logLike.buildFixedModelWts()

    def write_xml(self, xmlfile):
        """Write the XML model for this analysis component."""

        xmlfile = self.get_model_path(xmlfile)
        self.logger.info('Writing %s...' % xmlfile)
        self.like.writeXml(xmlfile)

    def get_model_path(self, name):
        """Infer the path to the XML model name."""

        name, ext = os.path.splitext(name)
        ext = '.xml'
        xmlfile = name + self.config['file_suffix'] + ext
        xmlfile = resolve_path(xmlfile,
                               workdir=self.config['fileio']['workdir'])

        return xmlfile

    def _tscube_app(self, xmlfile):
        """Run gttscube as an application."""

        xmlfile = self.get_model_path(xmlfile)

        outfile = os.path.join(self.config['fileio']['workdir'],
                               'tscube%s.fits' % (self.config['file_suffix']))

        kw = dict(cmap=self._ccube_file,
                  expcube=self._ltcube,
                  bexpmap=self._bexpmap_file,
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
