#!/usr/bin/env python
#

"""
Interface to Dark Matter spectra
"""
from __future__ import absolute_import, division, print_function

import sys

import yaml
import numpy as np

from astropy import table
from astropy.io import fits

from fermipy import castro
from fermipy import fits_utils
from fermipy.jobs.utils import is_null, is_not_null
from fermipy.stats_utils import create_prior_functor
from fermipy.spectrum import SpectralFunction

from .lnl_norm_prior import LnLFn_norm_prior
from .stack_castro import StackCastroData


class StackSpecTable(object):
    """ Version of the stacking spectral tables in tabular form

    This will make one table for each "spectrum" that is defined
    Each table can consist of a series of spectra with different
    values of the parameters for that spectrum

    """

    def __init__(self, e_table, s_tables, ref_val_dict):
        """ C'tor to build this object from energy binning and spectral values tables.
        """
        self._e_table = e_table
        self._s_tables = s_tables
        self._ref_val_dict = ref_val_dict

    @property
    def ebounds_table(self):
        """Return the energy binning table """
        return self._e_table

    @property
    def spectra_tables(self):
        """Return the dictionary of spectral values table """
        return self._s_tables

    @property
    def ref_val_dict(self):
        """Return the spectral reference values """
        return self._ref_val_dict

    @property
    def spec_names(self):
        """Return the spec to index mapping """
        return sorted(self._s_tables.keys())

    def spectra_table(self, spec_name):
        """Return a particular spectral table"""
        return self._s_tables[spec_name]

    def ref_vals(self, spec_name):
        """Return a dictionary of the references values for a particular spectrum"""
        return self._ref_val_dict[spec_name]

    def ebin_edges(self):
        """Return an array with the energy bin edges
        """
        return np.hstack([self._e_table["E_MIN"].data,
                          self._e_table["E_MAX"].data])

    def ebin_refs(self):
        """Return an array with the energy bin reference energies
        """
        return self._e_table["E_REF"].data


    def check_energy_bins(self, ref_spec, tol=1e-3):
        """ Make sure that the energy binning matches the reference spectrum

        Parameters
        ----------

        ref_spec :
            The reference spectrum

        tol : float
            The maximum allowed relative difference in bin edges


        Returns
        -------

        status : bool
            True if the energy binnings match

        """
        emin_local = self._e_table['E_MIN']
        emax_local = self._e_table['E_MAX']
        try:
            if str(emin_local.unit) == 'keV':
                emin_local = emin_local / 1000.
        except KeyError:
            pass
        try:
            if str(emax_local.unit) == 'keV':
                emax_local = emax_local / 1000.
        except KeyError:
            pass

        if len(emin_local) != len(ref_spec.emin):
            return False
        if len(emax_local) != len(ref_spec.emax):
            return False
        if (np.abs(emin_local - ref_spec.emin) > tol * emin_local).any():
            return False
        if (np.abs(emax_local - ref_spec.emax) > tol * emax_local).any():
            return False
        return True


    def spectra(self, spec_name, par_dict, spec_type):
        """Return the spectra for a particular spec_name, parameters and spectral type
        """
        try:
            s_table = self._s_tables[spec_name]
        except:
            raise KeyError("No spectral table for %s " % spec_name)
        nrows = len(s_table)
        mask = np.ones((nrows), dtype=bool)
        for par_name, par_val in par_dict.items():
            mask *= np.abs(s_table["par_%s" % par_name] - par_val) < 1e-9
        if mask.sum() < 1:
            raise ValueError("%i spectra match the conditions: %s" % (mask.sum(), str(par_dict)))
        spec_vals = s_table[mask]["ref_%s" % spec_type].data
        return spec_vals


    def par_vals(self, spec_name, par_name, other_pars=None):
        """Return the array of xscan_vals for a given spec
        """
        try:
            s_table = self._s_tables[spec_name]
        except:
            raise KeyError("No spectral table for %s " % spec_name)
        nrows = len(s_table)
        mask = np.ones((nrows), dtype=bool)
        if other_pars is None:
            other_pars = {}
        for opar_name, opar_val in other_pars.items():
            mask *= np.abs(s_table["par_%s" % opar_name] - opar_val) < 1e-9
        if mask.sum() == 0:
            raise ValueError("%i spectra match the conditions: %s" % (mask.sum(), str(other_pars)))
        return s_table[mask]["par_%s" % par_name]


    def write_fits(self, filepath, clobber=False):
        """ Write this object to a FITS file

        Paramters
        ---------

        filepath : str
            Path to output file

        clobber : bool
            Flag to allow overwriting existing files

        """
        tables = [self._e_table]
        tablenames = ["EBOUNDS"]
        ref_cards = [{}]
        for key, val in self._s_tables.items():
            tables.append(val)
            tablenames.append(key)
            ref_cards.append(self._ref_val_dict[key])
        fits_utils.write_tables_to_fits(filepath, tables,
                                        clobber=clobber,
                                        namelist=tablenames,
                                        cardslist=ref_cards)


    @staticmethod
    def extract_energy_limits(configfile):
        """ Build a StackSpecTable object from a yaml config file

        Parameters
        ----------

        configfile : str
            Fermipy yaml configuration file

        Returns
        -------

        emins: `np.array`
            Energy bin lower edges

        emaxs: `np.array`
            Energy bin upper edges

        """
        config = yaml.safe_load(open(configfile))

        # look for components
        components = config.get('components', [config])

        emins = np.array([])
        emaxs = np.array([])

        binsperdec = config['binning'].get('binsperdec', None)

        for comp in components:
            try:
                emin = comp['selection']['emin']
                emax = comp['selection']['emax']
                logemin = np.log10(emin)
                logemax = np.log10(emax)
            except AttributeError:
                logemin = comp['selection']['logemin']
                logemax = comp['selection']['logemax']
                emin = np.power(10., logemin)
                emax = np.power(10., logemax)

            try:
                nebins = comp['binning'].get('enumbins', None)
            except KeyError:
                nebins = np.round(binsperdec * np.log10(emax / emin))

            if nebins is None:
                nebins = np.round(comp['binning']['binsperdec'] * np.log10(emax / emin))

            ebin_edges = np.logspace(logemin, logemax, nebins + 1)
            emins = np.append(emins, ebin_edges[:-1])
            emaxs = np.append(emaxs, ebin_edges[1:])

        return (emins, emaxs)


    @staticmethod
    def get_ref_vals(filepath, spec):
        """ Extract the reference values from a FITS header

        Paramters
        ---------

        filepath : str
            Path to input file

        spec : str
            Name of the spectrum

        Returns
        -------

        REF_NORM : float
            Reference value of stacking variable

        REF_ASTRO : float
            Reference value of astrophysical factor

        """
        fin = fits.open(filepath)
        hdu = fin[spec]
        hin = hdu.header
        dref = {"REF_NORM": hin["REF_NORM"],
                "REF_ASTRO": hin["REF_ASTRO"]}
        fin.close()
        return dref

    @staticmethod
    def build_func_info(emin, emax, sdict, norm_type):
        """Create a stacking spectrum table.
        """
        specType = sdict['SpectrumType']
        spec_min = sdict['spec_var']['min']
        spec_max = sdict['spec_var']['max']
        spec_nstep = sdict['spec_var']['nstep']
        scale = sdict['scale_var']['value']
        fn = SpectralFunction.create_functor(specType, norm_type, emin, emax, scale=scale)
        fn.params[0] = sdict['norm_var']['value']
        fn.params[1] = sdict['spec_var']['value']
        o_dict = dict(functor=fn.spectral_fn,
                      params=fn.params,
                      xscan_vals=np.linspace(spec_min, spec_max, spec_nstep),
                      xscan_name=sdict['spec_var']['name'])
        return o_dict

    @staticmethod
    def make_ebounds_table(emin, emax, eref):
        """ Construct the energy bounds table

        Returns
        -------

        table : `astropy.table.Table`
            The table has these columns and one row per energy bin

        E_MIN : float
            Energy bin lower edge

        E_MAX : float
            Energy bin upper edge

        E_REF : float
            Reference energy for bin, typically geometric mean
            of bin edges

        """
        col_emin = table.Column(name="E_MIN", dtype=float, unit="MeV", data=emin)
        col_emax = table.Column(name="E_MAX", dtype=float, unit="MeV", data=emax)
        col_eref = table.Column(name="E_REF", dtype=float, unit="MeV", data=eref)

        tab = table.Table(data=[col_emin, col_emax, col_eref])
        return tab


    @staticmethod
    def make_spectra_tables(nebins, spec_dict):
        """Construct the spectral values table

        Returns
        -------

        table : `astropy.table.Table`
            The table has these columns

        ref_<par_names> : float
            The value of other parameters (aside from the normalization)

        ref_spec : int
            The index of the stacking spectrum for this row

        ref_dnde : array
            The reference differential photon flux fpr each energy [ph / (MeV cm2 s)]

        ref_flux : array
            The reference integral photon flux for each energy [ph / (cm2 s)]

        ref_eflux : array
            The reference integral energy flux for each energy [MeV / (cm2 s)]

        """
        table_dict = {}
        for spec, spec_data in spec_dict.items():
            par_dict = spec_data['params']
            cols = []
            for par_name, par_value in par_dict.items():
                col_par = table.Column(name="par_%s" % par_name, dtype=float, data=par_value)
                cols.append(col_par)
            col_dnde = table.Column(name="ref_dnde", dtype=float, shape=nebins, unit="ph / (MeV cm2 s)",
                                    data=spec_data['dnde'])
            col_flux = table.Column(name="ref_flux", dtype=float, shape=nebins, unit="ph / (cm2 s)",
                                    data=spec_data['flux'])
            col_eflux = table.Column(name="ref_eflux", dtype=float, shape=nebins, unit="MeV / (cm2 s)",
                                     data=spec_data['eflux'])
            cols += [col_dnde, col_flux, col_eflux]
            otable = table.Table(data=cols)
            table_dict[spec] = otable
        return table_dict


    @classmethod
    def create_from_fits(cls, filepath):
        """ Build a StackSpecTable object from a FITS file

        Paramters
        ---------

        filepath : str
            Path to the input file

        Returns
        -------

        output : `StackSpecTable`
            The newly created object

        """
        e_table = table.Table.read(filepath, "EBOUNDS")
        fin = fits.open(filepath)
        s_tables = {}
        d_refs = {}
        for hdu in fin[1:]:
            if hdu.name in ["EBOUNDS"]:
                continue
            s_table = table.Table.read(filepath, hdu.name)
            s_tables[hdu.name] = s_table
            d_refs[hdu.name] = cls.get_ref_vals(filepath, hdu.name)
        return cls(e_table, s_tables, d_refs)


    @classmethod
    def create_from_data(cls, emins, emaxs, erefs, data_dict, ref_vals):
        """ Build a StackSpecTable object from energy binning, reference values and spectral values

        Paramters
        ---------
        emins: `numpy.array`
             Energy bin lower edges

        emaxs: `numpy.array`
             Energy bin upper edges

        erefs: `numpy.array`
             Refences energies (typically geometric mean of energy bins)

        data_dict : dict
             Dictionay of tables, keyed by spectra name

        ref_vals : dict
             Dictionary of reference values, keyed by spectra name

        Returns
        -------

        output : `StackSpecTable`
            The newly created object
        """
        e_table = cls.make_ebounds_table(emins, emaxs, erefs)
        s_tables = cls.make_spectra_tables(len(emins), data_dict)
        return cls(e_table, s_tables, ref_vals)


    @classmethod
    def create(cls, emin, emax, specs):
        """Create a stacking spectrum table.

        Parameters
        ----------

        emin : `~numpy.ndarray`
            Low bin edges.

        emax : `~numpy.ndarray`
            High bin edges.

        specs : dict
            List of spectral information dictionary

        Returns
        -------

        output : `StackSpecTable`
            The newly created object

        """
        ebin_edges = np.concatenate((emin, emax[-1:]))
        evals = np.sqrt(ebin_edges[:-1] * ebin_edges[1:])

        spec_dicts = {}
        ref_vals = {}

        for spec_name, spec_info in specs.items():
            func_info = cls.build_func_info(emin, emax, spec_info, norm_type='eflux')
            params = func_info['params']
            xscan_vals = func_info['xscan_vals']
            xscan_name = func_info['xscan_name']
            stackf = func_info['functor']
            dnde = stackf.dnde(evals, (params[0], xscan_vals)).T
            flux = stackf.flux(emin, emax, (params[0], xscan_vals)).T
            eflux = stackf.eflux(emin, emax, (params[0], xscan_vals)).T
            spec_dict = {"dnde": dnde,
                         "flux": flux,
                         "eflux": eflux,
                         "xscan_vals": xscan_vals,
                         "xscan_name": xscan_name}
            spec_dicts[spec_name] = spec_dict
            ref_vals[spec_name] = func_info["ref_vals"]

        return cls.create_from_data(ebin_edges[0:-1], ebin_edges[1:],
                                    evals, spec_dicts, ref_vals)


    @classmethod
    def create_from_config(cls, configfile, specfile):
        """ Build a StackSpecTable object from a yaml config file

        Parameters
        ----------

        configfile : str
            Fermipy yaml configuration file

        specfile : str
            yaml configuration file with spectra

        Returns
        -------

        output : `StackSpecTable`
            The newly created object

        """
        emins, emaxs = cls.extract_energy_limits(configfile)
        specs = yaml.safe_load(open(specfile))
        return cls.create(emins, emaxs, specs)



    @staticmethod
    def compute_castro_data(castro_data, spec_vals, xscan_vals, **kwargs):
        """ Convert CastroData object, i.e., Likelihood as a function of
        flux and energy flux, to a StackCastroData object, i.e., Likelihood as
        a function of stacking normalization and index

        Parameters
        ----------

        castro_data : `CastroData`
            Input data

        spec_vals : `numpy.array`
            Input spectra

        xscan_vals : `numpy.array`
            Values for scan variables


        Returns
        -------

        output : `StackCastroData`
            The stacking-space likelihood curves


        """
        ref_stack = kwargs.get('ref_stack', 1.)
        norm_factor = kwargs.pop('norm_factor', 1.)
        spec_name = kwargs.pop('spec_name')
        astro_prior = kwargs.pop('astro_prior')
        n_xval = len(xscan_vals)
        n_yval = kwargs.get('nystep', 200)

        norm_limits = castro_data.getLimits(1e-5)
        # This puts the spectrum in units of the reference spectrum
        # This means that the scan values will be expressed
        # In units of the reference spectra as well
        spec_vals /= norm_factor

        norm_vals = np.ndarray((n_xval, n_yval))
        dll_vals = np.ndarray((n_xval, n_yval))
        mle_vals = np.ndarray((n_xval))
        nll_offsets = np.ndarray((n_xval))

        scan_mask = np.ones((n_xval), bool)
        # for i, index in enumerate(xscan_vals):
        for i in range(n_xval):
            max_ratio = 1. / ((spec_vals[i] / norm_limits).max())
            log_max_ratio = np.log10(max_ratio)
            norm_vals[i][0] = 10**(log_max_ratio - 5)
            norm_vals[i][1:] = np.logspace(log_max_ratio - 4, log_max_ratio + 4,
                                           n_yval - 1)
            test_vals = (np.expand_dims(
                spec_vals[i], 1) * (np.expand_dims(norm_vals[i], 1).T))
            dll_vals[i, 0:] = castro_data(test_vals)
            mle_vals[i] = norm_vals[i][dll_vals[i].argmin()]
            nll_offsets[i] = dll_vals[i].min()
            dll_vals[i] -= nll_offsets[i]

            msk = np.isfinite(dll_vals[i])
            if not msk.any():
                print (
                    "Skipping scan value %0.2e for spec %s" %
                    (xscan_vals[i], spec_name))
                scan_mask[i] = False
                continue

            if is_not_null(astro_prior):
                try:
                    lnlfn = castro.LnLFn(norm_vals[i], dll_vals[i], 'dummy')
                    lnlfn_prior = LnLFn_norm_prior(lnlfn, astro_prior)
                    dll_vals[i, 0:] = lnlfn_prior(norm_vals[i])
                    nll_offsets[i] = dll_vals[i].min()
                except ValueError:
                    print (
                        "Skipping index %0.2e for spec %s" %
                        (xscan_vals[i], spec_name))
                    scan_mask[i] = False
                    dll_vals[i, 0:] = np.nan * np.ones((n_yval))
                    nll_offsets[i] = np.nan

        kwcopy = kwargs.copy()
        kwcopy['xscan_vals'] = xscan_vals[scan_mask]
        kwcopy['astro_prior'] = astro_prior

        # Here we convert the normalization values to standard units
        norm_vals *= ref_stack
        stack_castro = StackCastroData(norm_vals[scan_mask], dll_vals[scan_mask],
                                       nll_offsets[scan_mask], **kwcopy)
        return stack_castro


    def convert_castro_data(self, castro_data, spec_name, **kwargs):
        """ Convert CastroData object, i.e., Likelihood as a function of
        flux and energy flux, to a StackCastroData object, i.e., Likelihood as
        a function of stacking normalization and index

        Parameters
        ----------

        castro_data : `CastroData`
            Input data

        spec_name : str
            Name of the stacking spectra


        Keyword arguments
        -----------------

        other_pars : dict
           Values for the other paramters

        norm_type : str
            Type of spectral normalization to use for the conversion

        astro_factor : dict or float or None
            Nuisance factor used to make the conversion.

            If astro_factor is None, it will use the reference value
            If astro_factor is a float, it will use that
            If astro_factor is a dict, it will use that to create a prior

        Returns
        -------

        output : `StackCastroData`
            The stacking-space likelihood curves

        """
        xscan_name = kwargs.get('xscan_name', 'index')
        astro_factor = kwargs.get('astro_factor', None)
        other_pars = kwargs.get('other_pars', {})
        spec_type = kwargs.get('spec_type', 'eflux')

        if not self.check_energy_bins(castro_data.refSpec):
            raise ValueError("CastroData energy binning does not match")

        xscan_vals = self.par_vals(spec_name, xscan_name, other_pars)
        spec_vals = self.spectra(spec_name, other_pars, spec_type)

        # Get the reference values
        ref_vals = self.ref_vals(spec_name)
        ref_norm = ref_vals["REF_NORM"]

        astro_prior = None
        if is_null(astro_factor):
            # Just use the reference values
            astro_value = ref_norm
            norm_factor = 1.
        elif isinstance(astro_factor, float):
            # Rescale the normalization values
            astro_value = astro_factor
            norm_factor = ref_norm / astro_factor
        elif isinstance(astro_factor, dict):
            astro_value = astro_factor.get('d_value')
            norm_factor = ref_norm / astro_value
            astro_factor['scale'] = astro_value
            astro_functype = astro_factor.get('functype', None)
            if is_null(astro_functype):
                astro_prior = None
            else:
                astro_prior = create_prior_functor(astro_factor)
        else:
            sys.stderr.write(
                "Did not recoginize Astro factor %s %s\n" %
                (astro_factor, type(astro_factor)))

        kwcopy = kwargs.copy()
        kwcopy['astro_prior'] = astro_prior
        kwcopy['spec_name'] = spec_name
        kwcopy['norm_factor'] = norm_factor
        kwcopy['ref_stack'] = ref_vals["REF_STACK"]
        kwcopy['ref_norm'] = ref_norm
        stack_castro = self.compute_castro_data(castro_data, spec_vals, xscan_vals, **kwcopy)
        return stack_castro
