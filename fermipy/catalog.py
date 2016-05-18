from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import numpy as np

from astropy import units as u
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import astropy.io.fits as pyfits

import fermipy
import fermipy.spectrum as spectrum

def add_columns(t0, t1):
    """Add columns of table t1 to table t0."""

    for colname in t1.colnames:
        col = t1.columns[colname]
        if colname in t0.columns:
            continue
        new_col = Column(name=col.name, length=len(t0), dtype=col.dtype)  # ,
        # shape=col.shape)
        t0.add_column(new_col)


def join_tables(t0, t1, key0, key1):
    v0, v1 = t0[key0], t1[key1]
    v0 = np.core.defchararray.strip(v0)
    v1 = np.core.defchararray.strip(v1)
    add_columns(t0, t1)

    # Get mask of elements in t0 that are shared with t0
    m0 = np.in1d(v0, v1)
    idx1 = np.searchsorted(v1, v0)[m0]

    for colname in t1.colnames:
        if colname == 'Source_Name':
            continue
        t0[colname][m0] = t1[colname][idx1]


def strip_columns(tab):
    """Strip whitespace from string columns."""
    for colname in tab.colnames:
        if not tab[colname].dtype.type is np.string_:
            continue
        tab[colname] = np.core.defchararray.strip(tab[colname])


def row_to_dict(row):
    """Convert a table row to a dictionary."""
    o = {}
    for colname in row.colnames:

        if isinstance(row[colname], np.string_):
            o[colname] = str(row[colname])
        else:
            o[colname] = row[colname]

    return o


class Catalog(object):
    """Source catalog object.  This class provides a simple wrapper around
    FITS catalog tables."""
    
    def __init__(self, table, extdir=''):
        self._table = table
        self._extdir = extdir

        if self.table['RAJ2000'].unit is None:
            self._src_skydir = SkyCoord(ra=self.table['RAJ2000']*u.deg,
                                        dec=self.table['DEJ2000']*u.deg)
        else:
            self._src_skydir = SkyCoord(ra=self.table['RAJ2000'],
                                        dec=self.table['DEJ2000'])
        self._radec = np.vstack((self._src_skydir.ra.deg,
                                 self._src_skydir.dec.deg)).T
        self._glonlat = np.vstack((self._src_skydir.galactic.l.deg,
                                   self._src_skydir.galactic.b.deg)).T

        if 'Spatial_Filename' not in self.table.columns:
            self.table['Spatial_Filename'] = Column(dtype='S20',length=len(self.table))
            
        m = self.table['Spatial_Filename'] != ''
        self.table['extended'] = False
        self.table['extended'][m] = True
        self.table['extdir'] = extdir

    @property
    def table(self):
        """Return the `~astropy.table.Table` representation of this
        catalog."""
        return self._table

    @property
    def skydir(self):
        return self._src_skydir

    @property
    def radec(self):
        return self._radec

    @property
    def glonlat(self):
        return self._glonlat

    @staticmethod
    def create(name):

        extname = os.path.splitext(name)[1]
        if extname == '.fits' or extname == '.fit':
            fitsfile = name
            if not os.path.isfile(fitsfile):
                fitsfile = os.path.join(fermipy.PACKAGE_DATA, 'catalogs',
                                        fitsfile)
                
            # Try to guess the catalog type form its name
            if 'gll_psc' in fitsfile:                
                return Catalog3FGL(fitsfile)

            tab = Table.read(fitsfile)
            
            if 'NickName' in tab.columns:
                return Catalog4FGLP(fitsfile)
            else:
                return Catalog(Table.read(fitsfile))
            
        elif name == '3FGL':
            return Catalog3FGL()
        elif name == '2FHL':
            return Catalog2FHL()
        else:
            raise Exception('Unrecognized catalog type.')


class Catalog2FHL(Catalog):
    def __init__(self, fitsfile=None, extdir=None):

        if extdir is None:
            extdir = os.path.join('$FERMIPY_DATA_DIR', 'catalogs',
                                  'Extended_archive_2fhl_v00')

        if fitsfile is None:
            fitsfile = os.path.join(fermipy.PACKAGE_DATA, 'catalogs',
                                    'gll_psch_v08.fit')

        hdulist = pyfits.open(fitsfile)
        table = Table(hdulist['2FHL Source Catalog'].data)
        table_extsrc = Table(hdulist['Extended Sources'].data)

        strip_columns(table)
        strip_columns(table_extsrc)

        join_tables(table, table_extsrc, 'Source_Name', 'Source_Name')

        super(Catalog2FHL, self).__init__(table, extdir)

        self._table['Flux_Density'] = \
            spectrum.PowerLaw.eval_norm(50E3, -np.array(self.table['Spectral_Index']),
                                        50E3, 2000E3,
                                        np.array(self.table['Flux50']))
        self._table['Pivot_Energy'] = 50E3
        self._table['SpectrumType'] = 'PowerLaw'


class Catalog3FGL(Catalog):
    def __init__(self, fitsfile=None, extdir=None):

        if extdir is None:
            extdir = os.path.join('$FERMIPY_DATA_DIR', 'catalogs',
                                  'Extended_archive_v15')

        if fitsfile is None:
            fitsfile = os.path.join(fermipy.PACKAGE_DATA, 'catalogs',
                                    'gll_psc_v16.fit')

        hdulist = pyfits.open(fitsfile)
        table = Table(hdulist['LAT_Point_Source_Catalog'].data)
        #table = Table.read(fitsfile)
        table_extsrc = Table(hdulist['ExtendedSources'].data)

        strip_columns(table)
        strip_columns(table_extsrc)

        self._table_extsrc = table_extsrc

        join_tables(table, table_extsrc, 'Extended_Source_Name', 'Source_Name')

        super(Catalog3FGL, self).__init__(table, extdir)

        m = self.table['SpectrumType'] == 'PLExpCutoff'
        self.table['SpectrumType'][m] = 'PLSuperExpCutoff'

        self.table['TS_value'] = 0.0
        self.table['TS'] = 0.0

        ts_keys = ['Sqrt_TS30_100', 'Sqrt_TS100_300',
                   'Sqrt_TS300_1000', 'Sqrt_TS1000_3000',
                   'Sqrt_TS3000_10000', 'Sqrt_TS10000_100000']

        for k in ts_keys:
            m = np.isfinite(self.table[k])
            self._table['TS_value'][m] += self.table[k][m] ** 2
            self._table['TS'][m] += self.table[k][m] ** 2


class Catalog4FGLP(Catalog):
    """This class supports preliminary releases of the 4FGL catalog.
    Because there is currently no dedicated extended source library
    for 4FGL this class reuses the extended source library from the
    3FGL."""
    
    def __init__(self, fitsfile=None, extdir=None):

        if extdir is None:
            extdir = os.path.join('$FERMIPY_DATA_DIR', 'catalogs',
                                  'Extended_archive_v15')

        hdulist = pyfits.open(fitsfile)
        table = Table.read(fitsfile)

        strip_columns(table)

        table['Source_Name'] = table['NickName']
        table['beta'] = table['Beta']

        m = table['Extended'] == True

        table['Spatial_Filename'] = Column(dtype='S20',length=len(table))
        
        spatial_filenames = []
        for i, row in enumerate(table[m]):
            spatial_filenames += [table[m][i]['Source_Name'].replace(' ','') + '.fits']
        table['Spatial_Filename'][m] =  np.array(spatial_filenames)
        
        super(Catalog4FGLP, self).__init__(table, extdir)
        
        m = self.table['SpectrumType'] == 'PLExpCutoff'
        self.table['SpectrumType'][m] = 'PLSuperExpCutoff'

        table['TS'] = table['Test_Statistic']
        table['Cutoff'] = table['Cutoff_Energy']
