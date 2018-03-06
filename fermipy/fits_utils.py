# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import fermipy
from fermipy import utils
from fermipy.hpx_utils import HPX


def write_fits(hdulist, outfile, keywords):

    for h in hdulist:
        for k, v in keywords.items():
            h.header[k] = v
        h.header['CREATOR'] = 'fermipy ' + fermipy.__version__
        h.header['STVER'] = fermipy.get_st_version()

    hdulist.writeto(outfile, clobber=True)


def find_and_read_ebins(hdulist):
    """  Reads and returns the energy bin edges.

    This works for both the CASE where the energies are in the ENERGIES HDU
    and the case where they are in the EBOUND HDU
    """
    from fermipy import utils
    ebins = None
    if 'ENERGIES' in hdulist:
        hdu = hdulist['ENERGIES']
        ectr = hdu.data.field(hdu.columns[0].name)
        ebins = np.exp(utils.center_to_edge(np.log(ectr)))
    elif 'EBOUNDS' in hdulist:
        hdu = hdulist['EBOUNDS']
        emin = hdu.data.field('E_MIN') / 1E3
        emax = hdu.data.field('E_MAX') / 1E3
        ebins = np.append(emin, emax[-1])
    return ebins


def read_energy_bounds(hdu):
    """ Reads and returns the energy bin edges from a FITs HDU
    """
    nebins = len(hdu.data)
    ebin_edges = np.ndarray((nebins + 1))
    try:
        ebin_edges[0:-1] = np.log10(hdu.data.field("E_MIN")) - 3.
        ebin_edges[-1] = np.log10(hdu.data.field("E_MAX")[-1]) - 3.
    except KeyError:
        ebin_edges[0:-1] = np.log10(hdu.data.field("energy_MIN"))
        ebin_edges[-1] = np.log10(hdu.data.field("energy_MAX")[-1])
    return ebin_edges


def read_spectral_data(hdu):
    """ Reads and returns the energy bin edges, fluxes and npreds from
    a FITs HDU
    """
    ebins = read_energy_bounds(hdu)
    fluxes = np.ndarray((len(ebins)))
    try:
        fluxes[0:-1] = hdu.data.field("E_MIN_FL")
        fluxes[-1] = hdu.data.field("E_MAX_FL")[-1]
        npreds = hdu.data.field("NPRED")
    except:
        fluxes = np.ones((len(ebins)))
        npreds = np.ones((len(ebins)))
    return ebins, fluxes, npreds


def make_energies_hdu(energy_vals, extname="ENERGIES"):
    """ Builds and returns a FITs HDU with the energy values

    extname   : The HDU extension name           
    """
    cols = [fits.Column("Energy", "D", unit='MeV', array=energy_vals)]
    hdu = fits.BinTableHDU.from_columns(cols, name=extname)
    return hdu


def write_maps(primary_map, maps, outfile, **kwargs):

    if primary_map is None:
        hdu_images = [fits.PrimaryHDU()]
    else:
        hdu_images = [primary_map.create_primary_hdu()]

    for k, v in sorted(maps.items()):
        hdu_images += [v.create_image_hdu(k, **kwargs)]

    energy_hdu = kwargs.get('energy_hdu', None)
    if energy_hdu:
        hdu_images += [energy_hdu]

    write_hdus(hdu_images, outfile)


def write_hdus(hdus, outfile, **kwargs):

    keywords = kwargs.get('keywords', {})

    hdulist = fits.HDUList(hdus)
    for h in hdulist:

        for k, v in keywords.items():
            h.header[k] = v

        h.header['CREATOR'] = 'fermipy ' + fermipy.__version__
        h.header['STVER'] = fermipy.get_st_version()
    hdulist.writeto(outfile, clobber=True)


def write_fits_image(data, wcs, outfile):
    hdu_image = fits.PrimaryHDU(data, header=wcs.to_header())
    hdulist = fits.HDUList([hdu_image])
    hdulist.writeto(outfile, clobber=True)


def write_hpx_image(data, hpx, outfile, extname="SKYMAP"):
    hpx.write_fits(data, outfile, extname, clobber=True)


def read_projection_from_fits(fitsfile, extname=None):
    """
    Load a WCS or HPX projection.
    """
    f = fits.open(fitsfile)
    nhdu = len(f)
    # Try and get the energy bounds
    try:
        ebins = find_and_read_ebins(f)
    except:
        ebins = None

    if extname is None:
        # If there is an image in the Primary HDU we can return a WCS-based
        # projection
        if f[0].header['NAXIS'] != 0:
            proj = WCS(f[0].header)
            return proj, f, f[0]
    else:
        if f[extname].header['XTENSION'] == 'IMAGE':
            proj = WCS(f[extname].header)
            return proj, f, f[extname]
        elif extname in ['SKYMAP', 'SKYMAP2']:
            proj = HPX.create_from_hdu(f[extname], ebins)
            return proj, f, f[extname]
        elif f[extname].header['XTENSION'] == 'BINTABLE':
            try:
                if f[extname].header['PIXTYPE'] == 'HEALPIX':
                    proj = HPX.create_from_hdu(f[extname], ebins)
                    return proj, f, f[extname]
            except:
                pass
        return None, f, None

    # Loop on HDU and look for either an image or a table with HEALPix data
    for i in range(1, nhdu):
        # if there is an image we can return a WCS-based projection
        if f[i].header['XTENSION'] == 'IMAGE':
            proj = WCS(f[i].header)
            return proj, f, f[i]
        elif f[i].header['XTENSION'] == 'BINTABLE':
            if f[i].name in ['SKYMAP', 'SKYMAP2']:
                proj = HPX.create_from_hdu(f[i], ebins)
                return proj, f, f[i]
            try:
                if f[i].header['PIXTYPE'] == 'HEALPIX':
                    proj = HPX.create_from_hdu(f[i], ebins)
                    return proj, f, f[i]
            except:
                pass

    return None, f, None


def write_tables_to_fits(filepath, tablelist, clobber=False,
                         namelist=None, cardslist=None, hdu_list=None):
    """
    Write some astropy.table.Table objects to a single fits file
    """
    outhdulist = [fits.PrimaryHDU()]
    rmlist = []
    for i, table in enumerate(tablelist):
        ft_name = "%s._%i" % (filepath, i)
        rmlist.append(ft_name)
        try:
            os.unlink(ft_name)
        except:
            pass
        table.write(ft_name, format="fits")
        ft_in = fits.open(ft_name)
        if namelist:
            ft_in[1].name = namelist[i]
        if cardslist:
            for k, v in cardslist[i].items():
                ft_in[1].header[k] = v
        ft_in[1].update()
        outhdulist += [ft_in[1]]

    if hdu_list is not None:
        for h in hdu_list:
            outhdulist.append(h)

    fits.HDUList(outhdulist).writeto(filepath, clobber=clobber)
    for rm in rmlist:
        os.unlink(rm)


def dict_to_table(input_dict):

    from astropy.table import Table, Column

    cols = []

    for k, v in sorted(input_dict.items()):

        if isinstance(v, dict):
            continue
        elif isinstance(v, float):
            cols += [Column(name=k, dtype='f8', data=np.array([v]))]
        elif isinstance(v, bool):
            cols += [Column(name=k, dtype=bool, data=np.array([v]))]
        elif utils.isstr(v):
            cols += [Column(name=k, dtype='S32', data=np.array([v]))]
        elif isinstance(v, np.ndarray):
            cols += [Column(name=k, dtype=v.dtype, data=np.array([v]))]

    return Table(cols)
