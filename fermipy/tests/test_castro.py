from __future__ import absolute_import, division, print_function, \
    unicode_literals

import pytest
import os

from numpy.testing import assert_allclose
import numpy as np

from fermipy import castro


@pytest.fixture(scope='module')
def dummy(request):
    return None


def test_castro_test_spectra_sed(dummy):    
    c = castro.CastroData.create_from_sedfile("/nfs/slac/kipac/fs1/u/dmcat/workdir/sandbox/sed.fits")
    test_dict = c.test_spectra()

    assert( np.abs(test_dict_sed['PowerLaw']['TS'][0] -  26.88) < 0.01)
    assert( np.abs(test_dict_sed['LogParabola']['TS'][0] -  27.30) < 0.01)
    assert( np.abs(test_dict_sed['PLExpCutoff']['TS'][0] -  28.17) < 0.01)

    assert_allclose(test_dict_sed['PowerLaw']['Result'],np.array([  1.10610705e-16,  -3.86502196e+00]))
    assert_allclose(test_dict_sed['LogParabola']['Result'],np.array([  4.22089847e-17,  -3.71123256e+00,  -5.72287142e-02]))
    assert_allclose(test_dict_sed['PLExpCutoff']['Result'],np.array([  5.53464102e-13,  -2.88974835e+00,   1.19098881e+00]))


def test_castro_test_spectra_castro(dummy):    
    c = castro.CastroData.create_from_fits("/nfs/slac/kipac/fs1/u/dmcat/workdir/sandbox/castro.fits",irow=0)
    test_dict_fits = c.test_spectra()

    assert( np.abs(test_dict_sed['PowerLaw']['TS'][0] -  2.74) < 0.01)
    assert( np.abs(test_dict_sed['LogParabola']['TS'][0] -  3.72) < 0.01)
    assert( np.abs(test_dict_sed['PLExpCutoff']['TS'][0] -  3.70) < 0.01)

    


