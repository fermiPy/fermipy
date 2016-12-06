# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

from fermipy.diffuse.spectral import SpectralLibrary


def test_spectral():
    the_yaml = """
Constant_Correction :
  SpectrumType : ConstantValue
  spectral_pars : 
    Value :
      name : Value
      scale : 1.0
      value : 1.0 
      min : 1e-4
      max: 1e4
      free : False       
Powerlaw_Correction :
  SpectrumType : PowerLaw
  spectral_pars : 
    Prefactor : 
      name : Prefactor
      scale : 1.0
      value : 1.0 
      min : 0.1
      max: 10.0
      free : False
    Index : 
      name:  Index
      scale : -1.0
      value : 0.0
      min : -1.0
      max : 1.0
      free : False
    Scale : 
      name: Scale
      scale : 1.0
      value : 1000.0
      min : 1000.0
      max : 1000.0
      free : False
LogParabola_Correction :
  SpectrumType : LogParabola
  spectral_pars : 
    norm : 
      name : norm
      scale : 1.0
      value : 1.0 
      min : 0.1
      max: 10.0
      free : False
    alpha : 
      name:  alpha
      scale : 1.0
      value : 0.0
      min : -2.0
      max : 2.0
      free : False
    beta : 
      name:  beta
      scale : 1.0
      value : 0.0
      min : -10.0
      max : 10.0
      free : False
    Eb : 
      name: Eb
      scale : 1.0
      value : 1000.0
      min : 10.0
      max : 100000.0
      free : False
"""
    spectra = SpectralLibrary.create_from_yamlstr(the_yaml)

    # spot check some values
    assert(spectra['Constant_Correction']['SpectrumType'] == 'ConstantValue')
    assert(len(spectra['Constant_Correction']['spectral_pars']) == 1)
    assert(spectra['Constant_Correction'][
           'spectral_pars']['Value']['value'] == 1.0)
    assert(spectra['Constant_Correction'][
           'spectral_pars']['Value']['scale'] == 1.0)
    #assert(spectra['Constant_Correction']['spectral_pars']['Value']['min'] == 1e-4)
    #assert(spectra['Constant_Correction']['spectral_pars']['Value']['max'] == 1e4)
    assert(spectra['Constant_Correction'][
           'spectral_pars']['Value']['free'] is False)

    assert(spectra['LogParabola_Correction']['SpectrumType'] == 'LogParabola')
    assert(len(spectra['LogParabola_Correction']['spectral_pars']) == 4)
    assert(spectra['LogParabola_Correction'][
           'spectral_pars']['norm']['value'] == 1.0)
    assert(spectra['LogParabola_Correction'][
           'spectral_pars']['norm']['scale'] == 1.0)
    assert(spectra['LogParabola_Correction'][
           'spectral_pars']['norm']['min'] == 0.1)
    assert(spectra['LogParabola_Correction'][
           'spectral_pars']['norm']['max'] == 10.0)
    assert(spectra['LogParabola_Correction'][
           'spectral_pars']['norm']['free'] is False)
