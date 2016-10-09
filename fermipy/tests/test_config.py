# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import copy
import numpy as np
import pytest
from fermipy import utils
from fermipy import config
from fermipy import defaults


secA_default_config = {
    'optFloatA' : (None,'',float),
    'optFloatB' : (3.0,'',float),
    'optDict' : (None,'',dict)}

secB_default_config = {
    'optFloatA' : (7.0,'',float),
    'optFloatB' : (None,'',float),
    'optDict' : (None,'',dict)}

default_config = {
    'optFloatA' : (None,'',float),
    'optFloatB' : (3.0,'',float),
    'optFloatC' : (4.0,'',float),
    'optDictA' : (None,'',dict),
    'optDictB' : ({'x' : 3.0},'',dict),
    'optListA' : (None,'',list),
    'secA' : secA_default_config,
    'secB' : secB_default_config,
    }

class TestClass(config.Configurable):
    defaults = default_config    
    def __init__(self,config=None,**kwargs):
        super(TestClass, self).__init__(config, **kwargs)

    def method(self,**kwargs):

        schema = config.ConfigSchema(self.defaults['secA'],
                                     secB=self.defaults['secB'])
        schema.add_option('extraOptA',3.0)
        schema.add_option('extraOptB',False)
        cfg = utils.create_dict(self.config['secA'],secB=self.config['secB'])
        return schema.create_config(cfg, **kwargs)
        
def test_class_config():

    cfg = {'optFloatA' : 2.0, 'optFloatB' : None,
           'optDictA' : {},
           'optDictB' : {'y' : 2.0},
           'optListA' : ['a','b','c'],
           'secA' : {'optFloatA' : 2.0 } }

    # Default configuration
    cls = TestClass()
    assert(cls.config['optFloatA'] == None)
    assert(cls.config['optFloatB'] == 3.0)
    assert(cls.config['optFloatC'] == 4.0)
    assert(cls.config['secA']['optFloatA'] == None)
    assert(cls.config['secA']['optFloatB'] == 3.0)

    # Configuration dictionary
    cls = TestClass(cfg)
    assert(cls.config['optFloatA'] == 2.0)
    assert(cls.config['optFloatB'] == None)
    assert(cls.config['optFloatC'] == 4.0)
    assert(cls.config['optDictA'] == {} )
    assert(cls.config['optDictB'] == {'x' : 3.0, 'y' : 2.0} )
    assert(cls.config['optListA'] == ['a','b','c'] )
    assert(cls.config['secA']['optFloatA'] == 2.0)
    assert(cls.config['secA']['optFloatB'] == 3.0)

    # Configuration dictionary + kwargs
    cls = TestClass(cfg, **{'optFloatA' : 5.0,
                            'optListA' : ['x','y','z'],
                            'secA' : {'optFloatA' : 4.0} })
    assert(cls.config['optFloatA'] == 5.0)
    assert(cls.config['optListA'] == ['x','y','z'])
    assert(cls.config['secA']['optFloatA'] == 4.0)


def test_method_config():

    cfg = {'secA' : {'optFloatA' : 1.0}}
    cls = TestClass(cfg)
    
    outcfg = cls.method()
    assert(outcfg['optFloatA'] == 1.0)
    assert(outcfg['secB']['optFloatA'] == 7.0)
    assert(outcfg['extraOptA'] == 3.0)
    assert(outcfg['extraOptB'] == False)
    
    cfg = {'optFloatA' : 5.0, 'extraOptB' : True}
    outcfg = cls.method(**cfg)
    assert(outcfg['optFloatA'] == 5.0)
    assert(outcfg['extraOptA'] == 3.0)
    assert(outcfg['extraOptB'] == True)
    
    outcfg = cls.method(extraOptA=4.0,
                             secB={'optFloatA' : 2.0},**cfg)
    assert(outcfg['extraOptA'] == 4.0)
    assert(outcfg['secB']['optFloatA'] == 2.0)

    
def test_config_class_validation():

    cfg = {'optFloatA' : 2.0, 'optFloatB' : None, 'optInvalid' : 1.0,
           'optDictA' : {},
           'optDictB' : {'y' : 2.0},
           'secA' : {'optFloatA' : 2.0 } }

    cls = TestClass(cfg,validate=False,secA={'optInvalid' : 3.0})
    assert('optInvalid' not in cls.config)
    assert('optInvalid' not in cls.config['secA'])
        
    with pytest.raises(KeyError):
        cls = TestClass(cfg,validate=True)

    cfg.pop('optInvalid')
    cfg['optFloatA'] = {}
    with pytest.raises(TypeError):
        cls = TestClass(cfg,validate=True)
    

def test_config_method_validation():

    cfg = {'optFloatA' : 2.0, 'optFloatB' : None, 'optFloatD' : 1.0,
           'optDictA' : {},
           'optDictB' : {'y' : 2.0},
           'secA' : {'optFloatA' : 2.0 } }

    cls = TestClass()    
    with pytest.raises(KeyError):    
        cls.method(optInvalid=3.0)

    with pytest.raises(KeyError):    
        cls.method(secB={'optInvalid' : 3.0})
        
    with pytest.raises(TypeError):    
        cls.method(**{'optFloatA' : {}})
    
    with pytest.raises(TypeError):    
        cls.method(**{'secB' : {'optFloatA' : {}}})
