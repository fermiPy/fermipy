from fermipy.AnalysisMain import *
from fermipy.roi_manager import *
from fermipy.config import *
import yaml
import pprint
from fermipy.Logger import *
        
config = ConfigManager.create('sample_config.yaml')

gta = GTAnalysis(config)

gta.setup()

gta.write_roi('input_model')

gta.free_source(radius=3.0)
gta.free_source('mkn421')
gta.free_source('galdiff')
gta.free_source('isodiff')
gta.free_norm('3FGL J1129.0+3705')

gta.generate_model()

gta.fit()

gta.write_xml('fit0.xml')

# Write results yaml file
gta.write_roi('fit_model')




