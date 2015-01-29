from fermipy.AnalysisMain import *
from fermipy.roi_manager import *
import yaml
import pprint

config = yaml.load(open('sample_config.yaml'))


gta = GTAnalysis(config)

pprint.pprint(gta.config)

gta.setup()

gta.write_results('input_model')
gta.write_results('input_model.yaml')

gta.free_source('mkn421')
gta.free_source('galdiff')
gta.free_source('isodiff')

gta.generate_model()

#import pdb; pdb.set_trace()

gta.fit()

gta.write_xml('fit0')
gta.write_xml('fit0.xml')

# Write results yaml file
gta.write_results('fit_model')

# Write results XML file



