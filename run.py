from fermipy.AnalysisMain import *
from fermipy.roi_manager import *
from fermipy.config import *
from fermipy.logger import *
import yaml
import pprint
import argparse


usage = "usage: %(prog)s [config file]"
description = "Run fermipy analysis chain."
parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('--config', default = 'sample_config.yaml')

args = parser.parse_args()

config = ConfigManager.create(args.config)

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




