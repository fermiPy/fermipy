from fermipy.gtanalysis import *
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

# Fix sources w/ significance < 10
gta.free_sources(cuts=('Detection_Significance',0,10),free=False)

# Free sources within 3 degrees of ROI center
gta.free_sources(distance=3.0)

# Free sources by name
gta.free_source('mkn421')
gta.free_source('galdiff')
gta.free_source('isodiff')
gta.free_norm('3FGL J1129.0+3705')

gta.fit()

# Compute the SED for a source
gta.sed('mkn421')

# Write the post-fit XML model
gta.write_xml('fit1.xml')

# Write all information about the ROI as a yaml file 
gta.write_roi('fit1')






