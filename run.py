from fermipy.gtanalysis import GTAnalysis
import argparse

usage = "usage: %(prog)s [config file]"
description = "Run fermipy analysis chain."
parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('--config', default = 'sample_config.yaml')

args = parser.parse_args()

gta = GTAnalysis(args.config)

gta.setup()

# Iteratively optimize all components in the ROI
gta.optimize()

# Fix sources w/ TS < 10
gta.free_sources(minmax_ts=[None,10],free=False)

# Free sources within 3 degrees of ROI center
gta.free_sources(distance=3.0)

# Free sources by name
gta.free_source('mkn421')
gta.free_source('galdiff')
gta.free_source('isodiff')

# Free only the normalization of a specific source
gta.free_norm('3FGL J1129.0+3705')

gta.fit()

# Compute the SED for a source
gta.sed('mkn421')

# Write the current state of the ROI model -- this will generate XML
# model files for each component as well as an output analysis
# dictionary in numpy and yaml formats
gta.write_roi('fit1')






