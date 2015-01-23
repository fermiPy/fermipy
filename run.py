from AnalysisMain import *
import yaml

config = yaml.load(open('sample_config.yaml'))


gta = GTAnalysis(config)

gta.setup()
gta.fit()

# Write results yaml file
gta.write_results()

# Write results XML file
gta.write_xml()


