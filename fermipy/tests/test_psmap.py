from fermipy.gtanalysis import GTAnalysis
from fermipy import gtpsmap

gta = GTAnalysis('fermipy_test_draco/config.yaml')

gta.setup()

gta.print_roi()

gta.write_roi('fit0')

gta.psmap(model_name='model01')