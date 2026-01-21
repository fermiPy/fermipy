#!/usr/bin/env python
"""
Example script demonstrating the RA format configuration in Fermipy.

This example shows how to configure map plots to display RA coordinates
in degrees instead of the default hour angle format.
"""

# Example 1: Setting RA format globally in configuration
example_config = {
    'plotting': {
        'ra_format': 'deg',  # Use degrees instead of hours
        'format': 'png',
        'cmap': 'magma'
    }
}

# Example 2: Using RA format with psmap
def example_psmap_degrees(gta):
    """Generate PS map with RA in degrees"""
    # Method 1: Set in plotting config
    gta.config['plotting']['ra_format'] = 'deg'
    
    gta.write_model_map(model_name="model01")
    psmap = gta.psmap(
        cmap='ccube_00.fits',
        mmap='mcube_model01_00.fits',
        make_plots=True
    )
    return psmap

def example_psmap_degrees_direct(gta):
    """Generate PS map with RA in degrees using direct parameter"""
    # Method 2: Pass ra_format directly
    gta.write_model_map(model_name="model01")
    psmap = gta.psmap(
        cmap='ccube_00.fits',
        mmap='mcube_model01_00.fits',
        make_plots=True,
        ra_format='deg'  # Override default setting
    )
    return psmap

# Example 3: Using RA format with other plotting methods
def example_other_plots_degrees(gta):
    """Use RA in degrees for various plot types"""
    
    # Residual map with RA in degrees
    resid = gta.residmap(make_plots=True, ra_format='deg')
    
    # TS map with RA in degrees
    tsmap = gta.tsmap(make_plots=True, ra_format='deg')
    
    # Localization with RA in degrees
    loc = gta.localize('SourceName', make_plots=True, ra_format='deg')
    
    # Extension analysis with RA in degrees
    ext = gta.extension('SourceName', make_plots=True, ra_format='deg')
    
    return resid, tsmap, loc, ext

# Example 4: YAML configuration
example_yaml_config = """
# config.yaml
data:
  evfile: events.fits
  scfile: spacecraft.fits

binning:
  roiwidth: 10.0
  binsz: 0.1
  binsperdec: 8

plotting:
  ra_format: 'deg'  # Display RA in degrees
  format: 'png'
  cmap: 'magma'
  figsize: [10.0, 8.0]

psmap:
  make_plots: true
  ra_format: 'deg'  # Can override plotting.ra_format specifically for psmap
  emin: 100
  emax: 1000000
  nbinloge: 20
"""

if __name__ == '__main__':
    print(__doc__)
    print("\nConfiguration examples:")
    print("\n1. Python dictionary configuration:")
    print(example_config)
    print("\n2. YAML configuration:")
    print(example_yaml_config)
    print("\nFor full documentation, see docs/ra_format_usage.md")
