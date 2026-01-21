# RA Format Configuration in Fermipy

This document describes how to configure the Right Ascension (RA) coordinate format in Fermipy plots.

## Overview

By default, Fermipy displays RA coordinates in hour angle format (hh:mm:ss). You can now configure plots to display RA in degrees instead.

## Configuration Options

### Global Configuration

Set the RA format globally in your plotting configuration:

```python
# In your configuration YAML file
plotting:
  ra_format: 'deg'  # Options: 'hour' (default) or 'deg'
```

Or in Python:

```python
gta = GTAnalysis('config.yaml')
gta.config['plotting']['ra_format'] = 'deg'
```

### Method-Specific Configuration

#### For psmap

You can set the RA format specifically for psmap operations:

```python
# Method 1: Using psmap configuration
gta.config['psmap']['ra_format'] = 'deg'

# Then call psmap normally
gta.write_model_map(model_name="model01")
psmap = gta.psmap(cmap='ccube_00.fits', 
                  mmap='mcube_model01_00.fits', 
                  make_plots=True)
```

Or pass it directly to the psmap call:

```python
# Method 2: Pass as parameter
psmap = gta.psmap(cmap='ccube_00.fits',
                  mmap='mcube_model01_00.fits',
                  make_plots=True,
                  ra_format='deg')
```

#### For Other Map Methods

The `ra_format` parameter is supported by **ALL** map generation methods in fermipy:

```python
# For residual maps
gta.residmap(make_plots=True, ra_format='deg')

# For TS maps
gta.tsmap(make_plots=True, ra_format='deg')

# For TS cubes
gta.tscube(make_plots=True, ra_format='deg')

# For localization plots
gta.localize('SourceName', make_plots=True, ra_format='deg')

# For extension analysis
gta.extension('SourceName', make_plots=True, ra_format='deg')

# For ROI diagnostic plots
gta.write_roi('roi_file', make_plots=True, plotting={'ra_format': 'deg'})
# or
gta.make_plots('prefix', ra_format='deg')
```

## Examples

### Example 1: PS Map with RA in Degrees

```python
from fermipy.gtanalysis import GTAnalysis

# Create analysis object
gta = GTAnalysis('my_config.yaml')

# Set RA format to degrees
gta.config['plotting']['ra_format'] = 'deg'

# Generate PS map with plots
gta.write_model_map(model_name="model01")
psmap = gta.psmap(cmap='ccube_00.fits',
                  mmap='mcube_model01_00.fits',
                  make_plots=True)
```

### Example 2: Per-Call Configuration

```python
# Use degrees for this specific call
psmap = gta.psmap(cmap='ccube_00.fits',
                  mmap='mcube_model01_00.fits',
                  make_plots=True,
                  ra_format='deg')

# Use hours for another call
tsmap = gta.tsmap(make_plots=True, ra_format='hour')
```

### Example 3: YAML Configuration

```yaml
# config.yaml
plotting:
  ra_format: 'deg'
  format: 'png'
  cmap: 'magma'

psmap:
  make_plots: true
  ra_format: 'deg'  # Can override plotting.ra_format for psmap
```

## Format Options

- `'hour'` (default): Displays RA in hour angle format (hh:mm:ss)
  - Example: 12:30:15
  
- `'deg'`: Displays RA in decimal degrees
  - Example: 187.562

## Supported Map Types

The `ra_format` configuration is supported by all map generation methods:

### Sky Maps
- **psmap**: PS (Poisson statistics) maps
- **tsmap**: Test Statistic maps
- **tscube**: Test Statistic cubes (energy-resolved TS maps)
- **residmap**: Residual maps (data - model)

### Analysis Maps
- **localization**: Source position localization maps
- **extension**: Source extension analysis maps
- **ROI plots**: Region of interest diagnostic maps
  - Model maps
  - Counts maps
  - Projection plots

## Notes

- The default format is `'hour'` to maintain backward compatibility
- The setting only affects ICRS (celestial) coordinate systems; galactic coordinates are unaffected
- **ALL** 2D sky maps in fermipy support this configuration
- The format can be set globally or overridden per method call
- Configuration can be set at multiple levels:
  - Global plotting configuration
  - Method-specific configuration (psmap, tsmap, residmap, etc.)
  - Direct parameter in method call
