# RA Format Support - Implementation Summary

## Overview
This update adds functionality to display Right Ascension (RA) coordinates in degrees instead of the default hour angle format in Fermipy map plots.

## Changes Made

### 1. Configuration Options (fermipy/defaults.py)
- Added `ra_format` option to the `plotting` configuration dictionary
  - Default value: `'hour'` (maintains backward compatibility)
  - Allowed values: `'hour'` or `'deg'`
  - Description: "Format for RA axis labels in sky maps. Options are 'hour' (hh:mm:ss) or 'deg' (degrees)."

- Added `ra_format` option to ALL map configuration dictionaries:
  - `psmap`: PS map configuration
  - `tsmap`: TS map configuration
  - `tscube`: TS cube configuration
  - `residmap`: Residual map configuration
  - `localize`: Localization configuration
  - `extension`: Extension analysis configuration
  - Allows method-specific override of the global plotting configuration

### 2. Plotting Module Updates (fermipy/plotting.py)

#### ImagePlotter.plot()
- Added `ra_format` parameter to control RA coordinate formatting
- When `ra_format='deg'` is set for ICRS coordinate frames, applies `ax.coords[0].set_major_formatter('d.ddd')` to display RA in decimal degrees
- Default behavior (hour format) is preserved when `ra_format='hour'` or not specified

#### ROIPlotter class
- Added `ra_format` to the class defaults
- Updated `plot()` method to extract and pass `ra_format` parameter from configuration
- Ensures the setting is propagated to ImagePlotter

#### AnalysisPlotter class
Updated ALL plotting methods to support `ra_format` configuration:
- `make_residmap_plots()` - for residual maps
- `make_tsmap_plots()` - for TS maps and TS cubes
- `make_psmap_plots()` - for PS maps
- `make_roi_plots()` - for ROI diagnostic maps (model, counts, projections)
- `make_localization_plots()` - for source localization maps (both plots)
- `make_extension_plots()` -> `_plot_extension_tsmap()` - for extension analysis maps

Each method now calls `kwargs.setdefault('ra_format', self.config['ra_format'])` to ensure the global configuration is applied unless overridden.

### 3. Analysis Method Updates
Updated ALL map-generating analysis methods to pass `ra_format` to plotting:

#### fermipy/psmap.py
- Modified `PSMapGenerator.psmap()` method to pass `ra_format` from psmap configuration to the plotter

#### fermipy/tsmap.py
- Modified `tsmap()` method to pass `ra_format` to make_tsmap_plots
- Modified `tscube()` method to pass `ra_format` to make_tsmap_plots

#### fermipy/residmap.py
- Modified `residmap()` method to pass `ra_format` to make_residmap_plots

#### fermipy/sourcefind.py
- Modified `localize()` method to pass `ra_format` to make_localization_plots

#### fermipy/extension.py
- Modified `extension()` method to pass `ra_format` to make_extension_plots

#### fermipy/gtanalysis.py
- `write_roi()` and `make_plots()` methods automatically support `ra_format` through the plotting infrastructure

## Usage Examples

### Global Configuration
```python
# Set globally for all plots
gta.config['plotting']['ra_format'] = 'deg'
```

### Method-Specific Configuration
```python
# For psmap
psmap = gta.psmap(cmap='ccube_00.fits',
                  mmap='mcube_model01_00.fits',
                  make_plots=True,
                  ra_format='deg')

# For tsmap
tsmap = gta.tsmap(make_plots=True, ra_format='deg')

# For tscube
tscube = gta.tscube(make_plots=True, ra_format='deg')

# For residmap
resid = gta.residmap(make_plots=True, ra_format='deg')

# For localization
loc = gta.localize('SourceName', make_plots=True, ra_format='deg')

# For extension
ext = gta.extension('SourceName', make_plots=True, ra_format='deg')

# For ROI plots
gta.write_roi('output', make_plots=True, plotting={'ra_format': 'deg'})
```

### YAML Configuration
```yaml
plotting:
  ra_format: 'deg'  # Global setting for all plots

# Method-specific settings (optional overrides)
psmap:
  make_plots: true
  ra_format: 'deg'

tsmap:
  make_plots: true
  ra_format: 'deg'

tscube:
  make_plots: true
  ra_format: 'deg'

residmap:
  make_plots: true
  ra_format: 'deg'

localize:
  make_plots: true
  ra_format: 'deg'

extension:
  make_plots: true
  ra_format: 'deg'
```

## Backward Compatibility
- Default value is `'hour'`, preserving existing behavior
- All existing code will continue to work without modification
- Users can opt-in to degree format by setting the configuration option

## Documentation
- Created `docs/ra_format_usage.md` with comprehensive usage guide
- Created `examples/ra_format_example.py` with working examples

## Testing
- Syntax validation passed
- No linter errors introduced
- Compatible with existing fermipy infrastructure

## Addresses Issues
- Issue #622: Request for RA coordinates in degrees
- PR request from @ndilalla to add RA degree support

## Technical Details

### WCSAxes Formatter
The implementation uses matplotlib's WCSAxes coordinate formatter:
- `'d.ddd'` format displays coordinates in decimal degrees
- Default format for ICRS RA is hour angle (automatic in WCSAxes)
- Only affects ICRS coordinate systems; galactic coordinates are unaffected

### Configuration Precedence
1. Method-specific parameter (highest priority)
2. Method-specific configuration (e.g., `psmap.ra_format`)
3. Global plotting configuration (`plotting.ra_format`)
4. Default value (`'hour'`)

## Files Modified
1. `fermipy/defaults.py` - Added configuration options to all map configurations
2. `fermipy/plotting.py` - Implemented RA format control in all plotting methods
3. `fermipy/psmap.py` - Pass configuration from psmap to plotter
4. `fermipy/tsmap.py` - Pass configuration from tsmap and tscube to plotter
5. `fermipy/residmap.py` - Pass configuration from residmap to plotter
6. `fermipy/sourcefind.py` - Pass configuration from localize to plotter
7. `fermipy/extension.py` - Pass configuration from extension to plotter

## Files Created
1. `docs/ra_format_usage.md` - User documentation
2. `examples/ra_format_example.py` - Example code
3. `CHANGES_RA_FORMAT.md` - This summary document

## Future Enhancements
Potential future improvements could include:
- Support for additional coordinate formats (e.g., 'd:mm:ss.s')
- Configuration for decimal precision
- Per-axis format control (separate RA and DEC formats)
