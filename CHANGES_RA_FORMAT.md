# RA Format Support - Implementation Summary

## Overview
This update adds functionality to display Right Ascension (RA) coordinates in degrees instead of the default hour angle format in Fermipy map plots.

## Changes Made

### 1. Configuration Options (fermipy/defaults.py)
- Added `ra_format` option to the `plotting` configuration dictionary
  - Default value: `'hour'` (maintains backward compatibility)
  - Allowed values: `'hour'` or `'deg'`
  - Description: "Format for RA axis labels in sky maps. Options are 'hour' (hh:mm:ss) or 'deg' (degrees)."

- Added `ra_format` option to the `psmap` configuration dictionary
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
Updated the following methods to support `ra_format` configuration:
- `make_residmap_plots()` - for residual maps
- `make_tsmap_plots()` - for TS maps
- `make_psmap_plots()` - for PS maps

Each method now calls `kwargs.setdefault('ra_format', self.config['ra_format'])` to ensure the global configuration is applied unless overridden.

### 3. PSMap Module Updates (fermipy/psmap.py)
- Modified `PSMapGenerator.psmap()` method to pass `ra_format` from psmap configuration to the plotter
- Allows users to set RA format specifically for psmap calls

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

# For residmap
resid = gta.residmap(make_plots=True, ra_format='deg')
```

### YAML Configuration
```yaml
plotting:
  ra_format: 'deg'

psmap:
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
1. `fermipy/defaults.py` - Added configuration options
2. `fermipy/plotting.py` - Implemented RA format control in plotting classes
3. `fermipy/psmap.py` - Pass configuration from psmap to plotter

## Files Created
1. `docs/ra_format_usage.md` - User documentation
2. `examples/ra_format_example.py` - Example code
3. `CHANGES_RA_FORMAT.md` - This summary document

## Future Enhancements
Potential future improvements could include:
- Support for additional coordinate formats (e.g., 'd:mm:ss.s')
- Configuration for decimal precision
- Per-axis format control (separate RA and DEC formats)
