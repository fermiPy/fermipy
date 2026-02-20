from __future__ import absolute_import, division, print_function
import os
import subprocess

__version__ = "unknown"

try:
    from .version import get_git_version
    __version__ = get_git_version()
except Exception as message:
    print(message)

__author__ = "Matthew Wood"

try:
    import pyLikelihood
except ImportError:
    pass


def get_st_version():
    """Get the version string of the ST release."""

    try:
        import st_version.__version__ as science_tools_version
        return science_tools_version
    except ImportError:
        pass
    
    try:
        import ST_Version
        if hasattr(ST_Version, 'version'):
            vv = ST_Version.version()
        elif hasattr(ST_Version, 'get_git_version'):
            vv = ST_Version.get_git_version()
        if vv == "unknown":
            vv = get_ft_conda_version()
        return vv
    except ImportError:
        return ''
    except AttributeError:
        return ''


def get_git_version_fp():
    """Get the version string of the ST release."""

    try:
        import ST_Version
        return ST_Version.get_git_version()
    except ImportError:
        return ''
    except AttributeError:
        return ''


def get_ft_conda_version():
    """Get the fermitools version from package metadata, conda-meta, or conda list."""
    # 1. Try Python package metadata (works in any env, no subprocess; works with Miniforge3)
    try:
        from importlib.metadata import version
        return version('fermitools')
    except Exception:
        pass

    # 2. Try reading conda-meta in current env (no subprocess)
    # Only match package name "fermitools", not "fermitools-data"
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        import glob
        import json
        meta_dir = os.path.join(conda_prefix, 'conda-meta')
        if os.path.isdir(meta_dir):
            for path in glob.glob(os.path.join(meta_dir, 'fermitools-*.json')):
                try:
                    with open(path) as f:
                        info = json.load(f)
                    if info.get('name') != 'fermitools':
                        continue
                    return info.get('version', 'unknown')
                except Exception:
                    pass
            # Fallback: parse version from filename (e.g. fermitools-2.2.0-py39h....json)
            # Exclude fermitools-data (filename would be fermitools-data-<version>-...)
            for path in glob.glob(os.path.join(meta_dir, 'fermitools-*.json')):
                base = os.path.basename(path)
                if base.startswith('fermitools-data-'):
                    continue
                # format: fermitools-<version>-<build>.json
                parts = base.replace('.json', '').split('-')
                if len(parts) >= 2:
                    return parts[1]
                break

    # 3. Try conda list, using CONDA_EXE when set (avoids PATH/permission issues on Miniforge3)
    conda_exe = os.environ.get('CONDA_EXE', 'conda')
    try:
        lines = subprocess.check_output(
            [conda_exe, 'list', '-f', 'fermitools'],
            stderr=subprocess.DEVNULL
        )
        if isinstance(lines, bytes):
            lines = lines.decode('utf-8', errors='replace')
        lines = lines.split('\n')
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    for l in lines:
        if not l or l[0] == '#':
            continue
        tokens = l.split()
        if len(tokens) >= 2:
            return tokens[1]
    return "unknown"
    


PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))
PACKAGE_DATA = os.path.join(PACKAGE_ROOT, 'data')
os.environ['FERMIPY_ROOT'] = PACKAGE_ROOT
os.environ['FERMIPY_DATA_DIR'] = PACKAGE_DATA
if 'FERMI_DIR' in os.environ and 'FERMI_DIFFUSE_DIR' not in os.environ:
    os.environ['FERMI_DIFFUSE_DIR'] = os.path.expandvars('$FERMI_DIR/refdata/fermi/galdiffuse')


def _get_test_runner():
    import os
    from astropy.tests.helper import TestRunner
    return TestRunner(os.path.dirname(__file__))


def test(package=None, test_path=None, args=None, plugins=None,
         verbose=False, pastebin=None, remote_data=False, pep8=False,
         pdb=False, coverage=False, open_files=False, **kwargs):
    """Run the tests using `py.test <http://pytest.org/latest>`_. A
    proper set of arguments is constructed and passed to `pytest.main
    <http://pytest.org/latest/builtin.html#pytest.main>`_.

    Parameters
    ----------
    package : str, optional
        The name of a specific package to test, e.g. 'io.fits' or 'utils'.
        If nothing is specified all default tests are run.
    test_path : str, optional
        Specify location to test by path. May be a single file or
        directory. Must be specified absolutely or relative to the
        calling directory.
    args : str, optional
        Additional arguments to be passed to pytest.main in the ``args``
        keyword argument.
    plugins : list, optional
        Plugins to be passed to pytest.main in the ``plugins`` keyword
        argument.
    verbose : bool, optional
        Convenience option to turn on verbose output from `py.test
        <http://pytest.org/latest>`_. Passing True is the same as
        specifying ``'-v'`` in ``args``.
    pastebin : {'failed','all',None}, optional
        Convenience option for turning on py.test pastebin output. Set to
        ``'failed'`` to upload info for failed tests, or ``'all'`` to upload
        info for all tests.
    remote_data : bool, optional
        Controls whether to run tests marked with @remote_data. These
        tests use online data and are not run by default. Set to True to
        run these tests.
    pep8 : bool, optional
        Turn on PEP8 checking via the `pytest-pep8 plugin
        <http://pypi.python.org/pypi/pytest-pep8>`_ and disable normal
        tests. Same as specifying ``'--pep8 -k pep8'`` in ``args``.
    pdb : bool, optional
        Turn on PDB post-mortem analysis for failing tests. Same as
        specifying ``'--pdb'`` in ``args``.
    coverage : bool, optional
        Generate a test coverage report.  The result will be placed in
        the directory htmlcov.
    open_files : bool, optional
        Fail when any tests leave files open.  Off by default, because
        this adds extra run time to the test suite.  Works only on
        platforms with a working ``lsof`` command.
    parallel : int, optional
        When provided, run the tests in parallel on the specified
        number of CPUs.  If parallel is negative, it will use the all
        the cores on the machine.  Requires the
        `pytest-xdist <https://pypi.python.org/pypi/pytest-xdist>`_ plugin
        installed. Only available when using Astropy 0.3 or later.
    kwargs
        Any additional keywords passed into this function will be passed
        on to the astropy test runner.  This allows use of test-related
        functionality implemented in later versions of astropy without
        explicitly updating the package template.

    """
    test_runner = _get_test_runner()
    return test_runner.run_tests(
        package=package, test_path=test_path, args=args,
        plugins=plugins, verbose=verbose, pastebin=pastebin,
        remote_data=remote_data, pep8=pep8, pdb=pdb,
        coverage=coverage, open_files=open_files, **kwargs)
