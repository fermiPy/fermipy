from __future__ import absolute_import, division, print_function

from astropy.table import Table

from fermipy import catalog


class _CtorRecorder(object):

    def __init__(self, label):
        self.label = label

    def __call__(self, *args, **kwargs):
        return {'label': self.label, 'args': args, 'kwargs': kwargs}


class _FakeHDU(object):

    def __init__(self, header):
        self.header = header


class _FakeHDULIST(object):

    def __init__(self, header):
        self._hdu = _FakeHDU(header)

    def __getitem__(self, idx):
        if idx == 1:
            return self._hdu
        raise IndexError(idx)


def test_catalog_create_named_registry(monkeypatch):
    monkeypatch.setattr(catalog, 'Catalog3FGL', _CtorRecorder('3FGL'))
    monkeypatch.setattr(catalog, 'Catalog2FHL', _CtorRecorder('2FHL'))
    monkeypatch.setattr(catalog, 'CatalogFL8Y', _CtorRecorder('FL8Y'))
    monkeypatch.setattr(catalog, 'Catalog4FGL', _CtorRecorder('4FGL'))
    monkeypatch.setattr(catalog, 'Catalog4FGLDR2', _CtorRecorder('4FGL-DR2'))
    monkeypatch.setattr(catalog, 'Catalog4FGLDR3', _CtorRecorder('4FGL-DR3'))
    monkeypatch.setattr(catalog, 'Catalog4FGLDR4', _CtorRecorder('4FGL-DR4'))
    monkeypatch.setattr(catalog, 'CatalogFL16Y', _CtorRecorder('FL16Y'))

    for name in ['3FGL', '2FHL', 'FL8Y', '4FGL', '4FGL-DR2',
                 '4FGL-DR3', '4FGL-DR4', 'FL16Y']:
        out = catalog.Catalog.create(name)
        assert out['label'] == name
        assert out['args'] == ()


def test_catalog_create_fits_4fgl_detection(tmp_path, monkeypatch):
    fitsfile = tmp_path / 'dummy.fit'
    fitsfile.write_text('')

    monkeypatch.setattr(catalog, 'Catalog4FGL', _CtorRecorder('4FGL'))
    monkeypatch.setattr(catalog, 'Catalog4FGLDR3', _CtorRecorder('4FGL-DR3'))
    monkeypatch.setattr(catalog.fits, 'open',
                        lambda _: _FakeHDULIST({'CDS-NAME': '4FGL'}))

    monkeypatch.setattr(catalog.Table, 'read',
                        lambda *_args, **_kwargs: Table({'PLEC_Index': [1.0]}))
    out = catalog.Catalog.create(str(fitsfile))
    assert out['label'] == '4FGL'
    assert out['args'] == (str(fitsfile),)

    monkeypatch.setattr(catalog.Table, 'read',
                        lambda *_args, **_kwargs: Table({'PLEC_IndexS': [1.0]}))
    out = catalog.Catalog.create(str(fitsfile))
    assert out['label'] == '4FGL-DR3'
    assert out['args'] == (str(fitsfile),)


def test_catalog_create_fits_fallback_to_4fglp(tmp_path, monkeypatch):
    fitsfile = tmp_path / 'dummy.fit'
    fitsfile.write_text('')

    monkeypatch.setattr(catalog, 'Catalog4FGLP', _CtorRecorder('4FGLP'))
    monkeypatch.setattr(catalog, 'CatalogFPY', _CtorRecorder('FPY'))
    monkeypatch.setattr(catalog.fits, 'open', lambda _: _FakeHDULIST({}))

    monkeypatch.setattr(catalog.Table, 'read',
                        lambda *_args, **_kwargs: Table({'NickName': ['src']}))
    out = catalog.Catalog.create(str(fitsfile))
    assert out['label'] == '4FGLP'
    assert out['args'] == (str(fitsfile),)

    monkeypatch.setattr(catalog.Table, 'read',
                        lambda *_args, **_kwargs: Table({'Source_Name': ['src']}))
    out = catalog.Catalog.create(str(fitsfile))
    assert out['label'] == 'FPY'
    assert out['args'] == (str(fitsfile),)
