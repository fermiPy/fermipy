# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import textwrap


def update_docstring(docstring, options_dict):
    """Update a method docstring by inserting option docstrings defined in
    the options dictionary.  The input docstring should define `{options}`
    at the location where the options docstring block should be inserted.

    Parameters
    ----------
    docstring : str
        Existing method docstring.

    options_dict : dict
        Dictionary defining the options that will be appended to the
        method docstring.  Dictionary keys are mapped to option names
        and each element of the dictionary should have the format
        (default value, docstring, type).

    Returns
    -------
    docstring : str
        Updated method docstring.
    """

    options_str = []
    for i, (k, v) in enumerate(sorted(options_dict.items())):

        option_str = ''
        if i == 0:
            option_str += '%s : %s\n' % (k, v[2].__name__)
        else:
            option_str += ' ' * 8 + '%s : %s\n' % (k, v[2].__name__)

        option_doc = v[1]
        option_doc += ' (default : %s)' % v[0]
        option_doc = textwrap.wrap(option_doc, 72 - 12)
        option_str += ' ' * 12 + ('\n' + ' ' * 12).join(option_doc)
        options_str += [option_str]

    options_str = '\n\n'.join(options_str)

    return docstring.format(options=options_str)


class DocstringMeta(type):
    """Meta class to update docstrings."""

    def __new__(cls, name, parents, attrs):

        if '_docstring_registry' in attrs:
            doc_reg = attrs['_docstring_registry']
        else:
            return super(DocstringMeta, cls).__new__(cls, name, parents, attrs)

        for attr_name in attrs:

            # skip special methods
            if attr_name.startswith("__"):
                continue

            # skip non-functions
            attr = attrs[attr_name]
            if not hasattr(attr, '__call__'):
                continue

            if not attr_name in doc_reg:
                continue

            # update docstring
            attr.__doc__ = update_docstring(attr.__doc__,
                                            doc_reg[attr_name])

        for parent in parents:
            for attr_name in dir(parent):

                # we already have this method
                if attr_name in attrs:
                    continue

                # skip special methods
                if attr_name.startswith("__"):
                    continue

                if not attr_name in doc_reg:
                    continue

                # get the original function and copy it
                a = getattr(parent, attr_name)

                # skip non-functions
                if not hasattr(a, '__call__'):
                    continue

                # copy function
                f = a.__func__
                attr = type(f)(
                    f.func_code, f.func_globals, f.func_name,
                    f.func_defaults, f.func_closure)
                doc = f.__doc__

                # update docstring and add attr
                attr.__doc__ = update_docstring(doc,
                                                doc_reg[attr_name])
                attrs[attr_name] = attr

        # we need to call type.__new__ to complete the initialization
        return super(DocstringMeta, cls).__new__(cls, name, parents, attrs)
