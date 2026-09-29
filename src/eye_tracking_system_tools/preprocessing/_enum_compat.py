"""
Compatibility shim for StrEnum on Python 3.10.
StrEnum was introduced in Python 3.11, so we provide a backport for 3.10.

This module patches enum.StrEnum globally when imported, which is needed
for open_ephys.analysis library compatibility.
"""
import sys
from enum import Enum

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    # Backport StrEnum for Python 3.10
    class StrEnum(str, Enum):
        """
        Enum where members are also (and must be) strings.
        This is a backport for Python 3.10 compatibility.
        """
        def __new__(cls, *values):
            if len(values) > 3:
                raise TypeError('too many arguments for str(): %r' % (values,))
            if len(values) == 1:
                # it must be a string
                if not isinstance(values[0], str):
                    raise TypeError('%r is not a string' % (values[0],))
            if len(values) >= 2:
                # check that encoding argument is a string
                if not isinstance(values[1], str):
                    raise TypeError('encoding must be a string, not %r' % (values[1],))
            if len(values) == 3:
                # check that errors argument is a string
                if not isinstance(values[2], str):
                    raise TypeError('errors must be a string, not %r' % (values[2],))
            value = str(*values)
            member = str.__new__(cls, value)
            member._value_ = value
            return member

        def _generate_next_value_(name, start, count, last_values):
            return name

        def __str__(self):
            return self.value

        def __repr__(self):
            return '%s.%s' % (self.__class__.__name__, self.name)
    
    # CRITICAL: Patch the enum module globally so open_ephys.analysis can import it
    import enum
    enum.StrEnum = StrEnum