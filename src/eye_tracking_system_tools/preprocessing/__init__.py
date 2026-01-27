"""
Eye-tracking preprocessing module.

This module provides tools for synchronizing eye-tracking data with arena videos
and Open Ephys recordings.
"""
# IMPORTANT: Import enum compatibility patch FIRST, before any other imports
# This patches enum.StrEnum globally for Python 3.10 compatibility
from . import _enum_compat
from .BlockSync_class import BlockSync
from .OERecording import OERecording

__all__ = ['BlockSync', 'OERecording']
