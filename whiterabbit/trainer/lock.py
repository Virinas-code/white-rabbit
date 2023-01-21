# -*- coding: utf-8 -*-
"""
White Rabbit Chess Engine.

Trainer lock file management.
"""
from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from . import Trainer


def func_acquire_lock(self: Trainer) -> None:
    """
    Acquire lock.

    Lock is stored in data/training/training.lock.
    """
    self.cli.lock_start()
    self.lock.acquire()
    self.cli.lock_end()
