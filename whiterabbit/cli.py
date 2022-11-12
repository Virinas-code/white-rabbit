#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Command Line Interface.
"""
import click

from . import train_cleanup
from .uci import UCI


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """Main context."""
    if not ctx.invoked_subcommand:
        print("Connected! #7089")
        uci: UCI = UCI()
        uci.mainloop()


@main.group()
def train():
    """
    Training CLI group.

    Manage training.
    """


@train.command()
def start():
    """
    Start training.

    Launch a training session.
    """


@train.command()
def cleanup():
    """
    Cleanup files.

    Removes config file.
    """
    train_cleanup()
