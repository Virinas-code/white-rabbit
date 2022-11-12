#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Rabbit chess engine.

Command Line Interface.
"""
import click

from . import train_start, train_cleanup
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
@click.option(
    "-c", "--config", "config", default=False, is_flag=True, type=bool
)
def start(config: bool = False):
    """
    Start training.

    Launch a training session.

    :param bool config: Wether to reconfigure arguments or not.
    """
    train_start(config)


@train.command()
def cleanup():
    """
    Cleanup files.

    Removes config file.
    """
    train_cleanup()
