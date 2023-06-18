from __future__ import absolute_import

import logging
import sys

from six import print_ as p3print

from size import Size

from press.configuration.util import configuration_from_file
from press.exceptions import PressCriticalException
from press.log import setup_logging
from press.plugin_init import init_plugins
from press.press import PressOrchestrator
from press.press_cli import parse_args
from press.hooks.hooks import clear_hooks

log = logging.getLogger('press')


def safe_convert_sizes(size):
    """
    Used to convert Size() values from the command line
    :param size: string
    :return: int
    """
    try:
        return Size(size).bytes
    except ValueError as ve:
        p3print('Error converting value : {}'.format(ve), file=sys.stderr)
        sys.exit(1)


def apply(namespace):
    try:
        configuration = configuration_from_file(namespace.configuration)
    except PressCriticalException:
        p3print(
            'Could not open configuration at {}'.format(
                namespace.configuration),
            file=sys.stderr)
        sys.exit(1)

    # setup logging
    console_logging = True
    if namespace.debug:
        log_level = logging.DEBUG
    elif namespace.quiet:
        log_level = logging.ERROR
        console_logging = False
    else:
        log_level = logging.INFO

    setup_logging(log_level, console_logging, namespace.log_file,
                  namespace.cli_debug)

    # plugins
    init_plugins(configuration, namespace.plugin_dirs, namespace.plugins)

    try:
        orchestrator = PressOrchestrator(
            configuration,
            parted_path=namespace.parted_path,
            deployment_root=namespace.deployment_root,
            staging_dir=namespace.staging_dir,
            explicit_use_fibre_channel=namespace.use_fibre_channel,
            explicit_loop_only=namespace.loop_only,
            partition_start=namespace.partition_start,
            alignment=namespace.alignment,
            lvm_pe_size=namespace.lvm_pe_size,
            http_proxy=namespace.proxy)
    except Exception as e:
        p3print(
            'Encountered an error while initializing : {}'.format(e),
            file=sys.stderr)
        if namespace.debug:
            raise
        sys.exit(1)

    error = False
    try:
        orchestrator.run()
    except Exception as e:
        p3print('Error applying configuration : {}'.format(e), file=sys.stderr)
        if namespace.debug:
            raise
        error = True
    finally:
        if orchestrator.layout.committed:
            orchestrator.teardown()

        del logging.getLogger('press').handlers[:]

        clear_hooks()

    if error:
        sys.exit(1)

    log.info('Completed.')


def main():
    """ Command line entry point """
    namespace = parse_args(sys.argv[1:])

    if namespace.command == 'apply':
        apply(namespace)
    else:
        print(namespace)
