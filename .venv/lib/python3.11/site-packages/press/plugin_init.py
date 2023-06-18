# TODO: re-implement with importlib
from __future__ import absolute_import

import imp
import logging
import os

log = logging.getLogger('press.plugins')


def get_plugin(plugin_name, plugin_dirs=None):
    if not plugin_dirs:
        log.warning('There are no plugin.scan_directories defined')
        return

    mod_info = None
    for d in plugin_dirs:
        if not os.path.isdir(d):
            log.warning(
                'Plugin directory {} is missing, or relative path is incorrect.'
                'See press.configuration.global_defaults'.format(d))
            continue

        log.debug('Attempting to discover plugin: %s' % plugin_name)

        try:
            mod_info = imp.find_module(plugin_name,
                                       [os.path.join(d, plugin_name)])
        except ImportError:
            log.debug('Plugin %s module does not exist.' % plugin_name)
            continue

    if not mod_info:
        log.error(
            'plugin "{}" could not be found in any search directory. Skipping'.
            format(plugin_name))
        return

    mod = imp.load_module(plugin_name, *mod_info)

    if not hasattr(mod, 'plugin_init'):
        log.error('Plugin {} found but is missing init'.format(plugin_name))
        return

    log.info('%s plugin module discovered' % plugin_name)
    return mod


def init_plugins(configuration, plugin_dirs, plugins=None):
    if not plugins:
        log.debug('No plugins are enabled')
        return

    for plugin in plugins:
        mod = get_plugin(plugin, plugin_dirs)
        if not mod:
            continue

        log.info('Attempting to initialize %s plugin' % plugin)
        try:
            mod.plugin_init(configuration)
        except Exception as e:
            log.error('Error running %s plugin init: %s' % (plugin, e))

        log.info('%s plugin successfully initialized' % plugin)
