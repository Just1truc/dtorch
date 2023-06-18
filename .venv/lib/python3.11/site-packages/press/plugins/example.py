"""
Example plugin interface
"""
import logging

log = logging.getLogger('press.plugins.example')


def plugin_init(configuration):
    """
    Press always calls the plugin_init function of discovered plugin modules
    :param configuration: This is the whole press configuration
    :return: None
    """
    log.info('Enabled plugins: {}'.format(configuration.get('plugins', [])))
