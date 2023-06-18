import imp
import logging

from press import exceptions

log = logging.getLogger(__name__)

target_extensions = []


def get_module(module_name, path):
    log.debug('Looking up module: %s @ %s' % (module_name, path))
    mod_info = imp.find_module(module_name, [path])
    return imp.load_module(module_name, *mod_info)


def register_extension(cls, run_method='run'):
    if not hasattr(cls, '__extends__'):
        raise exceptions.PressCriticalException(
            '__extends__ attribute is missing')
    if not hasattr(cls, run_method):
        raise exceptions.PressCriticalException(
            'Run method is missing from class')
    log.info('Extending %s target using %s' % (cls.__extends__, cls.__name__))
    target_extensions.append(cls)


def apply_extension(extension_cls, target_object):
    for mro in target_object.__class__.__mro__:
        if vars(mro).get('name') == extension_cls.__extends__:
            return True
    return False


def register_post_handlers():
    __import__('press.targets', globals(), locals(), [], -1)
