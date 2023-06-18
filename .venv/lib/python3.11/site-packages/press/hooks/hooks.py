"""
Hooks are to be used to execute sections of code as needed throughout the Press process.
"""

import logging
import inspect
from functools import wraps
from collections import namedtuple

from press.exceptions import HookError

log = logging.getLogger(__name__)

Hook = namedtuple("Hook", "name function args kwargs")

valid_hook_points = [
    "post-press-init", "pre-apply-layout", "pre-mount-fs", "pre-image-ops",
    "pre-image-acquire", "post-image-acquire", "pre-image-validate",
    "post-image-validate", "pre-image-extract", "post-image-extract",
    "pre-post-config", "pre-create-staging", "pre-target-run", "pre-extensions",
    "post-extensions"
]
target_hooks = {}

for valid_point in valid_hook_points:
    target_hooks[valid_point] = []


def clear_hooks():
    for point in target_hooks:
        del target_hooks[point][:]


def add_hook(func, point, hook_name=None, *args, **kwargs):
    """Add a function as a hook, make sure the function accepts a keyword argument 'press_config'"""
    if not hook_name:
        hook_name = func.__name__

    if "press_config" not in inspect.getargspec(func).args:
        raise HookError(
            "Attempted hook '{0}' does not accept argument 'press_config'".
            format(hook_name))

    log.debug("Adding hook '{0}' for point '{1}'".format(hook_name, point))
    if point not in valid_hook_points:
        raise HookError("Not a valid hook point '{0}'".format(point))

    for hook in target_hooks[point]:
        if hook.hook_name == hook_name:
            raise HookError("Hook '{0}' already exists in '{0}'!".format(
                hook_name, point))

    target_hooks[point].append(
        Hook(name=hook_name, function=func, args=args, kwargs=kwargs))


def run_hooks(point, press_config):

    log.info("Running hooks for point '{0}'".format(point))
    if point not in valid_hook_points:
        log.warning("Specified hook point '{0}' does not exist".format(point))
        return

    for hook in target_hooks[point]:
        log.debug("Running hook '{0}' for point '{1}'".format(
            hook.function.__name__, point))
        hook.function(*hook.args, press_config=press_config, **hook.kwargs)


def hook_point(point, hook_name=None, *args, **kwargs):
    """
    Wrapper for adding a function as a hook,
    make sure the function accepts a keyword argument 'press_config'
    """

    def decorate(func):
        add_hook(func, point, hook_name, *args, **kwargs)

        @wraps(func)
        def wrapped(*wargs, **wkwargs):
            return func(*wargs, **wkwargs)

        return wrapped

    return decorate
