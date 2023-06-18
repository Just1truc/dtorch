import os
import pkg_resources


def get_package_version(package):
    try:
        dist = pkg_resources.get_distribution(package)
    except pkg_resources.DistributionNotFound:
        return None
    return dist.version


def get_press_version():
    return get_package_version('press')


def get_press_location():
    return os.path.abspath(pkg_resources.get_distribution('press').location)
