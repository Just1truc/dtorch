import logging

from press import helpers
from press.exceptions import NetworkConfigurationError

log = logging.getLogger(__name__)

# Type: func
lookup_methods = {
    'mac': helpers.networking.get_device_by_mac,
    'dev': helpers.networking.get_device_by_dev
}


def lookup_interface(interface, dev_missing_ok=False):
    """
    This function is transitory, Network configuration will be re-worked in 0.4
    :param interface:
    :return:
    """
    ref = interface['ref']
    lookup_method = lookup_methods.get(ref['type'])
    if not lookup_method:
        raise NetworkConfigurationError(
            'Press 0.3 Network configuration error, missing type')
    ndi = lookup_method(ref['value'])
    if not ndi:
        if ref['type'] == 'dev' and dev_missing_ok:

            class Duck(object):
                devname = ref['value']

            ndi = Duck
            log.warn(
                'Duck typing NetDeviceInfo object, %s does not seem to exist' %
                ref['value'])
        else:
            raise NetworkConfigurationError(
                'Press 0.3 Could not find device: %s' % ref['value'])
    return interface['name'], ndi
