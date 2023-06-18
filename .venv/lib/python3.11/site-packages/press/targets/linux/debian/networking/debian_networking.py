import logging

from press.helpers import deployment
from press.targets import util
from press.targets.linux.debian.networking.interfaces_template import (
    INTERFACES_TEMPLATE
)
from press.targets.linux.debian.networking.netplan_template import (
    NETPLAN_TEMPLATE
)

# TODO: Remove jinja2 dependency
from jinja2 import Environment

log = logging.getLogger(__name__)


def render_template(template, networks):
    """
    Render a network configuration file using Jinja templating.

    Args:
        template (str): The template to render
        networks (dict): A dictionary containing network information.

    Returns:
        str: A rendered unicode string derived from the Jinja template and the
            network configuration.
    """
    environment = Environment(trim_blocks=True, lstrip_blocks=True)
    template = environment.from_string(template)
    log.debug('Rendering: %s' % networks)
    data = template.render(dict(networks=networks))
    log.debug('Write:\n%s' % data)
    return data


def write_interfaces(path, network_configuration, use_netplan=False):
    """
    Write a network interface configuration file to configure network on the target.

    Args:
        path (str): The network configuration file that will be created for configuring network.
        network_configuration (dict): A dictionary containing network configuration.
        use_netplan (bool): A boolean that will determine the the type of configuration file to
        write (interfaces vs netplan). (optional)
    """
    interfaces = network_configuration.get('interfaces', list())
    networks = network_configuration.get('networks')

    for interface in interfaces:
        name, device = util.networking.lookup_interface(interface,
                                                        interface.get(
                                                            'missing_ok',
                                                            False))
        for network in networks:
            if use_netplan and 'netmask' in network and 'prefix' not in network:
                # convert netmask to a 'prefix' value used by netplan_interface.template
                network['prefix'] = sum([bin(int(x)).count('1')
                                        for x in network['netmask'].split('.')])
            if name == network.get('interface'):
                network['device'] = device.devname
                network['type'] = network.get('type', 'AF_INET')

    if use_netplan:
        template = NETPLAN_TEMPLATE
        deployment.write(path, render_template(template, network_configuration))
    else:
        template = INTERFACES_TEMPLATE
        deployment.write(path, render_template(template, networks))
