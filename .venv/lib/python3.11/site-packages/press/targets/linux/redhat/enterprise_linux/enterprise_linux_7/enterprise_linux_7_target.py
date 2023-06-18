import logging
import os

import ipaddress

from press.exceptions import OSImageException
from press.helpers import deployment, sysfs_info
from press.targets import GeneralPostTargetError
from press.targets import util
from press.targets.linux.grub2_target import Grub2
from press.targets.linux.redhat.enterprise_linux.enterprise_linux_target \
    import EnterpriseLinuxTarget
from press.targets.linux.redhat.enterprise_linux.enterprise_linux_7 \
    import networking

log = logging.getLogger(__name__)


class EL7Target(EnterpriseLinuxTarget, Grub2):
    """
    Should work with CentOS and RHEL.
    """
    name = 'enterprise_linux_7'

    grub2_efi_command = None

    network_file_path = '/etc/sysconfig/network'
    network_scripts_path = '/etc/sysconfig/network-scripts'

    def get_efi_label(self):
        os_id = self.get_el_release_value('os')
        if 'red hat' in os_id.lower():
            return 'redhat', 'Red Hat Enterprise Linux'
        elif 'oracle' in os_id.lower():
            return 'redhat', 'Oracle Linux'
        elif 'centos' in os_id.lower():
            return 'centos', 'CentOS Linux'
        else:
            raise OSImageException('Could not determine EL distribution')

    def check_for_grub(self):
        _required_packages = ['grub2', 'grub2-tools']
        if sysfs_info.has_efi():
            os_id, os_label = self.get_efi_label()
            _required_packages += ['grub2-efi', 'efibootmgr', 'shim']
            self.grub2_config_path = '/boot/efi/EFI/{}/grub.cfg'.format(os_id)
            self.grub2_efi_command = (
                'efibootmgr --create --gpt '
                '--disk {} --part 1 --write-signature '
                '--label "{}" '
                '--loader /EFI/{}/shim.efi'.format(self.disk_target, os_label, os_id))
        if not self.packages_exist(_required_packages):
            self.baseline_yum(self.proxy)
            if self.install_packages(_required_packages):
                raise GeneralPostTargetError(
                    'Error installing required packages for grub2')
            self.revert_yum(self.proxy)

    def rebuild_initramfs(self):
        if not self.package_exists('dracut-config-generic'):
            self.baseline_yum(self.proxy)
            self.install_package('dracut-config-generic')
            self.revert_yum(self.proxy)

        kernels = os.listdir(self.join_root('/usr/lib/modules'))
        for kernel in kernels:
            initramfs_path = '/boot/initramfs-%s.img' % kernel
            if os.path.exists(self.join_root(initramfs_path)):
                log.info('Rebuilding %s' % initramfs_path)
                self.chroot('dracut -v -f %s %s' % (initramfs_path, kernel))

    def enable_ipv6(self):
        with open(self.join_root(self.network_file_path)) as f:
            contents = f.read()
        if "NETWORKING_IPV6" in contents:
            if "NETWORKING_IPV6=NO" in contents:
                contents.replace("NETWORKING_IPV6=NO", "NETWORKING_IPV6=YES")
            else:
                return
        else:
            contents += "\nNETWORKING_IPV6=YES\n"
        log.info("Enabling IPv6 in {}".format(self.network_file_path))
        with open(self.join_root(self.network_file_path), "w") as f:
            f.write(contents)

    def write_network_script(self, device, network_config, dummy=False):

        def generate_script_name(devname, vlan=None):
            if not vlan:
                return 'ifcfg-{0}'.format(devname)
            return 'ifcfg-{0}.{1}'.format(devname, vlan)

        def generate_script_path(script_name):
            return self.join_root(
                os.path.join(self.network_scripts_path, script_name))

        if dummy:
            _template = networking.DummyInterfaceTemplate(device.devname)
            vlan = None
        else:
            if network_config.get('type', 'AF_INET') == 'AF_INET6':
                self.enable_ipv6()
                interface_template = networking.IPv6InterfaceTemplate
            else:
                interface_template = networking.InterfaceTemplate

            ip_address = network_config.get('ip_address')
            gateway = network_config.get('gateway')
            prefix = network_config.get('prefix')
            vlan = network_config.get('vlan')
            if not prefix:
                prefix = ipaddress.ip_network(
                    "{ip_address}/{netmask}".format(**network_config),
                    strict=False).prefixlen

            _template = interface_template(
                device.devname,
                default_route=network_config.get('default_route', False),
                ip_address=ip_address,
                prefix=prefix,
                gateway=gateway,
                vlan=vlan)

        if vlan:
            script_name = generate_script_name(device.devname)
            script_path = generate_script_path(script_name)
            log.info('Writing {0}'.format(script_path))
            deployment.write(script_path, _template.generate_parent_interface())

        script_name = generate_script_name(device.devname, vlan)
        script_path = generate_script_path(script_name)
        log.info('Writing {0}'.format(script_path))
        deployment.write(script_path, _template.generate())

    def write_route_script(self, device, routes):
        script_name = 'route-%s' % device.devname
        script_path = self.join_root(
            os.path.join(self.network_scripts_path, script_name))
        log.info('Writing %s' % script_path)
        deployment.write(script_path, networking.generate_routes(routes))

    def configure_networks(self):
        network_configuration = self.press_configuration.get('networking')
        if not network_configuration:
            log.warn('Network configuration is missing')
            return

        interfaces = network_configuration.get('interfaces', list())
        networks = network_configuration.get('networks')
        for interface in interfaces:
            name, device = util.networking.lookup_interface(
                interface, interface.get('missing_ok', False))

            for network in networks:
                if name == network.get('interface'):
                    self.write_network_script(
                        device, network, dummy=network.get('dummy', False))
                    routes = network.get('routes')
                    if routes:
                        self.write_route_script(device, routes)

    def run(self):
        super(EL7Target, self).run()
        self.localization()
        self.update_host_keys()
        self.configure_networks()
        self.rebuild_initramfs()
        self.check_for_grub()
        self.install_grub2()
