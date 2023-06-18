import collections
import logging
import os

import ipaddress

from press.helpers import deployment, sysfs_info
from press.helpers.package import get_press_version
from press.targets import GeneralPostTargetError
from press.targets import util
from press.targets.linux.grub_target import Grub
from press.targets.linux.redhat.enterprise_linux.enterprise_linux_target \
    import EnterpriseLinuxTarget
from press.targets.linux.redhat.enterprise_linux.enterprise_linux_7 \
    import networking

log = logging.getLogger(__name__)


class EL6Target(EnterpriseLinuxTarget, Grub):
    """
    Should work with CentOS and RHEL.
    """
    name = 'enterprise_linux_6'

    ssh_protocol_2_key_types = ('rsa', 'ecdsa', 'dsa')
    rpm_path = '/bin/rpm'
    network_file_path = '/etc/sysconfig/network'
    network_scripts_path = '/etc/sysconfig/network-scripts'
    sysconfig_scripts_path = 'etc/sysconfig'

    def get_efi_label(self):
        os_id = self.get_el_release_value('os')
        if 'red hat' in os_id.lower():
            return 'redhat', 'Red Hat Enterprise Linux'
        return 'centos', 'CentOS Linux'

    def check_for_grub(self):
        _required_packages = ['grub', 'grubby']
        if sysfs_info.has_efi():
            os_id, os_label = self.get_efi_label()
            self.grub_efi_bootloader_name = os_label

        if not self.packages_exist(_required_packages):
            if not self.install_packages(_required_packages):
                raise GeneralPostTargetError(
                    'Error installing required packages for grub')

    def rebuild_initramfs(self):
        _required_packages = ['dracut', 'dracut-kernel']
        if not self.packages_exist(_required_packages):
            if not self.install_package(_required_packages):
                raise GeneralPostTargetError(
                    'Error install required packages for dracut')

        kernels = os.listdir(self.join_root('/lib/modules'))
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
        script_name = 'ifcfg-%s' % device.devname
        script_path = self.join_root(
            os.path.join(self.network_scripts_path, script_name))
        if dummy:
            _template = networking.DummyInterfaceTemplate(device.devname)
        else:
            if network_config.get('type', 'AF_INET') == 'AF_INET6':
                self.enable_ipv6()
                interface_template = networking.IPv6InterfaceTemplate
            else:
                interface_template = networking.InterfaceTemplate

            ip_address = network_config.get('ip_address')
            gateway = network_config.get('gateway')
            prefix = network_config.get('prefix')
            if not prefix:
                prefix = ipaddress.ip_network(
                    "{ip_address}/{netmask}".format(
                        **network_config).decode("utf-8"),
                    strict=False).prefixlen

            _template = interface_template(
                device.devname,
                default_route=network_config.get('default_route', False),
                ip_address=ip_address,
                prefix=prefix,
                gateway=gateway)
        log.info('Writing %s' % script_path)
        deployment.write(script_path, _template.generate())

    def write_route_script(self, routes):
        script_name = 'static-routes'
        script_path = self.join_root(
            os.path.join(self.sysconfig_scripts_path, script_name))
        log.info('Writing %s' % script_path)
        deployment.write(script_path, self.generate_static_routes(routes))

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
                        self.write_route_script(routes)

    def set_hostname(self):
        network_configuration = self.press_configuration.get(
            'networking', dict())
        hostname = network_configuration.get('hostname')
        if not hostname:
            log.warn('Hostname not defined')
            return
        log.info('Setting hostname: %s' % hostname)
        sysconfig_network = deployment.read(
            self.join_root('/etc/sysconfig/network'))
        updated_sysconfig_network = deployment.replace_line_matching(
            sysconfig_network, 'HOSTNAME', 'HOSTNAME=%s' % hostname)
        deployment.write(
            self.join_root('/etc/sysconfig/network'), updated_sysconfig_network)
        deployment.write(self.join_root('/etc/hostname'), hostname + '\n')

    @staticmethod
    def generate_static_routes(routes):
        script = '# Generated by press v%s\n' % get_press_version()
        for idx in range(len(routes)):
            script += 'any net %s gw %s\n' % (routes[idx]['cidr'],
                                              routes[idx]['gateway'])
        return script

    def generate_pseudo_mount_entry(self,
                                    filesystem=None,
                                    directory=None,
                                    fs_type=None,
                                    options='defaults',
                                    dump_pass='0 0'):

        entry = '%s\t\t%s\t\t%s\t\t%s\t\t%s\n' % (filesystem, directory,
                                                  fs_type or filesystem,
                                                  options, dump_pass)
        return entry

    def add_pseudo_mounts(self):
        """
        EL6 needs a few items in fstab that other press distros didn't:
        <filesystem>            <dir>                   <type>  <options>       <dump pass>
        tmpfs		            /dev/shm		        tmpfs	defaults,nosuid,nodev,noexec    0 0
        devpts                  /dev/pts                devpts  gid=5,mode=620  0 0
        sysfs                   /sys                    sysfs   defaults        0 0
        proc                    /proc                   proc    defaults        0 0
        """
        pseudo_mounts = [
            dict(
                filesystem='tmpfs',
                directory='/dev/shm',
                options='defaults,nosuid,nodev,noexec'),
            dict(
                filesystem='devpts',
                directory='/dev/pts',
                options='gid=5,mode=620'),
            dict(filesystem='sysfs', directory='/sys'),
            dict(filesystem='proc', directory='/proc')
        ]

        fstab_entry = ''
        for mount in pseudo_mounts:
            fstab_entry += self.generate_pseudo_mount_entry(**mount)

        fstab_file = self.join_root('/etc/fstab')
        log.info('Writing pseudo filesystem mounts to /etc/fstab.')
        deployment.write(fstab_file, fstab_entry, append=True)

    def generate_udev_net_entry(self,
                                subsystem='net',
                                action='add',
                                drivers='?*',
                                attributes=None,
                                kernel_iface_name=None,
                                iface_name=None):
        """ Generate a single udev entry line for a net device """
        if not attributes or \
                not all(attributes.get(k) for k in ('address', 'type')):
            log.error('Missing required information to generate net udev rule')
            return
        return 'SUBSYSTEM=="{}", ACTION=="{}", DRIVERS=="{}", ' \
               'ATTR{{address}}=="{}", ATTR{{type}}=="{}", KERNEL=="{}", ' \
               'NAME="{}"'.format(subsystem, action, drivers,
                                  attributes['address'], attributes['type'],
                                  kernel_iface_name, iface_name)

    def generate_udev_rules(self):
        """ Generate udev entries for net devices found in /sys/class/net """

        def probe_net_devices():
            """ Return a list of tuples, e.g. [(eth0, '00:aa:bb:cc:dd')] """
            devices = list()
            class_path = '/sys/class/net'
            NetDevice = collections.namedtuple('NetDevice', 'name mac')
            for dev in list(set(os.listdir(class_path)) - {'lo'}):
                with open(os.path.join(class_path, dev, 'address')) as f:
                    devices.append(NetDevice(dev, f.read().strip()))
            log.debug(devices)
            return devices

        udev_rules = []
        for device in probe_net_devices():
            udev_rules.append(
                self.generate_udev_net_entry(
                    attributes=dict(address=device.mac, type=1),
                    kernel_iface_name='eth*',
                    iface_name=device.name))
        return udev_rules

    def copy_udev_rules(self, create_if_missing=False):
        rules_file = '/etc/udev/rules.d/70-persistent-net.rules'
        if not os.path.exists(rules_file):
            log.warn('Host 70-persistent-net.rules is missing')
            if create_if_missing:
                log.info('Generating 70-persistent-net.rules')
                deployment.write(
                    self.join_root(rules_file),
                    "\n\n".join(self.generate_udev_rules()) + "\n")
            return
        deployment.write(
            self.join_root(rules_file), deployment.read(rules_file))

    def set_timezone(self, timezone):
        log.debug('Setting timezone, EL6 style')
        localtime_path = self.join_root('/etc/localtime')
        clock_file_path = self.join_root('/etc/sysconfig/clock')
        zone_info = self.join_root(
            os.path.join('/usr/share/zoneinfo/', timezone))

        deployment.copy(zone_info, localtime_path)

        data = 'ZONE="%s"\n' % timezone
        log.info('Updating /etc/sysconfig/clock: %s' % data)
        deployment.replace_file(clock_file_path, data)

    def run(self):
        super(EL6Target, self).run()
        self.localization()
        self.update_host_keys()
        self.configure_networks()
        self.add_pseudo_mounts()
        self.copy_udev_rules(create_if_missing=True)
        self.rebuild_initramfs()
        self.check_for_grub()
        self.install_grub()
