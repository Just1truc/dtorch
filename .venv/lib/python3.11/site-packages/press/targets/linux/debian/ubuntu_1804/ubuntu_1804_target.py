import logging

from press.helpers import deployment, sysfs_info
from press.targets import GeneralPostTargetError
from press.targets.linux.grub2_debian_target import Grub2Debian
from press.targets.linux.debian.debian_target import DebianTarget
from press.targets.linux.debian.networking import debian_networking


log = logging.getLogger(__name__)


class Ubuntu1804Target(DebianTarget, Grub2Debian):
    name = 'ubuntu_1804'
    dist = 'bionic'
    netplan_interfaces_path = '/etc/netplan/01-netcfg.yaml'
    ifupdown_interfaces_path = '/etc/network/interfaces'
    netplan_readme_msg = ("Press defaults to /etc/network/interfaces, to enable netplan; "
                          "add use_netplan: True to your network configuration.")

    def __init__(self, press_configuration, layout, root, chroot_staging_dir):
        super(Ubuntu1804Target, self).__init__(press_configuration,
                                               layout, root, chroot_staging_dir)
        self.use_netplan = press_configuration.get('image_opts', {}).get('use_netplan', False)
        self.default_networking = self.join_root('/etc/default/networking')
        self.systemd_generators_directory = self.join_root('/etc/systemd/system-generators')
        self.netplan_systemd_generator_file = self.systemd_generators_directory + '/netplan'
        self.netplan_readme_file = self.join_root('/etc/netplan/README.TXT')

    def check_for_grub(self):
        """Check for grub multiboot boot loader and install its required packages if needed."""
        if sysfs_info.has_efi():
            _required_packages = ['shim-signed', 'grub-efi-amd64-signed']
            if self.install_packages(_required_packages):
                raise GeneralPostTargetError('Error installing required packages for grub2')

    def write_interfaces(self):
        self.netplan_or_ifupdown()
        interfaces_path = self.join_root(self.interfaces_path)
        if self.network_configuration:
            log.info('Writing network configuration')
            debian_networking.write_interfaces(interfaces_path, self.network_configuration,
                                               use_netplan=self.use_netplan)

    def netplan_or_ifupdown(self):
        if self.install_packages(['netplan.io', 'ifupdown']):
            raise GeneralPostTargetError('Error installing required network packages.')
        if self.use_netplan:
            log.info("Using Netplan network configuration")
            self.interfaces_path = self.netplan_interfaces_path
            self.ensure_netplan_enabled()
        else:
            log.info("Using ifupdown network configuration")
            self.interfaces_path = self.ifupdown_interfaces_path
            self.ensure_ifupdown_enabled()

    def ensure_netplan_enabled(self):
        # removing self.netplan_systemd_generator_file allows netplan to run at boot. Disabling and
        # masking of networking.service and setting CONFIGURE_INTERFACES=no disables ifupdown
        deployment.remove_file(self.netplan_systemd_generator_file)
        self.chroot('systemctl disable networking.service')
        self.chroot('systemctl mask networking.service')
        deployment.replace_line_in_file(self.default_networking, 'CONFIGURE_INTERFACES', 'CONFIGURE_INTERFACES=no')

    def ensure_ifupdown_enabled(self):
        # linking the self.netplan_systemd_generator_file to /dev/null prevents netplan from starting
        # at boot, and enabling networking.service and setting CONFIGURE_INTERFACES=yes enables ifupdown
        deployment.recursive_makedir(self.systemd_generators_directory)
        deployment.create_symlink('/dev/null', self.netplan_systemd_generator_file)
        self.chroot('systemctl unmask networking.service')
        self.chroot('systemctl enable networking.service')
        deployment.replace_line_in_file(self.default_networking, 'CONFIGURE_INTERFACES', 'CONFIGURE_INTERFACES=yes')
        deployment.write(self.netplan_readme_file, self.netplan_readme_msg)

    def run(self):
        """Run Debian functions."""
        super(Ubuntu1804Target, self).run()
        self.grub_disable_recovery()
        self.check_for_grub()
        self.install_grub2()
