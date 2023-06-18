import logging

from press.helpers import deployment, sysfs_info
from press.targets import GeneralPostTargetError
from press.targets.linux.grub2_debian_target import Grub2Debian
from press.targets.linux.debian.debian_target import DebianTarget

log = logging.getLogger(__name__)


class Debian8Target(DebianTarget, Grub2Debian):
    name = 'debian_8'
    dist = 'jessie'

    def check_for_grub(self):
        if sysfs_info.has_efi():
            _required_packages = ['shim', 'grub-efi-amd64']
            if self.install_packages(_required_packages):
                raise GeneralPostTargetError(
                    'Error installing required packages for grub2')

    def run(self):
        super(Debian8Target, self).run()
        self.grub_disable_recovery()
        self.check_for_grub()
        self.install_grub2()
