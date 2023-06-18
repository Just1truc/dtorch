import logging

from press.helpers import deployment
from press.targets.linux.grub2_debian_target import Grub2Debian
from press.targets.linux.debian.debian_target import DebianTarget

log = logging.getLogger(__name__)


class Ubuntu1404Target(DebianTarget, Grub2Debian):
    name = 'ubuntu_1404'
    dist = 'trusty'

    def run(self):
        super(Ubuntu1404Target, self).run()
        self.grub_disable_recovery()
        self.install_grub2()
