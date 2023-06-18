import logging

from .enterprise_linux_6_target import EL6Target

log = logging.getLogger(__name__)


class CentOS6Target(EL6Target):
    name = 'centos_6'
