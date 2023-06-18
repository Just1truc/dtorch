"""sysfs helpers...

Really horrible class design, rewrite please. Lets say I wasn't invested.
"""
import os


class SYSFSInfoException(Exception):
    pass


def parse_cookie(path):
    try:
        with open(path) as cookie_file:
            return cookie_file.read().strip()
    except IOError:
        return ''


def append_sys(path):
    return os.path.join('/sys', path.lstrip('/'))


class SysFSInfo(object):
    attributes = {}
    devpath = ''

    def magic_set_attr(self, attr):
        val, path = self.attributes[attr]
        if val is not None:
            return val
        val = parse_cookie(os.path.join(self.devpath, path))
        if val.isdigit():
            val = int(val)
        self.attributes[attr] = (val, path)
        return val


class AlignmentInfo(SysFSInfo):

    def __init__(self, devpath):
        self.attributes = {
            'alignment_offset': (None, 'alignment_offset'),
            'physical_block_size': (None, 'queue/physical_block_size'),
            'logical_block_size': (None, 'queue/logical_block_size'),
            'optimal_io_size': (None, 'queue/optimal_io_size')
        }
        self.devpath = devpath

    @property
    def alignment_offset(self):
        return self.magic_set_attr('alignment_offset')

    @property
    def physical_block_size(self):
        return self.magic_set_attr('physical_block_size')

    @property
    def logical_block_size(self):
        return self.magic_set_attr('logical_block_size')

    @property
    def optimal_io_size(self):
        return self.magic_set_attr('optimal_io_size')


class NetDeviceInfo(SysFSInfo):
    """
    https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-class-net
    """
    # TODO: Finish implementation
    class_path = 'class/net'

    def __init__(self, devname):
        self.attributes = {
            'address': (None, 'address'),
            'carrier': (None, 'carrier'),
            'dev_port': (None, 'dev_port'),
            'dev_id': (None, 'dev_id'),
            'duplex': (None, 'duplex'),
            'speed': (None, 'speed'),
            'ifindex': (None, 'ifindex'),
        }
        self.devname = devname
        self.devpath = append_sys(os.path.join(self.class_path, self.devname))
        if not os.path.exists(self.devpath):
            raise SYSFSInfoException('%s does not exist' % self.devpath)

    @property
    def address(self):
        return self.magic_set_attr('address')

    @property
    def carrier(self):
        return self.magic_set_attr('carrier')

    @property
    def dev_port(self):
        return self.magic_set_attr('dev_port')

    @property
    def dev_id(self):
        return self.magic_set_attr('dev_id')

    @property
    def duplex(self):
        return self.magic_set_attr('duplex')

    @property
    def speed(self):
        return self.magic_set_attr('speed')

    @property
    def ifindex(self):
        return self.magic_set_attr('ifindex')

    @classmethod
    def list_interfaces(cls, exclude_loopback=True):
        path = append_sys(cls.class_path)
        interfaces = os.listdir(path)
        if exclude_loopback and 'lo' in interfaces:
            interfaces.remove('lo')
        return interfaces


def has_efi():
    """
    Should be True if booting in EFI mode, False otherwise
    :return: bool
    """
    return os.path.exists("/sys/firmware/efi")
