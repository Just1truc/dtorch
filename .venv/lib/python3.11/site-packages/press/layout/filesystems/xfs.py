import logging
import re
from itertools import groupby

from press.helpers.cli import run
from press.layout.filesystems import FileSystem
from press.exceptions import (FileSystemCreateException,
                              FileSystemFindCommandException)

log = logging.getLogger(__name__)
supported_switches = {
    'inode_options': '-i',
    'naming_options': '-n',
    'global_metadata_options': '-m'
}


class XFSParam(object):

    def __init__(self, switch):
        self.switch = switch

    def has_value(self, val):
        return str(self.value) == str(val)


class XFSMultiParam(XFSParam):

    def __init__(self, switch, param=None, key=None, value=None):
        super(XFSMultiParam, self).__init__(switch)
        if param and str(param).find('=') >= 0:
            key, value = param.split('=', 1)
            self.key = key
            self.value = value
        else:
            self.key = key
            self.value = value

    def __eq__(self, other):
        if hasattr(other, 'key'):
            return self.key == other.key
        spec = re.compile('{}\s?{}\s?=.*'.format(self.switch, self.key))
        return spec.match(str(other)) is not None

    def __repr__(self):
        return str('{} {}={}'.format(self.switch, self.key, self.value))

    def __str__(self):
        return '{}={}'.format(self.key, self.value)

    def __hash__(self):
        return hash((self.switch, self.value))


class XFSSingleParam(XFSParam):

    def __init__(self, switch, value):
        super(XFSSingleParam, self).__init__(switch)
        self.value = value

    def __eq__(self, other):
        if hasattr(other, 'switch'):
            return self.switch == other.switch
        spec = re.compile('{}\s?.*'.format(self.switch))
        return spec.match(str(other)) is not None

    def __repr__(self):
        return str('{} {}'.format(self.switch, self.value))

    def __str__(self):
        return self.value

    def __hash__(self):
        return hash((self.switch, self.key, self.value))


class XFS(FileSystem):
    fs_type = 'xfs'
    # parted uses ext2 as a catch all for fs-type 0x83h (Linux Filesystem)
    parted_fs_type_alias = 'ext2'
    command_name = 'mkfs.xfs'

    xfs_required_mount_options = ['inode64', 'nobarrier']

    def __init__(self, label=None, mount_options=None, **extra):
        super(XFS, self).__init__(label, mount_options)
        self.extra = extra
        self.disable_crc_feature = extra.get("disable_crc_feature")

        for option in self.xfs_required_mount_options:
            if option not in self.mount_options:
                self.mount_options.append(option)

        self.command_path = self.locate_command(self.command_name)

        if not self.command_path:
            raise \
                FileSystemFindCommandException(
                    'Cannot locate %s in PATH' % self.command_name
                )

        self.full_command = \
            '{command_path} -m uuid={uuid}  -f ' + \
            '{inode_options}{naming_options}{global_metadata_options}' + \
            '{label_options}{device}'

        self.label_options = ''
        if self.fs_label:
            self.label_options = ' -L {} '.format(self.fs_label)

        self.parse_options()
        for option_name, switch in supported_switches.items():
            setattr(self, option_name, '')
            opts = self.get_option(switch)
            if opts:
                setattr(self, option_name, ' {} {} '.format(
                    switch, ','.join([str(opt) for opt in opts])))

    def parse_options(self):
        """
        Look for supported mkfs.xfs switches by name in self.extra.features
        and for each construct an object that can be manipulated and stored.

        Also check for special kwargs passed to XFS() that should add
        any missing switches to keep compatibility with older versions of
        xfsprogs.

        Currently supported special flags:
            - disable_crc_feature
        """
        features = self.extra.get('features', {})
        self.addl_options = set([])
        for option_name, switch in supported_switches.items():
            supplied_feature_options = features.get(option_name)
            if not isinstance(supplied_feature_options, list):
                continue
            for params in supplied_feature_options:
                while params:
                    p = params.popitem()
                    self.addl_options.add(
                        XFSMultiParam(switch, key=p[0], value=p[1]))

        if self.disable_crc_feature:
            disable_crc_options = (XFSMultiParam('-m', 'crc=0'),
                                   XFSMultiParam('-m', 'finobt=0'),
                                   XFSMultiParam('-n', 'ftype=0'))
            # This toggle implies the need to stay compatible with older
            # xfsprogs and kernels with xfs drivers
            # This requires turning off -mcrc=0,finobt=0 -nftype=0
            if not all([k in self.features for k in disable_crc_options]):
                self.addl_options.update(disable_crc_options)

    def get_options(self):
        """
        Return a parsed list of mkfs.xfs switches and the corresponding
        options grouped by the switch itself.

        Example:
            {
                '-n': [XFSMultiParam('-n ftype=0'),
                       XFSMultiParam('-n version=2')],
                '-i': [XFSSingleParam('-f')]
            }

        """
        # Chunk all of our supplied options into groups keyed by switch name
        return {switch: list(option) for switch, option in
                groupby(self.addl_options, key=lambda x: x.switch)} \
            if self.addl_options else {}

    def get_option(self, key):
        """
        Return a list of XFSMultiParam() or XFSSingleParam() for a given
        parsed switch (-i, -n .. etc.)
        """
        options = self.get_options()
        return options.get(key)

    def create(self, device):
        command = self.full_command.format(**dict(
            command_path=self.command_path,
            uuid=self.fs_uuid,
            label_options=self.label_options,
            inode_options=self.inode_options,
            naming_options=self.naming_options,
            global_metadata_options=self.global_metadata_options,
            device=device))
        log.info("Creating filesystem: %s" % command)
        result = run(command)
        if result.returncode:
            raise FileSystemCreateException(self.fs_label, command, result)
