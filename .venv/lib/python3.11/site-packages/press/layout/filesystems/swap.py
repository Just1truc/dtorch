import logging

from press.helpers.cli import run
from . import FileSystem
from press.exceptions import FileSystemCreateException, FileSystemFindCommandException
from press.helpers.udev import UDevHelper

log = logging.getLogger(__name__)


class SWAP(FileSystem):
    fs_type = 'swap'
    parted_fs_type_alias = 'linux-swap'
    command_name = 'mkswap'

    def __init__(self, label=None, mount_options=None, **extra):
        super(SWAP, self).__init__(label, mount_options)

        # SWAP does not require any extra arguments
        del extra

        self.command_path = self.locate_command(self.command_name)

        if not self.command_path:
            raise \
                FileSystemFindCommandException(
                    'Cannot locate %s in PATH' % self.command_name)

        self.command = '{command_path} -U {uuid} {label_option} {device}'
        self.mount_options = mount_options or self.default_mount_options

        if self.fs_label:
            self.label_option = ' -L %s' % self.fs_label
        else:
            self.label_option = ''
        self.udev = UDevHelper()

    def create(self, device):
        command = self.command.format(**dict(
            command_path=self.command_path,
            label_option=self.label_option,
            device=device,
            uuid=self.fs_uuid))
        log.info("Creating filesystem: %s" % command)
        result = run(command)

        if result.returncode:
            raise FileSystemCreateException(self.fs_label, command, result)
