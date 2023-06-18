import uuid

from press.helpers.cli import find_in_path, run


class FileSystem(object):
    fs_type = ''
    parted_fs_type_alias = ''
    default_mount_options = ('defaults',)

    def __init__(self, label=None, mount_options=None, late_uuid=False):
        self.fs_label = label
        self.mount_options = mount_options or list(self.default_mount_options)
        if not late_uuid:
            self.fs_uuid = uuid.uuid4()
        self.require_fsck = False

    def create(self, device):
        """

        :param device:
        :raise NotImplemented:
        """
        raise NotImplemented(
            '%s base class should not be used.' % self.__name__)

    def generate_mount_options(self):
        if hasattr(self, 'mount_options'):
            options = self.mount_options
        else:
            options = self.default_mount_options

        return ','.join(options)

    def __repr__(self):
        return self.fs_type or 'Undefined'

    @classmethod
    def locate_command(cls, command_name):
        return find_in_path(command_name)

    @staticmethod
    def blkid_uuid(device):
        return run("blkid -s UUID -o value {}".format(device)).stdout.strip()
