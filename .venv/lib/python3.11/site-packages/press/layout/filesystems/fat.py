#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging

from press.helpers.cli import run
from press.layout.filesystems import FileSystem
from press.exceptions import FileSystemCreateException, \
    FileSystemFindCommandException

log = logging.getLogger(__name__)


class FAT32(FileSystem):
    fs_type = 'fat'
    parted_fs_type_alias = 'fat32'
    command_name = 'mkfs.vfat'

    def __init__(self, label=None, mount_options=None, **extra):
        super(FAT32, self).__init__(label, mount_options,
                                    extra.get('late_uuid'))
        self.extra = extra

        self.command_path = self.locate_command(self.command_name)

        if not self.command_path:
            raise FileSystemFindCommandException(
                'Cannot locate {} in PATH'.format(self.command_name))

        self.full_command = '{command_path} {device}'

    def create(self, device):
        command = self.full_command.format(
            command_path=self.command_path, device=device)

        log.info('Creating filesystem: {}'.format(command))
        result = run(command)
        if result.returncode:
            raise FileSystemCreateException(self.fs_label, command, result)


class EFI(FAT32):
    fs_type = 'vfat'

    def __init__(self, label=None, mount_options=None, late_uuid=True, **extra):
        super(EFI, self).__init__(
            label, mount_options, late_uuid=late_uuid, **extra)
        self.full_command = '{command_path} -s2 -F 32 {device}'

    def create(self, device):
        super(EFI, self).create(device)
        self.fs_uuid = self.blkid_uuid(device)
