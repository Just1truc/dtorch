#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging

from press.helpers.cli import run
from press.layout.filesystems import FileSystem
from press.exceptions import FileSystemCreateException, \
    FileSystemFindCommandException


log = logging.getLogger(__name__)


class NTFS(FileSystem):
    fs_type = 'ntfs'
    parted_fs_type_alias = 'NTFS'
    command_name = 'mkfs.ntfs'

    def __init__(self, label=None, mount_options=None, late_uuid=True, **extra):
        super(NTFS, self).__init__(label, mount_options,
                                   late_uuid=late_uuid)
        self.extra = extra

        self.command_path = self.locate_command(self.command_name)

        if not self.command_path:
            raise FileSystemFindCommandException(
                'Cannot locate {} in PATH'.format(self.command_name))

        self.full_command = '{command_path} -U {quick_option} {device}'

    def create(self, device, quick_flag=True):
        command = self.full_command.format(
                command_path=self.command_path,
                quick_option="-f" if quick_flag else '',
                device=device)

        log.info('Creating filesystem: {}'.format(command))
        result = run(command)
        if result.returncode:
            raise FileSystemCreateException(self.fs_label, command, result)
        self.fs_uuid = self.blkid_uuid(device)
