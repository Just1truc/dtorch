import logging
from press.helpers.cli import run

log = logging.getLogger(__name__)


def find_root(layout):
    for disk in layout.allocated:
        for partition in disk.partition_table.partitions:
            if partition.mount_point == '/':
                return partition

    for volume_group in layout.volume_groups:
        for logical_volume in volume_group.logical_volumes:
            if logical_volume.mount_point == '/':
                return logical_volume


def kexec(kernel, initrd, layout, kernel_type='bzImage', append=None):
    root_container = find_root(layout)
    log.info('Found root: %s' % root_container)
    root_uuid = root_container.file_system.fs_uuid
    append = append or []
    append.append('root=UUID=%s' % root_uuid)
    command = 'kexec --type=%s --initrd=%s --append="%s" %s' % (
        kernel_type, initrd, ' '.join(append), kernel)
    log.info('Kexec command: %s' % command)
    log.info('Goodbye cruel world')
    run(command)
