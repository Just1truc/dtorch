import logging
import os

from press.helpers import deployment, sysfs_info
from press.targets import Target
from press.targets import util

log = logging.getLogger(__name__)


class Grub(Target):
    grub_cmdline_config_path = '/boot/grub/grub.conf'
    grub_cmdline_name = 'kernel'
    default_grub_root_partition = '(hd0,0)'
    grub_efi_bootloader_name = " "

    grub_install_path = 'grub-install'
    grub_install_opts = '--recheck'
    grubby_path = 'grubby'

    @property
    def bootloader_configuration(self):
        return self.press_configuration.get('bootloader')

    @property
    def kernel_parameters(self):
        return self.bootloader_configuration.get('kernel_parameters')

    @property
    def disk_targets(self):
        return self.disk_target if isinstance(self.disk_target, list) else \
                [self.disk_target]

    @property
    def disk_target(self):
        _target = self.bootloader_configuration.get('target', 'first')
        if _target == 'first':
            return self.layout.disks.keys()[0]
        return _target

    def update_kernel_parameters(self):
        """
        A little hacktastic
        :return:
        """

        appending = self.kernel_parameters.get('append', list())
        removing = self.kernel_parameters.get('remove', list())
        modifying = self.kernel_parameters.get('modify', list())

        if not (appending or removing):
            return

        full_path = self.join_root(self.grub_cmdline_config_path)
        if not os.path.exists(full_path):
            log.warn('Grub configuration is missing from image')
            return

        data = deployment.read(full_path, splitlines=True)

        modified = False
        for idx in range(len(data)):
            line = data[idx]
            line = line.strip()

            if line and line[0] == '#':
                continue

            # grub1 is not smart to find efi partition,
            # have to force it to find.
            if self.default_grub_root_partition in line and sysfs_info.has_efi(
            ):
                if modifying:
                    data[idx] = util.misc.replace_grub_root_partition(
                        line, self.default_grub_root_partition, modifying[0])
                    modified = True
                    continue

            if self.grub_cmdline_name in line:
                data[idx] = util.misc.opts_modifier(
                    line, appending, removing, quoted=False)
                log.debug('{} > {}'.format(line, data[idx]))
                modified = True
                continue

        if modified:
            log.info('Updating {}'.format(self.grub_cmdline_config_path))
            deployment.replace_file(full_path, '\n'.join(data) + '\n')
        else:
            log.warn('Grub configuration was not updated, no matches!')

    def install_grub(self):
        if not self.bootloader_configuration:
            log.warn('Bootloader configuration is missing')
            return

        self.update_kernel_parameters()

        log.info('Generating grub configuration')
        # TODO(mdraid): We may need run grub2-mkconfig on all targets?
        root_partition = deployment.find_root(self.layout)
        root_uuid = root_partition.file_system.fs_uuid
        log.info('Configuring mtab')
        self.chroot('grep -v rootfs /proc/mounts > /etc/mtab')

        kernels = os.listdir(self.join_root('/lib/modules'))
        for kernel in kernels:
            self.chroot('{} --args=root=UUID={} '
                        '--update-kernel=/boot/vmlinuz-{}'.format(
                            self.grubby_path, root_uuid, kernel))
        for disk in self.disk_targets:
            log.info('Installing grub on {}'.format(disk))
            self.chroot('{} {} {}'.format(self.grub_install_path, self.grub_install_opts, disk))
            if sysfs_info.has_efi():
                log.info('Configuring bootloader for EFI')
                # For EFI kick to work we have to copy grub.conf from /boot/grub
                self.chroot('cp /boot/grub/grub.conf /boot/efi/EFI/redhat/')
                self.chroot('efibootmgr -c -d {} -L "{}"'.format(
                    disk, self.grub_efi_bootloader_name))
