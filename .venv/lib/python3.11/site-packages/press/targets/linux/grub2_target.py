import logging
import os

from press.helpers import deployment, sysfs_info
from press.targets import Target
from press.targets import util

log = logging.getLogger(__name__)


class Grub2(Target):
    grub2_cmdline_config_path = '/etc/default/grub'
    grub2_cmdline_name = 'GRUB_CMDLINE_LINUX'

    grub2_install_path = 'grub2-install'
    grub2_mkconfig_path = 'grub2-mkconfig'
    grub2_config_path = '/boot/grub2/grub.cfg'

    grub2_efi_command = ('{} --target=x86_64-efi '
                         '--efi-directory=/boot/efi '
                         '--bootloader-id=grub '
                         '--debug'.format(grub2_install_path))

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
            return list(self.layout.disks.keys())[0]
        return _target

    def update_kernel_parameters(self):
        """
        A little hacktastic
        :return:
        """

        appending = self.kernel_parameters.get('append', list())
        removing = self.kernel_parameters.get('remove', list())

        if not (appending or removing):
            return

        full_path = self.join_root(self.grub2_cmdline_config_path)
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

            if self.grub2_cmdline_name in line:
                data[idx] = util.misc.opts_modifier(line, appending, removing)
                log.debug('%s > %s' % (line, data[idx]))
                modified = True
                continue

        if modified:
            log.info('Updating %s' % self.grub2_cmdline_config_path)
            deployment.replace_file(full_path, '\n'.join(data) + '\n')
        else:
            log.warn('Grub configuration was not updated, no matches!')

    def install_grub2(self):
        if not self.bootloader_configuration:
            log.warn('Bootloader configuration is missing')
            return

        self.update_kernel_parameters()
        self.clear_grub_cmdline_linux_default()

        log.info('Generating grub configuration')
        self.chroot('{} -o {}'.format(self.grub2_mkconfig_path,
                                      self.grub2_config_path))

        if sysfs_info.has_efi():
            log.info("Installing EFI enabled grub")
            self.chroot(self.grub2_efi_command)
        else:
            command = '{} --target=i386-pc --recheck --debug'.format(
                self.grub2_install_path)
            for disk in self.disk_targets:
                log.info('Installing grub on {}'.format(disk))
                self.chroot("{} {}".format(command, disk))

    def update_grub_configuration(self, match, newvalue):
        grub_configuration = deployment.read(
            self.join_root(self.grub2_cmdline_config_path))
        log.info('Setting {} in {}'.format(newvalue,
                                           self.grub2_cmdline_config_path))
        updated_grub_configuration = deployment.replace_line_matching(
            grub_configuration, match, newvalue)
        deployment.write(
            self.join_root(self.grub2_cmdline_config_path),
            updated_grub_configuration)

    def grub_disable_recovery(self):
        match = 'GRUB_DISABLE_RECOVERY'
        value = 'GRUB_DISABLE_RECOVERY=true'
        self.update_grub_configuration(match, value)

    def clear_grub_cmdline_linux_default(self):
        match = 'GRUB_CMDLINE_LINUX_DEFAULT'
        value = 'GRUB_CMDLINE_LINUX_DEFAULT=""'
        self.update_grub_configuration(match, value)
