from press.targets.linux.grub2_target import Grub2


class Grub2Debian(Grub2):
    grub2_install_path = 'grub-install'
    grub2_mkconfig_path = 'grub-mkconfig'
    grub2_config_path = '/boot/grub/grub.cfg'
    grub2_efi_command = grub2_install_path
