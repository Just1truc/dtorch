import os
import shlex


def join_root(root, path):
    return os.path.join(root, path.lstrip('/'))


def opts_modifier(l, appending, removing, quoted=True):
    config_line = l.split('=', 1)
    if not len(config_line) > 1:
        var = config_line[0]
        options = '\"\"'
    else:
        var, options = tuple(l.split('=', 1))
    if options.startswith('"') and options.endswith('"'):
        options = options[1:-1]
    options = shlex.split(options)
    options = [o for o in options if o not in removing]
    if appending:
        options += appending
    if quoted:
        return '{}=\"{}\"'.format(var, ' '.join(options))
    return '{}={}'.format(var, ' '.join(options))


def replace_grub_root_partition(line, default_grub_root_partition, modifying):
    """
        This function will force grub to detect EFI partition.
        As grub is not smart enough to detect EFI partition
    """
    return line.replace(default_grub_root_partition, modifying)
