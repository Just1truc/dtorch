from . import cli
import logging
import os
import shutil

from tempfile import mkstemp

log = logging.getLogger(__name__)


def recursive_makedir(path, mode=0o775):
    if os.path.isdir(path):
        return False

    if os.path.exists(path):
        raise IOError('%s exists but is NOT a directory' % path)

    os.makedirs(path, mode)
    return True


def recursive_remove(path):
    if not os.path.exists(path):
        log.debug('Path, %s, does not exist, nothing to do' % path)
        return

    if os.path.islink(path):
        log.debug('Removing symbolic link at: %s' % path)
        os.unlink(path)
        return

    log.debug('Removing directory: %s' % path)
    shutil.rmtree(path)


def read(filename, splitlines=False):
    """
    Reads a file by absolute path and returns it's content.

    :param filename: Absolute path to a file.
    :type filename: str.
    :param splitlines: split the lines or just return string
    :type splitlines: boolean
    :return: str.

    """
    with open(filename, 'r') as f:
        read_data = f.read()
    if splitlines:
        return read_data.splitlines()
    return read_data


def write(filename, data, append=False, mode=0o644):
    """
    Writes to a file by absolute path.

    :param filename: Absolute path to a file.
    :type filename: str.

    :param data: A string of data which will be written to file.
    :type data: str.

    :param append: Should the file be appended to, or over-written.
    :type append: bool.

    :param mode: The linux permission mode to use for the file.
    :type mode: int.

    :return: None

    """

    # Use our append keyword argument to
    # determine our write mode.
    if append:
        write_mode = 'a'
    else:
        write_mode = 'w'

    with open(filename, write_mode) as f:
        f.write(data)

    # Last step lets change the file mode to specified mode
    os.chmod(filename, mode)


def replace_line_in_file(path, match, newline, mode=0o644):
    data = read(path)
    new_data = replace_line_matching(data, match, newline)
    write(path, new_data, mode=mode)


def replace_file(path, data):
    """
    Replace a file with data. Mode is maintained, ownership is not
    :param path:
    :param data:
    :return:
    """
    log.info('Replacing %s' % path)

    if not os.path.exists(path):
        log.warn(
            'The file you want to replace does not exist, writing a new file')
        return write(path, data.encode())

    if os.path.isdir(path):
        raise IOError('Cannot overwrite a directory')

    if not os.access(path, os.W_OK):
        raise IOError('Cannot write to target path')

    fp, temp_path = mkstemp()
    os.write(fp, data.encode())
    os.close(fp)
    shutil.copymode(path, temp_path)
    # Cannot rely on os.rename, doing this manually
    os.unlink(path)
    shutil.copy(temp_path, path)
    os.unlink(temp_path)


def replace_line_matching(data, match, newline):
    output = ""
    newline = newline if newline.endswith(
        os.linesep) else (newline + os.linesep)
    for line in data.splitlines(True):
        output += newline if match in line else line
    return output


def create_symlink(src, link_name, remove_existing_link=False):
    if os.path.exists(link_name):
        if os.path.islink(link_name):
            if remove_existing_link:
                os.unlink(link_name)
                log.warning('Removing existing link')
            else:
                log.warning('symbolic link %s already exists' % link_name)
                return
        log.error('File already exists at target: %s' % link_name)
        return

    os.symlink(src, link_name)


def remove_file(path):
    if not os.path.lexists(path):
        return
    if os.path.isdir(path):
        log.error('Path is a directory, use recursive_remove')
        return

    os.unlink(path)


def tar_extract(archive_path, chdir=''):
    # TODO: read header to determine compression / archive type?
    base_tar_cmd = 'tar --numeric-owner --xattrs --xattrs-include=* --acls'
    bzip_extensions = ('bz2', 'tbz', 'tbz2')
    compress_method = 'z'
    use_bzip = bool([i for i in bzip_extensions if archive_path.endswith(i)])
    if use_bzip:
        compress_method = 'j'

    return cli.run('%s -%sxf %s%s' %
                   (base_tar_cmd, compress_method, archive_path,
                    chdir and ' -C %s' % chdir or ''))


def create_fstab(fstab, target):
    path = os.path.join(target, 'etc/fstab')
    write(path, fstab)


def find_root(layout):
    for disk in layout.allocated:
        for partition in disk.partition_table.partitions:
            if partition.mount_point == '/':
                return partition

    for volume_group in layout.volume_groups:
        for logical_volume in volume_group.logical_volumes:
            if logical_volume.mount_point == '/':
                return logical_volume


def copy(src,
         dst,
         preserve_permissions=True,
         preserve_meta=True,
         preserve_owners=True):

    shutil.copy(src, dst)

    if preserve_permissions:
        shutil.copymode(src, dst)

    if preserve_meta:
        shutil.copystat(src, dst)

    if preserve_owners:
        st = os.stat(src)

        uid = st.st_gid
        gid = st.st_gid

        os.chown(dst, uid, gid)
