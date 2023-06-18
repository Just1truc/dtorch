"""
simple mdadm command wrapper, it doesn't support much. Software RAID? really?
"""

import logging
import os
from press.helpers.cli import run, find_in_path

log = logging.getLogger(__name__)
DEFAULT_METADATA = '1.2'


class MDADMError(Exception):
    pass


class MDADM(object):
    """
    class wrapper for mdadm
    """

    def __init__(self, mdadm_path='mdadm'):
        self.mdadm = mdadm_path
        if not find_in_path(self.mdadm):
            raise MDADMError('Missing mdadm binary: %s' % self.mdadm)

    def run_mdadm(self,
                  command,
                  raise_on_error=True,
                  ignore_error=False,
                  quiet=False):
        full_command = '%s %s' % (self.mdadm, command)
        result = run(
            full_command, ignore_error=ignore_error, quiet=quiet, _input='yes')
        if result and raise_on_error:
            raise MDADMError('%d: %s : %s' % (result.returncode, result.stdout,
                                              result.stderr))
        return result

    def create(self, device, level, members, name='',
               metadata=DEFAULT_METADATA):
        if not name:
            name = os.path.basename(device)

        raid_devices = len(members)

        command = '-C --level={level} {device} --metadata={metadata} ' \
                  '--raid-devices={raid_devices} --name={name} ' \
                  '{members}'.format(level=level,
                                     device=device,
                                     metadata=metadata,
                                     raid_devices=raid_devices,
                                     name=name,
                                     members=' '.join(members))

        log.info('Creating software RAID: %s [%s]' % (device,
                                                      ', '.join(members)))
        return self.run_mdadm(command)

    def stop(self, device):
        log.info('Stopping software RAID on %s' % device)
        command = '--stop {device}'.format(device=device)
        return self.run_mdadm(command)

    def zero_superblock(self, device):
        log.info('Clearing supper block on: %s' % device)
        command = '--zero-superblock {device}'.format(device=device)
        return self.run_mdadm(command, ignore_error=True, quiet=True)

    @staticmethod
    def zero_4k(device):
        log.info('Removing 4K from the beginning of the device: 1.2 metadata')
        command = 'dd if=/dev/zero of=%s bs=%d count=10' % (device, 4096)
        run(command)

    def remove(self, device):
        log.info('Removing %s' % device)
        command = '--remove %s' % device
        return self.run_mdadm(command)

    def fail_remove_member(self, device, member):
        command = '%s --fail %s --remove %s' % (device, member, member)
        return self.run_mdadm(command)

    @staticmethod
    def info():
        try:
            with open('/proc/mdstat') as fp:
                return fp.read()
        except IOError:
            return ''

    def _get_mdstat_line(self, device):
        info = self.info()
        if not info:
            return False

        entries = self.info().split('\n\n')

        for entry in entries:
            for line in entry.split('\n'):
                if line.find(os.path.basename(device)) == 0:
                    return line

    def is_present(self, device):
        return bool(self._get_mdstat_line(device))

    def get_members(self, device):
        line = self._get_mdstat_line(device)
        detail = line.split(':')[1]
        log.debug('detail0: ' + detail + ' ' + line)
        if detail.split()[0] == 'inactive':
            members = detail.split()[1:]
        else:
            members = detail.split()[2:]
        log.debug('members: %s' % members)
        return ['/dev/%s' % x.split('[')[0] for x in members]


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    from press.helpers.parted import PartedInterface
    mdadm = MDADM()

    mdadm.info()

    pi = PartedInterface(device='/dev/loop0')
    pi.wipe_table()
    pi.remove_mbr()
    mdadm.zero_4k('/dev/loop0')

    pi = PartedInterface(device='/dev/loop1')
    pi.wipe_table()
    pi.remove_mbr()
    mdadm.zero_4k('/dev/loop1')

    result = mdadm.create(
        '/dev/md0', 1, ['/dev/loop0', '/dev/loop1'], name='Pony')

    print(result)
    print(result.stderr)

    mdadm.info()

    input('Press Enter...')

    result = mdadm.stop('/dev/md0')
    print(result)
    print(result.stderr)

    mdadm.zero_superblock('/dev/loop0')
    mdadm.zero_superblock('/dev/loop1')

    mdadm.info()
