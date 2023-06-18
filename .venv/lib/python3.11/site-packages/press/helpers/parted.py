import logging

from press.helpers.cli import run, find_in_path
from press.helpers.sysfs_info import AlignmentInfo, append_sys
from press.helpers.udev import UDevHelper

log = logging.getLogger(__name__)


class PartedInterface(object):
    """
    Going to have to work with the parted command line tool due to time
    constraint. pyparted is not documented and messy.

    Once I can use libparted directly, I will move to that.
    """

    def __init__(self,
                 device,
                 parted_path='parted',
                 partition_start=1048576,
                 alignment=1048576):
        self.parted_path = parted_path
        if not find_in_path(self.parted_path):
            raise PartedInterfaceException(
                '%s does not exist.' % self.parted_path)
        self.device = device
        self.partition_start = partition_start
        self.alignment = alignment

        self.parted = self.parted_path + ' --script ' + self.device + ' unit b '
        #  Try to store the label, so that we'll raise a NullDiskException if we can't
        self.init_label = self.get_label()
        self.sector_size = self._get_sector_size()
        self.kernel_alignment_info = self.__get_alignment_info(device)

    @staticmethod
    def __get_alignment_info(device):
        uh = UDevHelper()
        devpath = uh.get_device_by_name(device)['DEVPATH']
        path = append_sys(devpath)
        return AlignmentInfo(path)

    def run_parted(self,
                   command,
                   raise_on_error=True,
                   ignore_error=False,
                   quiet=False):
        """
        parted does not use meaningful return codes. It pretty much returns 1 on
        any error and then prints an error message on to standard error stream.
        """
        result = run(
            self.parted + command, ignore_error=ignore_error, quiet=quiet)
        if result and raise_on_error:
            raise PartedException(result.stderr)
        return result

    def make_partition(self, type_or_name, start, end, fs_type=''):
        log.info("Creating partition type {}, start {}, end {}".format(
            type_or_name, start, end))
        fs_type_alias = ' {} '.format(fs_type) if fs_type else ' '
        command = 'mkpart {}{}{} {}'.format(type_or_name, fs_type_alias,
                                            start, end)
        return self.run_parted(command)

    def get_table(self, raw=False):
        result = self.run_parted(
            'print', raise_on_error=False, ignore_error=True, quiet=True)
        if result.returncode:
            if not result.stderr:
                #  udev sometimes maps /dev/loop devices before they are linked
                #  with losetup. When using parted on such a device, it will return 1
                #  with an no output. I need to find a better way to determine if a loop
                #  device is linked, ie, use losetup, ioctl, or /proc/partitions
                #  for now, we'll assume that missing output means the device is null
                raise NullDiskException(
                    'Cannot get table for uninitialized device')
            elif 'unrecognised disk label' in result.stderr:
                pass
            else:
                raise PartedException(result.stderr)

        if raw:
            return result
        return result.splitlines()

    def get_size(self):
        table = self.get_table()
        for line in table:
            if 'Disk' in line and line.split()[2][0].isdigit():
                return int(line.split()[2].strip('B'))

    def _get_info(self, term):
        table = self.get_table()
        for line in table:
            if term in line:
                return line.split(':')[1].strip()

    def get_model(self):
        return self._get_info('Model')

    def _get_sector_size(self):
        size = self._get_info('Sector size (logical/physical)')
        logical, physical = size.split('/')
        logical = int(logical[:-1])
        physical = int(physical[:-1])
        return dict(logical=logical, physical=physical)

    def get_disk_flags(self):
        return self._get_info('Disk Flags')

    def get_label(self):
        return self._get_info('Partition Table')

    @property
    def device_info(self):
        info = dict()
        info['model'] = self.get_model()
        info['device'] = self.device
        info['size'] = self.get_size()
        info['sector_size'] = self.sector_size
        info['partition_table'] = self.get_label()
        info['disk_flags'] = self.get_disk_flags()
        return info

    @property
    def partitions(self):
        p = list()
        table = self.get_table(raw=True)
        partition_type = self.get_label()

        if partition_type == 'unknown':
            return p

        part_data = table.split('\n\n')[1].splitlines()[1:]

        if not part_data:
            return p

        for part in part_data:
            part = part.split()
            part_info = dict()
            part_info['number'] = int(part[0].strip())
            part_info['start'] = int(part[1].strip('B'))
            part_info['end'] = int(part[2].strip('B'))
            part_info['size'] = int(part[3].strip('B'))

            if partition_type == 'msdos':
                part_info['type'] = part[4]
            p.append(part_info)

        return p

    @property
    def last_partition(self):
        partitions = self.partitions
        if not partitions:
            return None
        return partitions[-1]

    @property
    def extended_partition(self):
        if self.get_label() != 'msdos':
            return

        partitions = self.partitions

        if not partitions:
            return

        for part in partitions:
            if part['type'] == 'extended':
                return part

    def remove_partition(self, partition_number):
        """
        Uses run to spawn the process and looks for the return val.
        """
        command = self.parted + ' rm ' + str(partition_number)

        result = run(command)

        if result.returncode != 0:
            raise PartedException(
                'Could not remove partition: %d' % partition_number)

    def wipe_table(self):
        extended_partition = self.extended_partition
        if extended_partition:
            self.remove_partition(extended_partition['number'])

        for partition in self.partitions:
            self.remove_partition(partition['number'])

    def set_label(self, label='gpt'):
        result = run(self.parted + ' mklabel ' + label)
        if result.returncode != 0:
            raise PartedException('Could not create filesystem label')

    def set_name(self, number, name):
        """
        :param number:
        :param name:
        :return:
        """
        # The --script command line parser does not work properly, making it necessary to do some
        # silly escaping in order to support gpt partition names with spaces
        # name: BIOS boot partition becomes \'BIOS\ boot\ partition\', like I said, it is silly
        self.run_parted('name %d \\\'%s\\\'' % (number,
                                                name.replace(' ', '\\ ')))

    def set_flag(self, number, flag):
        log.info('Setting %s on partition #%d' % (flag, number))
        self.run_parted('set %d %s on' % (number, flag))

    @property
    def has_label(self):
        table = self.get_table()
        if not table:
            return False
        return True

    def create_partition(self, type_or_name, part_size, flags=None,
                         fs_type=None):
        """

        :rtype : int
        :param type_or_name:
        :param part_size:
        :param fs_type: parted fs_type alias must be one of
                   ext2
                   fat16, fat32
                   hfs, hfs+, hfsx
                   linux-swap
                   NTFS
                   reiserfs
                   ufs
                   btrfs

               These aliases map to partition type codes for MBR partitions or
               GUID file system hints on GPT.
        :param flags: list of partition flags
        """

        table_size = self.get_size()

        label = self.get_label()

        start = self.partition_start

        partition_number = 1

        last_partition = self.last_partition

        flags = flags or list()

        if last_partition:
            log.debug('Partition end (unmodified): %d' % last_partition['end'])
            aligned = \
                last_partition['end'] + (self.alignment - (last_partition['end'] % self.alignment))
            start = aligned
            partition_number = last_partition['number'] + 1

        end = start + part_size

        log.debug(
            'Partition start: %d, partition size: %d, end: %d, table size: %d' %
            (start, part_size, end, table_size))
        if end > table_size:
            #  Should be >= for msdos and end >= table_size - 34 sectors for gpt
            raise PartedInterfaceException(
                'The partition is too big. %d > %d (%d bytes)' %
                (end, table_size, end - table_size))

        if type_or_name == 'logical' and label == 'msdos':
            if not self.extended_partition:
                self.make_partition('extended', start, table_size - 1,
                                    fs_type=fs_type)
                start += self.partition_start
                partition_number = 5

        if label == 'gpt':
            # Parted command line parser and gpt support are crude bolt ons.
            # from parted source:
            #          /* The undocumented feature that mkpart sometimes takes a
            # partition name is next to useless, at least with a dvh
            # partition table, since it makes the "mkpart" command
            # fail unconditionally for a primary partition.  E.g.,
            # mkpart primary any-name xfs 4096s 5000s
            # requires the name, yet always fails, saying that only
            # logical partitions may have names.
            # If you want a name, use parted's separate "name" command.  */
            self.make_partition('unused', start, end, fs_type=fs_type)
        else:
            self.make_partition(type_or_name, start, end, fs_type=fs_type)

        if label == 'gpt':
            # obviously we need to determine the new partition's id.
            self.set_name(partition_number, type_or_name)

        for flag in flags:
            self.set_flag(partition_number, flag)

        return partition_number

    def remove_mbr(self):
        mbr_bytes = 512
        command = 'dd if=/dev/zero of=%s bs=%d count=1' % (self.device,
                                                           mbr_bytes)
        run(command)

    def remove_gpt(self):
        """
        512 Fake MBR
        512 GPT Header
        16KiB Primary Table
        16KiB Backup Table
        """
        gpt_bytes = 33792
        command = 'dd if=/dev/zero of=%s bs=%d count=1' % (self.device,
                                                           gpt_bytes)
        run(command)


class PartedException(Exception):
    pass


class PartedInterfaceException(Exception):
    pass


class NullDiskException(Exception):
    pass
