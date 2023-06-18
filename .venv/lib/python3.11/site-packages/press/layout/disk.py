import logging

from press.exceptions import PartitionValidationError
from size import Size, PercentString

GPT = 'gpt'
MSDOS = 'msdos'
GPT_BACKUP_SIZE = 17408

log = logging.getLogger(__name__)


class Disk(object):

    def __init__(self,
                 devname=None,
                 devlinks=None,
                 devpath=None,
                 partition_table=None,
                 size=0,
                 sector_size=512):
        """
        """
        self.devname = devname
        self.devlinks = devlinks or list()
        self.devpath = devpath
        self.size = Size(size)
        self.partition_table = partition_table
        self.sector_size = sector_size

    def new_partition_table(self,
                            table_type,
                            partition_start=1048576,
                            alignment=1048576):
        """Instantiate and link a PartitionTable object to Disk instance
        """
        self.partition_table = PartitionTable(
            table_type,
            self.size.bytes,
            partition_start=partition_start,
            alignment=alignment,
            sector_size=self.sector_size)

    def __repr__(self):
        return '%s: %s' % (self.devname, self.size.humanize)


class PartitionTable(object):

    def __init__(self,
                 table_type,
                 size,
                 partition_start=1048576,
                 alignment=1048576,
                 sector_size=512):
        """Logical representation of a partition
        """

        valid_types = [GPT, MSDOS]
        if table_type not in valid_types:
            raise ValueError('table not supported: %s' % table_type)
        self.type = table_type
        self.size = Size(size)
        self.partition_start = Size(partition_start)
        self.alignment = Size(alignment)
        self.sector_size = Size(sector_size)

        # This variable is used to store a pointer to the end of the partition
        # structure + (alignment - ( end % alignment ) )
        self.partition_end = self.partition_start
        self.partitions = list()

    def _validate_partition(self, partition):
        if partition.size < Size('1MiB'):
            raise PartitionValidationError('The partition cannot be < 1MiB.')

        if self.partitions:
            if self.size < self.current_usage + partition.size:
                raise PartitionValidationError(
                    'The partition is too big. %s < %s' %
                    (self.size - self.current_usage, partition.size))
        elif self.size < partition.size + self.partition_start:
            raise PartitionValidationError('The partition is too big. %s < %s' %
                                           (self.size - self.current_usage,
                                            partition.size))

    def calculate_aligned_size(self, size):
        return size + self.alignment - size % self.alignment

    def calculate_total_size(self, size):
        """Calculates total size after alignment.
        """
        if isinstance(size, PercentString):
            if size.free:
                log.debug('Calculating percentage of free space')
                size = self.get_percentage_of_free_space(size.value)
            else:
                log.debug('Calculating percentage of total space')
                size = self.get_percentage_of_usable_space(size.value)
        return size

    @property
    def is_gpt(self):
        return self.type == GPT

    @property
    def is_msdos(self):
        return self.type == MSDOS

    @property
    def current_usage(self):
        """
        Factors in alignment
        """
        if not self.partitions:
            return self.partition_start
        usage = self.partition_start
        for partition in self.partitions:
            usage += partition.size + (
                self.alignment - partition.size % self.alignment)
        return usage

    @property
    def free_space(self):
        """
        Calculate free space using standard logic
        """
        if not self.partitions:
            free = self.size - self.partition_start
        else:
            free = self.size - self.current_usage

        if self.type == GPT:
            free -= Size(GPT_BACKUP_SIZE)
        else:
            free -= Size(1)

        log.debug('Free space: %d' % free.bytes)
        return free

    @property
    def physical_volumes(self):
        return [
            partition for partition in self.partitions
            if 'lvm' in partition.flags
        ]

    def add_partition(self, partition):
        if partition.percent_string:
            adjusted_size = self.calculate_total_size(partition.percent_string)
        else:
            adjusted_size = self.calculate_total_size(partition.size)

        # Adjust size to conform with parted logic

        if not adjusted_size.bytes % self.sector_size.bytes:
            # We've landed on a sector boundary, increase the size of the partition to the next
            # sector boundary - 1 ( parted does this )
            log.debug('Landed on sector boundary, growing by one sector - 1')
            adjusted_size + self.sector_size - Size(1)

        #  TODO: Make Size operator overloads work with ints
        # percentage based partitions should NEVER hit this, this is for explicit definitions
        if self.is_gpt \
                and self.partition_end.bytes + adjusted_size.bytes > self.size.bytes - GPT_BACKUP_SIZE:
            # parted reserves 17408 bytes for a gpt backup at the end of the disk
            adjusted_size -= self.partition_end + adjusted_size - self.size - Size(
                GPT_BACKUP_SIZE)

        if self.is_msdos and self.partition_end + adjusted_size == self.size:
            # 100% full - 1, the last byte overruns disk geometry
            adjusted_size.bytes -= 1

        partition.size = adjusted_size
        log.info('Adding partition: %s size: %d, flags: %s' %
                 (partition.name, partition.size.bytes, partition.flags))
        self._validate_partition(partition)
        # allocate the partition (mapped to the physical world)
        partition.allocated = True
        self.partitions.append(partition)
        self.partition_end += adjusted_size
        log.debug('Partition end: %d, table size %d' %
                  (self.partition_end.bytes, self.size.bytes))
        if self.size < self.partition_end:
            raise PartitionValidationError('Logic is wrong, this is too big')
            # update partition size

    def get_percentage_of_free_space(self, percent):
        free_space = self.free_space.bytes
        result = free_space * percent
        log.debug('%d * %f = %d' % (free_space, percent, result))
        return Size(result)

    def get_percentage_of_usable_space(self, percent):
        return Size((self.size - self.partition_start).bytes * percent)

    def __repr__(self):
        out = 'Table: %s (%s / %s)\n' % (self.type, self.current_usage,
                                         self.size)
        for idx, partition in enumerate(self.partitions):
            out += '%d: %s %s\n' % (idx, partition.name, partition.size)
        return out


class Partition(object):
    """
    A partition class, used to represent a partition logically. When this
    logical representation is writen to the disk, the parted_id class variable
    will be populated with the physical id. After creation, we will try to
    enumerate the /dev/link
    """

    def __init__(self,
                 type_or_name,
                 size_or_percent,
                 flags=None,
                 file_system=None,
                 mount_point=None,
                 fsck_option=0):
        """
        Constructor:

        size: a Size compatible value (see Size object documentation)..
        flags: list()
        name: the name of the partition, valid only on gpt partition tables
        """
        if isinstance(size_or_percent, PercentString):
            self.size = None
            self.percent_string = size_or_percent
        else:
            self.size = Size(size_or_percent)
            self.percent_string = None

        self.flags = flags or list()
        self.name = type_or_name
        self.file_system = file_system
        self.mount_point = mount_point

        self.partition_id = None
        self.devname = None
        self.allocated = False
        self.fsck_option = fsck_option

    @property
    def is_linked(self):
        return self.devname and self.partition_id

    def generate_fstab_entry(self, method='UUID'):
        if not self.file_system:
            return

        uuid = self.file_system.fs_uuid
        if not uuid:
            return

        label = self.file_system.fs_label

        if (method == 'LABEL') and not label:
            # To the label - J. Kelly, 2nd shift slogan
            log.debug('Missing label, can\'t take it there')

        options = self.file_system.generate_mount_options()

        dump = 0

        fsck_option = self.fsck_option

        gen = ''
        if method == 'UUID':
            gen += '# DEVNAME=%s\tLABEL=%s\nUUID=%s\t\t' % (self.devname,
                                                            label or '', uuid)
        elif method == 'LABEL' and label:
            gen += '# DEVNAME=%s\tUUID=%s\nLABEL=%s\t\t' % (self.devname, uuid,
                                                            label)
        else:
            gen += '# UUID=%s\tLABEL=%s\n%s\t\t' % (uuid, label or '',
                                                    self.devname)
        gen += '%s\t\t%s\t\t%s\t\t%s %s\n\n' % (self.mount_point or 'none',
                                                self.file_system, options, dump,
                                                fsck_option)

        return gen

    def __repr__(self):
        return "device: %s, size: %s, fs: %s, mount point: %s, fsck_option: %s" % \
               (
                   self.devname or 'unlinked',
                   self.size or self.percent_string,
                   self.file_system,
                   self.mount_point,
                   self.fsck_option
               )
