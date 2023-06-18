import logging

from size import Size

log = logging.getLogger(__name__)


class PartitionTableModel(object):
    """
    Mimics PartitionTable structured class but does not contain size attribute.
    Used to stage a partition table prior to knowing physical geometry
    """

    def __init__(self,
                 table_type,
                 disk='first',
                 partition_start=1048576,
                 alignment=1048576):
        """
        :param table_type: (str) gpt or msdos
        :param disk: (str) first, any, devname (/dev/sda), devlink (/dev/disk/by-id/foobar),
        or devpath (/sys/devices/pci0000:00/0000:00:1f.2/ata1/host0/target0:0:0/0:0:0:0/block/sda)
            first: The first available disk, regardless of size will be used
            any: Any disk that can accommodate the static allocation of the partitions
        """
        log.debug('Modeling new Partition Table Model: Type: %s , Disk: %s' %
                  (table_type, disk))
        self.partitions = list()
        self.disk = disk
        self.partition_start = partition_start
        self.alignment = alignment

        valid_types = ['gpt', 'msdos']
        if table_type not in valid_types:
            raise ValueError('table not supported: %s' % table_type)

        self.type = table_type

    def add_partition(self, partition):
        """
        :param partition: (Partition) should be compatible with a structure.disk.Partition object
        """
        log.debug(
            'Modeling new partition: %s : %s, size: %s / %d, fs: %s, mount_point: %s'
            % (self.type == 'gpt' and 'name' or 'type', partition.name,
               partition.size or partition.percent_string,
               partition.size and partition.size.bytes or 0,
               partition.file_system, partition.mount_point))
        self.partitions.append(partition)

    def add_partitions(self, partitions):
        """Adds partitions from iterable
        """
        for partition in partitions:

            self.add_partition(partition)

    @property
    def allocated_space(self):
        """Return Size object of the sum of the size for statically
        allocated (not percent) partitions. I am not proud of this sentence.
        """
        size = Size(0)

        if not self.partitions:
            return size

        for part in self.partitions:
            if part.percent_string:
                continue
            size += part.size

        return size
