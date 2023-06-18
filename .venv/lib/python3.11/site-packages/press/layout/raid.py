import logging
from press.exceptions import PressCriticalException
from press.helpers.mdadm import MDADM
from size import Size

log = logging.getLogger(__name__)


class SoftwareRAID(object):
    raid_type = ''


class MDRaid(SoftwareRAID):
    raid_type = 'mdadm'

    def __init__(self,
                 devname,
                 level,
                 members,
                 spare_members=None,
                 size=0,
                 file_system=None,
                 mount_point=None,
                 fsck_option=0,
                 pv_name=None):
        """
        Logical representation of a mdadm controlled RAID
        :param devname: /dev/mdX
        :param level: 0, 1
        :param members: Partition objects that represent member disks
        :param spare_members: (optional) Partition objects that represent spare disks
        :param file_system: (optional) A file system object, if that is your intent
        :param mount_point: (optional) where to mount, needs file system
        :param pv_name: (optional) Am I a pv? if so, file_system and mount_point are ignored
        """

        self.devname = devname
        self.level = level
        self.members = members
        self.spare_members = spare_members or []
        self.file_system = file_system
        self.mount_point = mount_point
        self.pv_name = pv_name
        self.size = Size(0)
        self.allocated = False
        self.mdadm = MDADM()
        self.fsck_option = fsck_option

    @staticmethod
    def _get_partition_devnames(members):
        disks = []

        for partition in members:
            if not partition.allocated:
                raise PressCriticalException(
                    'Attempted to use unlinked partition')
            disks.append(partition.devname)

        return disks

    def calculate_size(self):
        for member in self.members:
            if not member.allocated:
                raise PressCriticalException(
                    'Member is not allocated, cannot calculate size')

        if self.level == 0:
            accumulated_size = Size(0)
            for member in self.members:
                accumulated_size += member.size
            self.size = accumulated_size

        if self.level == 1:
            self.size = self.members[0].size

    def create(self):
        """
        """

        member_partitions = self._get_partition_devnames(self.members)
        spare_partitions = self._get_partition_devnames(self.spare_members)

        log.info('Building Software RAID %s' % self.devname)
        self.mdadm.create(
            self.devname,
            self.level,
            member_partitions,
            spare_partitions,
        )

    def stop(self):
        self.mdadm.stop(self.devname)

    def remove(self):
        self.mdadm.remove(self.devname)

    @property
    def active(self):
        return self.mdadm.is_present(self.devname)

    def zero_members(self):
        for member in self.members + self.spare_members:
            if not member.devname:
                continue
            log.info('Cleaning super block: %s' % member.devname)
            self.mdadm.zero_superblock(member.devname)

    def clean(self):
        if self.active:
            log.info('Cleaning %s' % self.devname)
            active_members = self.mdadm.get_members(self.devname)
            log.debug('Found active members: %s' % active_members)
            self.stop()
            # for member in active_members:
            #     self.mdadm.fail_remove_member(self.devname, member)
            self.remove()
            import time
            time.sleep(2)
            for member in active_members:
                self.mdadm.zero_superblock(member)
                self.mdadm.zero_4k(member)

    def generate_fstab_entry(self, method='UUID'):
        # TODO: Move this shit out of the classes an into a library...
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
        return '%s : %s' % (self.devname, self.members)
