import os
import logging

from press.exceptions import LVMValidationError
from size import Size, PercentString

log = logging.getLogger(__name__)


class PhysicalVolume(object):
    """
    :param reference: This is typically a partition object, but could also
     be a disk object in the future, right now; whole device PV's are not supported.
     Once the reference has been applied, refernece.devname should be set, allowing
     the apply function to do it's work. From the configuration file, these will be
     linked via list index.
    """

    def __init__(self, reference):
        self.reference = reference


class VolumeGroup(object):
    """
    """

    def __init__(self, name, physical_volumes, pe_size=4194304):
        self.logical_volumes = list()

        self.name = name
        self.physical_volumes = physical_volumes
        self.pv_raw_size = Size(
            sum([pv.reference.size.bytes for pv in self.physical_volumes]))
        self.pe_size = Size(pe_size)
        self.extents = self.pv_raw_size.bytes / self.pe_size.bytes
        self.size = Size(self.pe_size.bytes * self.extents)

    @property
    def current_usage(self):
        used = Size(0)
        if not self.logical_volumes:
            return used
        for volume in self.logical_volumes:
            used += volume.size
        return used

    @property
    def current_pe(self):
        return self.current_usage.bytes / self.pe_size.bytes

    @property
    def free_space(self):
        return self.size - self.current_usage

    @property
    def free_pe(self):
        return self.free_space.bytes / self.pe_size.bytes

    def convert_percent_to_size(self, percent, free):
        if free:
            return self.get_percentage_of_free_space(percent)
        return self.get_percentage_of_usable_space(percent)

    def _validate_volume(self, volume):
        log.info('Validating volume {}'.format(volume.name))
        if not isinstance(volume, LogicalVolume):
            return ValueError('Expected LogicalVolume instance')
        if self.free_space < volume.size:
            adjustment = Size(volume.size.bytes - self.free_space.bytes)
            raise LVMValidationError(
                "There is not enough space for volume "
                "'{}' (avail: {}, requested: {}).  "
                "Please adjust the size approximately by: {}".format(
                    volume.name, self.free_space.bytes, volume.size.bytes,
                    adjustment))

    def add_logical_volume(self, volume):
        if volume.percent_string:
            volume.size = self.convert_percent_to_size(
                volume.percent_string.value, volume.percent_string.free)
        self._validate_volume(volume)
        extents = int(volume.size.bytes / self.pe_size.bytes)
        unused = volume.size % self.pe_size
        log.info('Adding logical volume <%s>: %d / %d LE, unusable: %s' %
                 (volume.name, volume.size.bytes, extents, unused))
        allocated_pe = self.current_pe + extents
        log.debug('allocated: %d , total: %d' % (allocated_pe, self.extents))
        if allocated_pe == self.extents:
            # Shrink extents by 1 to avoid overrun
            log.info('Shrinking volume by 1 extent')
            extents -= 1
        volume.extents = extents
        self.logical_volumes.append(volume)

    def add_logical_volumes(self, volumes):
        for volume in volumes:
            self.add_logical_volume(volume)

    def get_percentage_of_free_space(self, percent):
        return Size(self.free_space.bytes * percent)

    def get_percentage_of_usable_space(self, percent):
        return Size(self.size.bytes * percent)

    def __repr__(self):
        out = 'VG: %s\nPV(s): %s\nPE/LE: %d\nPE/LE Size: %s' \
              '\nSize: %s (Unusable: %s)\nUsed: %s / %d\nAvailable: %s / %d' \
              '\nLV(s): %s' % (
                  self.name,
                  str([pv.reference.devname for pv in self.physical_volumes]),
                  self.extents,
                  self.pe_size,
                  self.size,
                  self.pv_raw_size % self.pe_size,
                  self.current_usage,
                  self.current_pe,
                  self.free_space,
                  self.free_pe,
                  [(lv.name, str(lv.size)) for lv in self.logical_volumes]
              )
        return out


class LogicalVolume(object):
    """
    Very similar to Partition, device is the /dev/link after the device is created.
    """

    def __init__(self,
                 name,
                 size_or_percent,
                 file_system=None,
                 mount_point=None,
                 fsck_option=0):
        self.name = name
        if isinstance(size_or_percent, PercentString):
            self.size = None
            self.percent_string = size_or_percent
        else:
            self.size = Size(size_or_percent)
            self.percent_string = None

        self.file_system = file_system
        self.mount_point = mount_point
        self.fsck_option = fsck_option

        # extents are calculated and stored by the VolumeGroup.add_logical_volume() method
        self.extents = None

        self.devname = None
        self.devlinks = None

    @property
    def devlink(self):
        if not self.devlinks:
            return self.devname
        for link in self.devlinks:
            if os.path.split(link)[0] == '/dev/mapper':
                return link
        return self.devlinks[0]

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
            gen += '# DEVNAME=%s\tLABEL=%s\nUUID=%s\t\t' % (self.devlink,
                                                            label or '', uuid)
        elif method == 'LABEL' and label:
            gen += '# DEVNAME=%s\tUUID=%s\nLABEL=%s\t\t' % (self.devlink, uuid,
                                                            label)
        else:
            gen += '# UUID=%s\tLABEL=%s\n%s\t\t' % (uuid, label or '',
                                                    self.devlink)
        gen += '%s\t\t%s\t\t%s\t\t%s %s\n\n' % (self.mount_point or 'none',
                                                self.file_system, options, dump,
                                                fsck_option)

        return gen

    def __repr__(self):
        return "device: %s, name: %s, size: %s, fs: %s, mount point: %s, fsck_option: %s" % (
            self.devlinks and self.devlinks[-1] or 'unlinked', self.name,
            self.size or self.percent_string, self.file_system,
            self.mount_point, self.fsck_option)
