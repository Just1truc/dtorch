import os
import logging
import time
from collections import OrderedDict

from press import helpers
from press.helpers.cli import run
from press.helpers.parted import PartedInterface, NullDiskException, PartedException
from press.helpers.lvm import LVM
from press.helpers.mdadm import MDADM
from press.helpers.udev import UDevHelper
from press.layout.disk import Disk
from press.layout.lvm import VolumeGroup
from press.exceptions import (PhysicalDiskException, LayoutValidationError,
                              GeneralValidationException)

log = logging.getLogger(__name__)


class Layout(object):
    """The highest level class.
    """

    def __init__(self,
                 use_fibre_channel=False,
                 loop_only=False,
                 use_nvm_express=True,
                 parted_path='/sbin/parted',
                 clear_dm=False):
        """
        Docs, maybe later

        :param use_fibre_channel:
        :param loop_only:
        :param use_nvm_express:
        :param parted_path:
        :param clear_dm: Should apply clear the device mapper

        :ivar self.committed: False on __init__, True after calling apply()
        """

        self.committed = False
        self.fc_enabled = use_fibre_channel
        self.parted_path = parted_path
        self.clear_dm =clear_dm
        self.udev = UDevHelper()
        self.udisks = self.udev.discover_valid_storage_devices(
            fc_enabled=self.fc_enabled, loop_only=loop_only, nvme_enabled=use_nvm_express)

        if not self.udisks:
            raise PhysicalDiskException('There are no valid disks.')

        self.disks = OrderedDict()
        self.populate_disks()
        if not self.disks:
            raise PhysicalDiskException('There are no valid disks.')

        self.volume_groups = list()
        self.lvm = LVM()
        self.mdadm = None

        self.mount_handler = None

        self.software_raid_objects = []

    def populate_disks(self):
        for udisk in self.udisks:
            device = udisk.get('DEVNAME')
            try:
                parted = PartedInterface(device, self.parted_path)
            except NullDiskException as e:
                log.debug('NullDiskException for %s: %s' % (device, e))
                continue
            except PartedException as e:
                log.debug('PartedException for %s: %s' % (device, e))
                continue

            size = parted.get_size()
            disk = Disk(
                devname=device,
                devlinks=udisk.get('DEVLINKS'),
                devpath=udisk.get('DEVPATH'),
                size=size,
                sector_size=parted.sector_size.get('logical', 512))
            self.disks[device] = disk

    def find_device_by_ref(self, ref):
        """

        :param ref:
        :return:
        """
        for disk in self.disks.values():
            if ref == disk.devname:
                return disk
            if ref == disk.devpath:
                return disk
            for link in disk.devlinks:
                if ref == link:
                    return disk

    def find_device_by_size(self, size):
        """

        :param size:
        :return:
        """
        for disk in self.disks.values():
            if size < disk.size:
                return disk

    def _get_parted_interface_for_allocated_device(self, disk):
        """

        :param disk:
        :return: :raise LayoutValidationError:
        """
        if not disk.partition_table:
            raise LayoutValidationError('Disk is not allocated')
        return PartedInterface(
            device=disk.devname,
            parted_path=self.parted_path,
            partition_start=disk.partition_table.partition_start.bytes,
            alignment=disk.partition_table.alignment.bytes)

    @property
    def allocated(self):
        l = list()
        for disk in self.disks.values():
            if disk.partition_table:
                l.append(disk)
        return l

    @property
    def unallocated(self):
        l = list()
        for disk in self.disks.values():
            if not disk.partition_table:
                l.append(disk)
        return l

    def add_partition_table_from_model(self, partition_table):
        unallocated = self.unallocated
        if not unallocated:
            raise PhysicalDiskException('There are more available disks')

        if partition_table.disk == 'first':
            disk = unallocated[0]

        elif partition_table.disk == 'any':
            disk = self.find_device_by_size(partition_table.allocated_space)
            if not disk:
                raise PhysicalDiskException(
                    'There is no suitable disk, table is too big')

        else:
            disk = self.find_device_by_ref(partition_table.disk)
            if not disk:
                raise PhysicalDiskException(
                    'Could not associate disk, %s was not found' %
                    partition_table.disk)

        disk.new_partition_table(
            partition_table.type,
            partition_start=partition_table.partition_start,
            alignment=partition_table.alignment)

        for partition in partition_table.partitions:
            disk.partition_table.add_partition(partition)

    def find_partition_devname(self, disk, partition_id):
        # make this part of UDevHelper?
        partitions = self.udev.find_partitions(disk.devname)
        for partition in partitions:
            if int(partition.get('UDISKS_PARTITION_NUMBER',
                                 -1)) == partition_id:
                return partition.get('DEVNAME')

    def add_volume_group_from_model(self, model_vg):
        for pv in model_vg.physical_volumes:
            if not pv.reference.allocated:
                raise LayoutValidationError(
                    'Reference partition has not be allocated')
        real_vg = VolumeGroup(model_vg.name, model_vg.physical_volumes,
                              model_vg.pe_size)
        for lv in model_vg.logical_volumes:
            real_vg.add_logical_volume(lv)
        self.volume_groups.append(real_vg)

    def add_software_raid(self, raid_object):
        """
        :param raid_object:
        :return:
        """
        # instantiate mdadm object only once and only when we have to
        if not self.mdadm:
            self.mdadm = MDADM()

        raid_object.allocated = True
        raid_object.calculate_size()
        log.info('Adding RAID Volume %s, size: %s' % (raid_object.devname,
                                                      raid_object.size))
        self.software_raid_objects.append(raid_object)

    def apply_standard_partitions(self):
        log.info('Configuring standard partitions')
        for disk in self.allocated:
            parted = self._get_parted_interface_for_allocated_device(disk)

            #########################################################################################
            # Read into existing table and remove any active physical volumes
            # TODO: This needs a newer version of udev that is not currently available in Yolo
            # Critical Error, /lib64/libudev.so.0: undefined symbol: udev_enumerate_add_match_paren
            #########################################################################################
            # udev_partitions = self.udev.find_partitions(disk.devname)
            # for _udev_part in udev_partitions:
            #     _udev_devname = _udev_part['DEVNAME']
            #     if self.lvm.pv_exists(_udev_devname):
            #         self.lvm.pvremove(_udev_devname)
            #     # Also, zero the super block
            #     self.mdadm.zero_superblock(_udev_devname)
            #     self.mdadm.zero_4k(_udev_devname)
            #########################################################################################

            parted.remove_gpt()

            partition_table = disk.partition_table
            parted.set_label(partition_table.type)
            for partition in partition_table.partitions:
                fs_type = '' if not partition.file_system else \
                    partition.file_system.parted_fs_type_alias
                monitor = self.udev.get_monitor()
                monitor.start()
                log.debug(str(type(monitor)))
                partition_id = parted.create_partition(
                    partition.name,
                    partition.size.bytes,
                    flags=partition.flags,
                    fs_type=fs_type)
                # Dirty race condition hack, need to re-write udev monitor to make it more stable
                time.sleep(2)
                # end hack
                log.debug('Monitoring for devname')
                partition.devname = self.udev.monitor_partition_by_devname(
                    monitor, partition_id, action='add')
                log.debug('Found %s' % partition.devname)
                partition.partition_id = partition_id

                if not partition.devname:
                    log.error('%s %s %s' % (partition.devname, disk,
                                            partition_id))
                    raise PhysicalDiskException(
                        'Could not relate partition id to devname')

                if partition.file_system:
                    partition.file_system.create(partition.devname)

    def apply_software_raid(self):
        for raid in self.software_raid_objects:
            log.info('Building software RAID : {}'.format(raid))
            raid.create()
            # Not sure if this is necessary, but it seems to help
            # TODO: poll /proc/mdstat to determine when the array is ready
            time.sleep(5)
            if raid.file_system:
                raid.file_system.create(raid.devname)

    def destroy_volume_groups(self):
        for volume_group in self.volume_groups:
            if self.lvm.vg_exists(volume_group.name):
                log.info('Removing existing volume: %s' % volume_group.name)
                self.lvm.vgremove(volume_group.name)

    def destroy_physical_volumes(self):
        for volume_group in self.volume_groups:
            for pv in volume_group.physical_volumes:
                if not pv.reference:
                    continue
                if self.lvm.pv_exists(pv.reference.devname):
                    log.info('Removing existing physical volume: %s' %
                             pv.reference.devname)
                    self.lvm.pvremove(pv.reference.devname)

    def apply_lvm(self):
        for volume_group in self.volume_groups:
            devnames = list()
            monitor = self.udev.get_monitor()
            monitor.start()

            for pv in volume_group.physical_volumes:
                if not pv.reference.devname:
                    raise LayoutValidationError(
                        'devname is not populated, and it should be')
                devnames.append(pv.reference.devname)
                self.lvm.pvcreate(pv.reference.devname)
            self.lvm.vgcreate(volume_group.name, devnames,
                              volume_group.pe_size.bytes)

            for lv in volume_group.logical_volumes:
                self.lvm.lvcreate(lv.extents, volume_group.name, lv.name)
                log.debug(lv.name)
                log.debug('Monitoring for devname')
                device = self.udev.monitor_for_volume(monitor, lv.name)
                lv.devname = device['DEVNAME']
                log.debug('Found %s' % lv.devname)
                lv.devlinks = device.get('DEVLINKS', '').split()
                if lv.file_system:
                    lv.file_system.create(lv.devname)

    def clean_software_raid(self):
        """
        Attempt to detect and destroy
        :return:
        """
        log.info(
            '[paranoia] Removing existing mdraid and associated physical volumes'
        )
        for array in self.software_raid_objects:
            if array.pv_name and self.lvm.pv_exists(array.devname):
                self.lvm.pvremove(array.devname)
            array.clean()

    def clear_device_mapper(self):
        log.info('Clearing the device mapper')
        run('dmsetup remove_all')

    def apply(self):
        """Lots of logging here
        """

        # TODO: now that we have some clean up operations, determine if we still need to do this

        if self.clear_dm:
            self.clear_device_mapper()
        # Destroy any volume groups present at boot time
        self.destroy_volume_groups()
        self.clean_software_raid()

        self.apply_standard_partitions()

        # Now that we've built a partition table, destroy any resident data
        self.destroy_volume_groups()
        self.destroy_physical_volumes()

        # Now re-apply from scratch
        self.apply_software_raid()
        self.apply_lvm()

        # we've been sent to the loony bin
        self.committed = True

    def generate_fstab(self, method='UUID'):
        """
        This generates an fstab for this partition layout

        :param method: UUID, DEVNAME, or LABEL
        :return: (str) the generated fstab
        """

        supported_methods = ['DEVNAME', 'UUID', 'LABEL']

        if method not in supported_methods:
            raise GeneralValidationException('Method not supported')

        log.info('Generating %s partition table.' % method)
        header = '# Generated by Press v' + str(
            helpers.package.get_press_version())
        fstab = ''

        for disk in self.allocated:
            partition_table = disk.partition_table

            for partition in partition_table.partitions:
                entry = partition.generate_fstab_entry(method)
                if entry:
                    fstab += entry

        for vg in self.volume_groups:
            # forcing '/dev/mapper' DEVNAME format for LVM
            method = 'DEVNAME'
            for lv in vg.logical_volumes:
                entry = lv.generate_fstab_entry(method)
                if entry:
                    fstab += entry

        for swraid in self.software_raid_objects:
            entry = swraid.generate_fstab_entry(method)
            if entry:
                fstab += entry

        return header + '\n\n' + fstab

    @property
    def partitions(self):
        li = list()
        for disk in self.disks:
            partition_table = self.disks[disk].partition_table
            if partition_table:
                li += partition_table.partitions
        return li

    @property
    def logical_volumes(self):
        li = list()
        for volume_group in self.volume_groups:
            li += volume_group.logical_volumes
        return li

    @property
    def devname_index(self):
        index = dict()
        for partition in self.partitions:
            if partition.devname:
                index[partition.devname] = partition
        for volume in self.logical_volumes:
            if volume.devlinks:
                index[volume.devlinks[-1]] = volume
        return index

    @property
    def mount_point_index(self):
        index = dict()
        for partition in self.partitions:
            if partition.mount_point and partition.mount_point != 'swap':
                index[partition.mount_point] = partition
        for volume in self.logical_volumes:
            if volume.mount_point and volume.mount_point != 'swap':
                index[volume.mount_point] = volume
        return index


class MountHandler(object):
    """
    mount_points is an OrderedDict
    """

    def __init__(self, target, layout):
        self.target = target
        self.mount_points = OrderedDict()
        if not layout.committed:
            raise GeneralValidationException('Layout has not been applied')
        mount_point_index = layout.mount_point_index
        mp_list = list(mount_point_index.keys())
        idx = mp_list.index('/')
        if idx == -1:
            raise GeneralValidationException('root mount point is missing')
        root_device = mount_point_index.get('/').devname
        if not root_device:
            raise GeneralValidationException(
                'root partition is not linked to a physical device')

        self.mount_points[mp_list.pop(idx)] = dict(
            mount_point='/', level=1, mounted=False, device=root_device)
        mp_list.sort(key=lambda s: s.count('/'))
        for mp in mp_list:
            device = mount_point_index.get(mp).devname
            if not device:
                raise GeneralValidationException(
                    '%s is missing physical device' % mp)
            self.mount_points[mp] = dict(
                mount_point=mp,
                level=mp.count('/'),
                mounted=False,
                device=device)

    def join(self, path):
        return os.path.join(self.target, path.lstrip('/'))

    def mount(self, path, device='none', bind=False, mount_type=''):
        full_path = self.join(path)
        command = 'mount %s%s%s %s' % (
            bind and '--bind ' or '',
            mount_type and '-t %s ' % mount_type or '', device, full_path)
        run(command, raise_exception=True)
        self.mount_points[path]['mounted'] = True
        log.info('Mounted %s' % full_path)

    def mount_proc(self):
        self.create_directory(self.join('/proc'))
        self.mount_points['/proc'] = dict(
            mount_point='/proc', level=1, mounted=False, device=None)
        self.mount('/proc', mount_type='proc')

    def mount_sys(self):
        # mounting sys may not be needed
        self.create_directory(self.join('/sys'))
        self.mount_points['/sys'] = dict(
            mount_point='/sys', level=1, mounted=False, device=None)
        self.mount('/sys', mount_type='sysfs')

    def bind_dev(self):
        self.create_directory(self.join('/dev'))
        self.mount_points['/dev'] = dict(
            mount_point='/dev', level=1, mounted=False, device=None)
        self.mount('/dev', '/dev', bind=True)

    def umount(self, path):
        full_path = self.join(path)
        command = 'umount %s' % full_path
        run(command, raise_exception=True)
        self.mount_points[path]['mounted'] = True
        log.info('Unmounted %s' % full_path)

    @property
    def levels(self):
        return set([self.mount_points[d]['level'] for d in self.mount_points])

    def get_level(self, level):
        return [
            self.mount_points[d]
            for d in self.mount_points
            if self.mount_points[d]['level'] == level
        ]

    @staticmethod
    def create_directory(full_path):
        if helpers.deployment.recursive_makedir(full_path):
            log.info('Created directory %s' % full_path)

    def mount_physical(self):
        log.info('Mounting partitions and volumes')
        for level in self.levels:
            for mp in self.get_level(level):
                self.create_directory(self.join(mp['mount_point']))
                self.mount(mp['mount_point'], mp['device'])
                self.mount_points[mp['mount_point']]['mounted'] = True

    def mount_pseudo(self):
        log.info('Mounting pseudo file systems')
        self.mount_proc()
        self.mount_sys()
        self.bind_dev()

    def teardown(self):
        mount_points = \
            OrderedDict(reversed(sorted(iter(self.mount_points.items()), key=lambda d: d[1]['level'])))
        log.info('Unmounting everything')
        for mp in mount_points:
            if mount_points[mp]['mounted']:
                self.umount(mp)
