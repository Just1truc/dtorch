"""
From the pyudev docs:

    Once added, a filter cannot be removed anymore. Create a new object instead.

pyudev can be kind of silly, but I certainly don't feel like wrapping my own.
"""
import logging
import os
import pyudev

log = logging.getLogger(__name__)


class UDevHelper(object):

    def __init__(self):
        self.context = pyudev.Context()

    def get_monitor(self):
        """
        Because filters cannot be removed
        :return: fresh pyudev.Monitor
        """
        return pyudev.Monitor.from_netlink(self.context)

    def get_partitions(self):
        return self.context.list_devices(subsystem='block', DEVTYPE='partition')

    def get_disks(self):
        return self.context.list_devices(subsystem='block', DEVTYPE='disk')

    def find_partitions(self, device):
        """
        matches partitions belonging to a device.
        """

        devices = self.get_partitions()
        return devices.match_parent(
            pyudev.Device.from_device_file(self.context, device))

    def get_device_by_name(self, devname):
        try:
            udisk = pyudev.Device.from_device_file(self.context, devname)
        except OSError:
            return None
        return udisk

    def discover_valid_storage_devices(self, fc_enabled=False, nvme_enabled=True, loop_only=False):
        """
        Kind of ugly, but gets the job done. It strips devices we don't
        care about, such as cd roms, device mapper block devices, loop, and fibre channel.

        """
        invalid_id_type = ['cd', 'usb']
        invalid_major = ['253', '254', '1']  # 253/254 are LVM/DM, 1 is ramdisk
        nvme_major = '259'
        loop_major = '7'

        disks = self.get_disks()

        pruned = list()
        fc_devices = list()
        nvme_devices = list()
        loop_devices = list()

        for disk in disks:
            if '-fc-' in disk.get('ID_PATH', ''):
                fc_devices.append(disk)
            elif disk.get('MAJOR') == nvme_major:
                nvme_devices.append(disk)
            elif disk.get('MAJOR') == loop_major:
                loop_devices.append(disk)
            elif (disk.get('ID_TYPE') in invalid_id_type or
                  disk.get('MAJOR') in invalid_major):
                continue
            else:
                pruned.append(disk)

        if loop_only:
            return loop_devices

        if nvme_enabled:
            pruned += nvme_devices

        if fc_enabled:
            pruned += fc_devices

        return pruned

    def yield_mapped_devices(self):
        disks = self.get_disks()
        for disk in disks:
            if disk.get('MAJOR') == '254':  # Device Mapper (LVM)
                yield disk

    @staticmethod
    def monitor_partition_by_devname(monitor, partition_id, action=None):
        monitor.filter_by('block', device_type="partition")
        for _, device in monitor:
            log.debug('Seen: %s' % list(device.items()))
            if action and device.get('ACTION') != action:
                log.debug('Action, %s, does not match %s' %
                          (action, device.get('ACTION')))
                continue
            elif device.get('UDISKS_PARTITION_NUMBER') == str(partition_id):
                return str(device['DEVNAME'])
            elif not device.get('UDISKS_PARTITION_NUMBER') and device.get(
                    'ID_PART_ENTRY_NUMBER') == str(partition_id):
                return str(device['DEVNAME'])

    def get_network_devices(self):
        """ Returns a list of all network(ethernet/type 1] devices found on the system. """

        result = []

        for candidate in self.context.list_devices(subsystem='net'):
            if 'type' in candidate.attributes:
                if candidate.attributes.asint('type') == 1:
                    result.append(candidate)

        # let's go ahead and return it sorted..
        result.sort(key=lambda dev: dev.sys_name)
        return result

    @staticmethod
    def monitor_for_volume(monitor, lv_name):
        monitor.filter_by('block')
        for action, device in monitor:
            if device.get('DM_LV_NAME') == lv_name:
                return device
