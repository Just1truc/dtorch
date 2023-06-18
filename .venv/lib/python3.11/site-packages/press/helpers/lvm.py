"""
Original Author: Jeff Ness
"""

import logging

from press.helpers.cli import run

log = logging.getLogger(__name__)


class LVMError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class LVM(object):
    """
    Class to interacting with LVM in Python.
    """

    pvdisplay_command = 'pvdisplay -C --separator : --unit b -o +pvseg_size'
    vgdisplay_command = 'vgdisplay -C --separator : --unit b -o +vg_all'

    def __init__(self):
        """
        LVM Class constructor.
        """
        # List of required system binaries
        self.binaries = [
            'pvcreate', 'pvremove', 'pvdisplay', 'vgcreate', 'vgremove',
            'vgdisplay', 'lvcreate', 'lvremove', 'lvdisplay', 'vgchange', 'pvs',
            'vgs'
        ]

    @staticmethod
    def __execute(command, ignore_errors=False, quiet=False):
        """
        Execute a shell command using subprocess, then validate the return
        code and log either stdout or stderror to the logger.
        """
        log.debug('Running: %s' % command)
        res = run(command, ignore_error=ignore_errors, quiet=quiet)
        if res.returncode and not ignore_errors:
            log.error('stdout: %s, stderr: %s' % (res, res.stderr))
            raise LVMError('LVM command returned status {}: {}, {}'.format(
                res.returncode, res, res.stderr))
        return res

    @staticmethod
    def __to_dict(stdout):
        """
        Takes output from lvm commands and converts to dict
        """
        stdout = stdout.splitlines()
        stdout = [_f for _f in stdout if _f]
        headers = stdout.pop(0)
        headers = headers.strip().split(':')

        response = []
        for line in stdout:
            values = line.strip().split(':')
            response.append(dict(zip(headers, values)))

        return response

    @staticmethod
    def bytestring(bytes):
        return '%dB' % bytes

    def pvcreate(self, physical_volume):
        """
        Create a physical volume using pvcreate command line tool.
        """
        log.info('Creating physical volume: %s' % physical_volume)
        command = 'pvcreate --force %s' % physical_volume
        return self.__execute(command)

    def pvremove(self, physical_volume):
        """
        Delete a physical volume using pvcreate command line tool.
        """
        log.info('running pvremove')
        command = 'pvremove --force --force --yes %s' % physical_volume
        return self.__execute(command)

    def pvdisplay(self, physical_volume=''):
        """
        Display a physical volume using pvdisplay command line tool.
        """
        log.info('running pvdisplay')
        command = '%s %s' % (self.pvdisplay_command, physical_volume)
        res = self.__execute(command)
        return self.__to_dict(res)

    def pv_exists(self, physical_volume):
        command = '%s %s' % (self.pvdisplay_command, physical_volume)
        if self.__execute(command, ignore_errors=True, quiet=True).returncode:
            return False
        return True

    def vgcreate(self, vg_name, physical_volumes, pe_size=4194304):
        """
        Create a volume group using vgcreate command line tool.

        physical_volumes is a list containing at least one physical_volume,
        or a single physical_volume.
        """
        # physical_volumeS should always be a list becaouse it is plural
        # Justification: 'Explicit is better than implicit' PEP 20
        pe_size = self.bytestring(pe_size)
        log.info('Creating VG: %s, PV: %s, PE/LE Size: %s' %
                 (vg_name, physical_volumes, pe_size))
        physical_volumes = ' '.join(physical_volumes)
        command = 'vgcreate --physicalextentsize %s %s %s' % (pe_size, vg_name,
                                                              physical_volumes)
        return self.__execute(command)

    def vgremove(self, group_label):
        """
        Delete a volume group by label using vgremove command line tool.
        """
        log.info('running vgremove')
        command = 'vgremove --force %s' % group_label
        return self.__execute(command)

    def vgdisplay(self, group_label=''):
        """
        Display a volume group using vgdisplay command line tool.
        """
        log.info('running vgdisplay')
        command = 'vgdisplay -C --separator : --unit b %s' % group_label
        res = self.__execute(command)
        return self.__to_dict(res)

    def vg_exists(self, vg_name):
        command = 'vgdisplay -C --separator : --unit b %s' % vg_name
        if self.__execute(command, ignore_errors=True, quiet=True).returncode:
            return False
        return True

    def lvcreate(self, extents, vg_name, lv_name):
        """
        Create a logical volume using lvcreate command line tool.
        """
        log.info('Creating Volume Group: %s, Extents: %s, VG: %s' %
                 (lv_name, extents, vg_name))
        create_command = 'lvcreate --yes --extents %s -n %s %s' % (extents,
                                                                   lv_name,
                                                                   vg_name)
        self.__execute(create_command)

    def lvdisplay(self, combined_label=''):
        """
        Display a logical volume using lvdisplay command line tool.
        """
        log.info('running lvdisplay')
        command = 'lvdisplay -C --separator : --unit b %s' % combined_label
        res = self.__execute(command)
        return self.__to_dict(res)

    def lvremove(self, combined_label):
        """
        Deletes a logical volume using lvremove command line tool.get_logger
        """
        log.info('running lvremove')
        command = 'lvremove -f %s' % combined_label
        return self.__execute(command)

    def vgchange(self, args):
        command = 'vgchange %s' % args
        return self.__execute(command)

    def activate_volume(self, volume_group):
        return self.vgchange('-a y %s' % volume_group)

    @staticmethod
    def get_volume_groups():
        """
        :return: A list of volume group names
        """
        command = 'vgs --noheadings --rows -o vg_name'
        out = run(command, raise_exception=False)
        if out.returncode:
            return list()
        if not out.stdout:
            return list()
        return [vg.strip() for vg in out.splitlines()]

    @staticmethod
    def get_physical_volumes():
        """

        :return: A list of physical volume names
        """
        command = 'pvs --noheadings --rows -o pv_name'
        out = run(command, raise_exception=False)
        if out.returncode:
            return list()
        if not out.stdout:
            return list()
        return [pv.strip() for pv in out.splitlines()]
