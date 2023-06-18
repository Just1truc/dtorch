import logging

from press.helpers import sysfs_info

from press.layout import (
    Layout,
    Partition,
)
from press.exceptions import (GeneratorError)
from press.layout.lvm import (PhysicalVolume, LogicalVolume)
from press.layout.filesystems.extended import (EXT2, EXT3, EXT4)
from press.layout.filesystems.swap import (SWAP)
from press.layout.filesystems.xfs import XFS
from press.layout.filesystems.fat import FAT32, EFI
from press.layout.filesystems.ntfs import NTFS
from press.layout.raid import MDRaid
from press.models.lvm import VolumeGroupModel
from press.models.partition import PartitionTableModel

# noinspection PyUnresolvedReferences
from size import PercentString, Size

from six import string_types

LOG = logging.getLogger(__name__)

MBR_LOGICAL_MAX = 128
PARTED_PATH = 'parted'

__pv_linker__ = dict()
__partition_linker__ = dict()

fs_selector = dict(
    ext2=EXT2, ext3=EXT3, ext4=EXT4, swap=SWAP, xfs=XFS, efi=EFI, fat32=FAT32, ntfs=NTFS)

default_use_fibre_channel = False
default_loop_only = False
default_clear_device_mapper = True


def has_logical(partitions):
    for partition in partitions:
        if partition.get('mbr_type') == 'logical':
            return True

    if len(partitions) > 4:
        return True

    return False


def get_max_primary(partitions):
    if has_logical(partitions):
        return 3
    return 4


def fsck_pass(fs_object, lv_or_part, mount_point):
    if 'fsck_option' not in lv_or_part:
        if fs_object.require_fsck and mount_point:
            # root should fsck on pass 1
            if mount_point == '/':
                fsck_option = 1
            # Everything else
            else:
                fsck_option = 2
        else:
            fsck_option = 0
    else:
        # Explicitly defined in configuration
        fsck_option = lv_or_part['fsck_option']

    return fsck_option


def generate_size(size):
    if isinstance(size, string_types):
        if '%' in size:
            return PercentString(size)
    return size


def generate_file_system(fs_dict):
    fs_type = fs_dict.get('type', 'undefined')

    fs_class = fs_selector.get(fs_type)
    if not fs_class:
        raise GeneratorError('%s type is not supported!' % fs_type)

    fs_object = fs_class(**fs_dict)

    return fs_object


def generate_partition(type_or_name, partition_dict):
    fs_dict = partition_dict.get('file_system')

    if fs_dict:
        fs_object = generate_file_system(fs_dict)
        LOG.debug('Adding %s file system' % fs_object)
    else:
        fs_object = None

    mount_point = partition_dict.get('mount_point')

    if fs_object:
        fsck_option = fsck_pass(fs_object, partition_dict, mount_point)
    else:
        fsck_option = False

    p = Partition(
        type_or_name=type_or_name,
        size_or_percent=generate_size(partition_dict['size']),
        flags=partition_dict.get('flags', []),
        file_system=fs_object,
        mount_point=mount_point,
        fsck_option=fsck_option)

    if 'lvm' in p.flags:
        # We need to preserve this mapping for generating volume groups
        __pv_linker__[partition_dict['name']] = p

    # For software RAID, we need an index to rendered partition objects
    __partition_linker__[partition_dict['name']] = p

    return p


def _generate_mbr_partitions(partition_dicts):
    """
    There can be four primary partitions unless a logical partition is
    explicitly defined, in such cases, there can be only three
    primary partitions.

    If there are more than four partitions defined and neither
    primary or logical is explicitly defined, then there will be three primary,
    one extended, and up to 128 logical partitions.
    """
    max_primary = get_max_primary(partition_dicts)

    primary_count = 0
    logical_count = 0

    partitions = list()
    max_err = 'Maximum logical partitions have been exceeded'

    for partition in partition_dicts:
        explicit_name = partition.get('mbr_type')
        if explicit_name:
            if explicit_name == 'primary':
                if primary_count >= max_primary:
                    raise GeneratorError(max_err)
                primary_count += 1
            elif explicit_name == 'logical':
                if logical_count > MBR_LOGICAL_MAX:
                    raise GeneratorError(max_err)
                logical_count += 1

            partition_type = explicit_name
        else:
            if primary_count < max_primary:
                partition_type = 'primary'
                primary_count += 1
            else:
                if logical_count > MBR_LOGICAL_MAX:
                    raise GeneratorError(max_err)
                partition_type = 'logical'
                logical_count += 1
        partitions.append(generate_partition(partition_type, partition))
    return partitions


def _generate_gpt_partitions(partition_dicts):
    """
    Use the name field in the configuration or p + count
    :param partition_dicts:
    :return: list of Partition objects
    """
    count = 0
    partitions = list()
    for partition in partition_dicts:
        partitions.append(
            generate_partition(
                partition.get('name', 'p' + str(count)), partition))
        count += 1
    return partitions


def generate_partitions(table_type, partition_dicts):
    if table_type == 'msdos':
        return _generate_mbr_partitions(partition_dicts)
    if table_type == 'gpt':
        return _generate_gpt_partitions(partition_dicts)
    else:
        raise GeneratorError('Table type is invalid: %s' % table_type)


def generate_partition_table_model(partition_table_dict,
                                   default_partition_start, default_alignment):
    """
    Generate a PartitionTableModel, a PartitionTable without a disk association

    Note: if label is not specified, the Press object will determine label based
     off:
        1. Device booted in UEFI mode
        2. A disk is larger then 2.2TiB

    Most people don't care if their partitions are primary/logical
    or what their gpt names are, so we'll care for them.

    :param partition_table_dict:
    :param default_alignment:
    :param default_partition_start:

    :return: PartitionTableModel
    """
    table_type = partition_table_dict['label']

    pm = PartitionTableModel(
        table_type=table_type,
        disk=partition_table_dict['disk'],
        partition_start=partition_table_dict.get('partition_start',
                                                 default_partition_start),
        alignment=partition_table_dict.get('alignment', default_alignment))

    partition_dicts = partition_table_dict.get('partitions')
    if partition_dicts:
        partitions = generate_partitions(table_type, partition_dicts)
        pm.add_partitions(partitions)
    return pm


def generate_volume_group_models(volume_group_dict, default_pe_size):
    """
    We use __pv_linker__ to reference partition objects by name
    :param default_pe_size:
    :param volume_group_dict:
    :return:
    """
    if not __pv_linker__:
        raise GeneratorError(
            '__pv_linker__ is null, have you flagged any partitions with LVM?')
    vgs = list()
    for vg in volume_group_dict:
        pvs = list()
        if not vg.get('physical_volumes'):
            raise GeneratorError('No physical volumes are defined')
        for pv in vg.get('physical_volumes'):
            ref = __pv_linker__.get(pv)
            if not ref:
                raise GeneratorError('invalid ref: %s' % pv)
            pvs.append(PhysicalVolume(ref))
        vgm = VolumeGroupModel(vg['name'], pvs, pe_size=default_pe_size)
        lv_dicts = vg.get('logical_volumes')
        lv_dicts.sort(key=lambda s: (s.get('mount_point', '')).count('/'))
        lvs = list()
        if lv_dicts:
            for lv in lv_dicts:
                if lv.get('file_system'):
                    fs = generate_file_system(lv.get('file_system'))
                else:
                    fs = None
                mount_point = lv.get('mount_point')

                fsck_option = fsck_pass(fs, lv, mount_point)
                lvs.append(
                    LogicalVolume(
                        name=lv['name'],
                        size_or_percent=generate_size(lv['size']),
                        file_system=fs,
                        mount_point=mount_point,
                        fsck_option=fsck_option))
            vgm.add_logical_volumes(lvs)
        vgs.append(vgm)
    return vgs


def generate_layout_stub(layout_config, parted_path):
    LOG.debug('Using parted at: %s' % parted_path)
    return Layout(
        use_fibre_channel=layout_config.get('use_fibre_channel',
                                            default_use_fibre_channel),
        loop_only=layout_config.get('loop_only', default_loop_only),
        parted_path=parted_path,
        clear_dm=layout_config.get('clear_device_mapper', default_clear_device_mapper)
    )


def add_bios_boot_partition(partition_table):
    """ Insert a bios boot partition if one is not already defined"""
    partitions = partition_table['partitions']
    if partitions:
        if 'bios_grub' not in partitions[0].get('flags', list()):
            LOG.info('Automatically inserting a BIOS boot partition')
            bios_boot_partition = dict(
                name='BIOS boot partition', size='1MiB', flags=['bios_grub'])
            partitions.insert(0, bios_boot_partition)
        else:
            LOG.info('BIOS boot partition seems to already be present, kudos!')
    partition_table['partitions'] = partitions


def add_efi_boot_partition(partition_table):
    """ Insert an EFI boot partition if one is not already defined"""
    LOG.debug("Adding EFI boot partition")
    for partition in partition_table.get('partitions', ()):
        if partition.get('file_system', {}).get('type') == 'efi':
            LOG.info('EFI boot partition is already present, kudos!')
            break
    else:
        efi_partition = dict(
            name='EFI',
            size='253MiB',
            flags=['boot'],
            file_system=dict(type="efi"),
            mount_point="/boot/efi")
        partition_table.setdefault('partitions', []).insert(0, efi_partition)


def set_disk_labels(layout, layout_config):
    """
    Read into configuration and set label to gpt or msdos based on size.
    If label is present in the configuration and is gpt but not efi,
    make sure bios boot partition is present.
    """
    # TODO: Trace disk generator and inject this
    partition_tables = layout_config.get('partition_tables')
    for partition_table in partition_tables:
        label = partition_table.get('label')
        if label:
            LOG.info('Table: %s is set as %s in configuration' %
                     (partition_table.get('disk', 'undefined'),
                      partition_table['label']))

        # 'first' and 'any' are valid disk references in the configuration
        # 'first' indicates the first unallocated disk
        #       (as sorted by udev (subsystem->sub_id)
        # 'any' references that first disk encountered
        #       that is large enough to hold the partitions
        # 'any' is slated for removal in v0.4.0 roadmap

        if partition_table['disk'] == 'first':
            disk = layout.unallocated[0]
        elif partition_table['disk'] == 'any':
            size = Size(0)
            for partition in partition_table.get('partitions'):
                # Percent strings are relative and
                # cannot be used to calculate total size
                if '%' not in partition['size']:
                    size += Size(partition['size'])
            disk = layout.find_device_by_size(size)
        else:
            disk = layout.find_device_by_ref(partition_table['disk'])

        if not sysfs_info.has_efi():
            if disk.size.over_2t:
                LOG.info('%s is over 2.2TiB, using gpt' % disk.devname)
                label = 'gpt'
                if not layout_config.get('no_bios_boot_partition'):
                    # TODO: Add config option to disks,
                    # allowing user to specify the boot disk
                    # if disk == the first disk or presumed boot disk
                    if list(layout.disks.keys()).index(disk.devname) == 0:
                        add_bios_boot_partition(partition_table)
            elif label == 'gpt':
                add_bios_boot_partition(partition_table)
            else:
                LOG.info('%s is under 2.2TiB, using msdos' % disk.devname)
                label = 'msdos'
        else:
            LOG.info('Booting in UEFI mode, using gpt')
            label = 'gpt'
            # Only install boot partition on "first" drive
            # TODO: Allow the user to specifygit
            if list(layout.disks.keys()).index(disk.devname) == 0:
                add_efi_boot_partition(partition_table)

        partition_table['label'] = label


def generate_software_raid(raid_config):
    raid_objects = []

    for raid in raid_config:
        fs_dict = raid.get('file_system')

        if fs_dict:
            fs_object = generate_file_system(fs_dict)
            LOG.debug('Adding %s file system' % fs_object)
        else:
            fs_object = None

        mount_point = raid.get('mount_point')

        if fs_object:
            fsck_option = fsck_pass(fs_object, raid, mount_point)
        else:
            fsck_option = False

        partitions = list()
        for part_name in raid['partitions']:
            partitions.append(__partition_linker__[part_name])
        mdraid = MDRaid(
            devname=raid['name'],
            level=raid['level'],
            members=partitions,
            spare_members=None,  # We might add support for this later
            file_system=fs_object,
            fsck_option=fsck_option,
            pv_name=raid.get('pv'),
            mount_point=raid.get('mount_point'))
        raid_objects.append(mdraid)

        if mdraid.pv_name:
            LOG.info('Rigging software RAID as PV: %s' % mdraid)
            __pv_linker__[mdraid.pv_name] = mdraid

    return raid_objects


def clear_linkers():
    __pv_linker__.clear()
    __partition_linker__.clear()


def layout_from_config(layout_config,
                       parted_path='parted',
                       partition_start=1048576,
                       alignment=1048576,
                       pe_size='4MiB'):
    LOG.info('Generating Layout')
    clear_linkers()  # Long running processes will leave these behind
    layout = generate_layout_stub(layout_config, parted_path)
    partition_tables = layout_config.get('partition_tables')
    if not partition_tables:
        raise GeneratorError('No partition tables have been defined')

    set_disk_labels(layout, layout_config)

    for pt in partition_tables:
        ptm = generate_partition_table_model(pt, partition_start, alignment)
        layout.add_partition_table_from_model(ptm)

    raid_configuration = layout_config.get('software_raid')
    if raid_configuration:
        raid_objects = generate_software_raid(raid_configuration)
        for raid in raid_objects:
            layout.add_software_raid(raid)

    volume_groups = layout_config.get('volume_groups')
    if volume_groups:
        vg_objects = generate_volume_group_models(volume_groups, pe_size)

        for vg in vg_objects:
            layout.add_volume_group_from_model(vg)

    return layout
