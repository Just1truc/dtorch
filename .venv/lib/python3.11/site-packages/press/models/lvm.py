from size import Size


class VolumeGroupModel(object):

    def __init__(self, name, physical_volumes, pe_size='4MiB'):
        self.logical_volumes = list()

        self.name = name
        self.physical_volumes = physical_volumes
        self.pe_size = Size(pe_size)

    def add_logical_volume(self, lv):
        self.logical_volumes.append(lv)

    def add_logical_volumes(self, lvs):
        for lv in lvs:
            self.add_logical_volume(lv)
