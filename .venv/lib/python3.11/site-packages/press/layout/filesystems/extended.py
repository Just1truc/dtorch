import logging

from press.helpers.cli import run
from press.layout.filesystems import FileSystem
from press.exceptions import FileSystemCreateException, FileSystemFindCommandException

log = logging.getLogger(__name__)

FEATURE_HUGE_FILE = 'huge_file'
FEATURE_64BIT = '64bit'


class EXTFeature(object):

    def __init__(self, feature):
        self.feature = feature

    @staticmethod
    def is_feature_disabled(feature):
        return False if not feature else feature.startswith('^')

    def is_disabled(self):
        return self.is_feature_disabled(self.feature)

    @staticmethod
    def enable_feature(feature):
        if not feature:
            return None
        return feature.lstrip('^')

    def enabled(self, enable_flag):
        self.enable() if enable_flag else self.disable()

    def enable(self):
        self.feature = self.enable_feature(self.feature)
        return self.feature

    @staticmethod
    def disable_feature(feature):
        if not feature:
            return None
        if EXTFeature.is_feature_disabled(feature):
            return feature
        return '^{}'.format(feature)

    def disable(self):
        self.feature = self.disable_feature(self.feature)
        return self.feature

    @classmethod
    def negate_feature(cls, feature):
        feature = EXTFeature.normalize_feature(feature)
        if EXTFeature.is_feature_disabled(feature):
            return EXTFeature.enable_feature(feature)
        return EXTFeature.disable_feature(feature)

    @classmethod
    def normalize_feature(cls, feature):
        if hasattr(feature, 'feature'):
            return feature.feature
        return feature

    def __eq__(self, other_feature):
        other_feature = self.normalize_feature(other_feature)
        return self.feature == other_feature or \
            self.negate_feature(self.feature) == other_feature or \
            self.feature == self.negate_feature(other_feature) or \
            self.negate_feature(self.feature) == self.negate_feature(
                                                    other_feature)

    def __repr__(self):
        return self.feature

    def __str__(self):
        return self.feature


class EXT(FileSystem):
    fs_type = ''
    parted_fs_type_alias = 'ext2'
    command_name = ''

    # class level defaults
    _default_superuser_reserve = .03
    _default_stride_size = 0
    _default_stripe_width = 0
    _default_features = set()

    def __init__(self, label=None, mount_options=None, **extra):

        super(EXT, self).__init__(label, mount_options)

        self.superuser_reserve = extra.get('super_user_reserve',
                                           self._default_superuser_reserve)
        self.stride_size = extra.get('stride_size', self._default_stride_size)
        self.stripe_width = extra.get('stripe_width',
                                      self._default_stripe_width)

        if not hasattr(self, 'features'):
            self.features = set(extra.get('features', self._default_features))

        self.command_path = self.locate_command(self.command_name)

        if not self.command_path:
            raise \
                FileSystemFindCommandException(
                    'Cannot locate %s in PATH' % self.command_name)

        self.full_command = \
            '{command_path} -F -U{uuid} -m{superuser_reserve}' + \
            '{feature_options}{extended_options}{label_options} {device}'

        # algorithm for calculating stripe-width: stride * N where N are
        # member disks that are not used as parity disks or hot spares
        self.extended_options = ''
        if self.stripe_width and self.stride_size:
            self.extended_options = ' -E stride=%s,stripe_width=%s' % (
                self.stride_size, self.stripe_width)

        self.label_options = ''
        if self.fs_label:
            self.label_options = ' -L %s' % self.fs_label

        self.feature_options = ''
        if self.features:
            self.feature_options = ' -O {}'.format(','.join(
                list(self.features)))

        self.require_fsck = True

    def _enable_or_disable_features(self):
        raise NotImplementedError()

    def create(self, device):
        command = self.full_command.format(**dict(
            command_path=self.command_path,
            superuser_reserve=self.superuser_reserve,
            feature_options=self.feature_options,
            extended_options=self.extended_options,
            label_options=self.label_options,
            device=device,
            uuid=self.fs_uuid))
        log.info("Creating filesystem: {}".format(command))
        result = run(command)

        if result.returncode:
            raise FileSystemCreateException(self.fs_label, command, result)


class EXT2(EXT):
    fs_type = 'ext2'
    command_name = 'mkfs.ext2'


class EXT3(EXT):
    fs_type = 'ext3'
    command_name = 'mkfs.ext3'


class EXT4(EXT):
    fs_type = 'ext4'
    command_name = 'mkfs.ext4'

    _default_features = set()

    def __init__(self, label=None, mount_options=None, **extra):
        self.force_32_bit = extra.get('force_32_bit')
        self.features = set(extra.get('features', self._default_features))
        self._enable_or_disable_features()
        super(EXT4, self).__init__(label, mount_options, **extra)

    def _enable_or_disable_features(self):
        features_32bit = (EXTFeature.negate_feature(FEATURE_64BIT),
                          EXTFeature.negate_feature(FEATURE_HUGE_FILE))
        for f in self.features:
            feature = EXTFeature(f)
            if feature in (FEATURE_64BIT, FEATURE_HUGE_FILE):
                feature.enabled(self.force_32_bit is False)
            original_size = len(self.features)
            self.features.add(str(feature))
            if len(self.features) != original_size:
                self.features.remove(f)

        if (not self.features or
                not all(feat in self.features for feat in features_32bit)) and \
                self.force_32_bit is True:
            self.features.update(features_32bit)
