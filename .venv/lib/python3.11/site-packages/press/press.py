from __future__ import absolute_import

import logging

from functools import wraps
from size import Size

# Press imports
from press.exceptions import PressOrchestrationError, ImageValidationException
from press.generators.layout import layout_from_config
from press.generators.image import imagefile_generator
from press.helpers import deployment
from press.helpers.kexec import kexec
from press.layout.layout import MountHandler
from press.targets import VendorRegistry
from press.targets.registration import apply_extension, target_extensions
from press.hooks.hooks import run_hooks

log = logging.getLogger('press')


def run_if_layout(f):

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.has_layout:
            raise PressOrchestrationError(
                'Method {} requires a valid layout declaration')
        return f(self, *args, **kwargs)

    return wrapper


def run_if_imagefile(f):

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.has_imagefile:
            raise PressOrchestrationError(
                'Method {} requires a valid image declaration')
        return f(self, *args, **kwargs)

    return wrapper


class PressOrchestrator(object):
    """"""

    def __init__(self,
                 configuration,
                 parted_path='/usr/bin/parted',
                 deployment_root='/mnt/press',
                 staging_dir='/.press',
                 explicit_use_fibre_channel=None,
                 explicit_loop_only=None,
                 partition_start='1MiB',
                 alignment='1MiB',
                 lvm_pe_size='4MiB',
                 http_proxy=None,
                 skip_init=False):
        """

        :param configuration:
        :param parted_path:
        :param deployment_root:
        :param staging_dir:
        :param explicit_use_fibre_channel:
        :param explicit_loop_only:
        :param partition_start:
        :param alignment:
        :param lvm_pe_size:
        :param http_proxy:
        :param skip_init:
        """

        self.press_configuration = configuration
        self.parted_path = parted_path
        self.deployment_root = deployment_root
        self.staging_dir = staging_dir
        self.explicit_use_fibre_channel = explicit_use_fibre_channel
        self.explicit_loop_only = explicit_loop_only
        self.partition_start = partition_start
        self.alignment = alignment
        self.lvm_pe_size = lvm_pe_size
        self.http_proxy = http_proxy

        self.mount_handler = None
        self.perform_teardown = True

        self.layout = None
        self.imagefile = None

        self.image_target = self.press_configuration.get('target')
        self.post_configuration_target = VendorRegistry.targets.get(
            self.image_target)

        log.info('Press initializing', extra={'press_event': 'initializing'})

        if not skip_init:
            self.init_layout()
            self.init_imgfile()
            self.init_target()

        run_hooks("post-press-init", self.press_configuration)

    def init_layout(self):
        if 'layout' in self.press_configuration:
            # hacks allow us to easily override these from the library or
            # command line interface
            if self.explicit_loop_only is not None:
                self.press_configuration['layout']['loop_only'] = \
                    self.explicit_loop_only
            if self.explicit_use_fibre_channel is not None:
                self.press_configuration['layout']['use_fibre_channel'] = \
                    self.explicit_use_fibre_channel
            self.layout = layout_from_config(
                self.press_configuration['layout'],
                parted_path=self.parted_path,
                partition_start=Size(self.partition_start).bytes,
                alignment=Size(self.alignment).bytes,
                pe_size=self.lvm_pe_size)

    def init_imgfile(self):
        if 'image' in self.press_configuration:
            self.imagefile = imagefile_generator(
                self.press_configuration['image'], self.deployment_root,
                self.http_proxy)

    def init_target(self):
        if 'target' in self.press_configuration.get('target'):
            self.post_configuration_target = VendorRegistry.targets.get(
                self.press_configuration['target'])

    @property
    def has_layout(self):
        return bool(self.layout)

    @run_if_layout
    def apply_layout(self):
        log.info('Applying layout')
        self.layout.apply()

    @run_if_layout
    def mount_file_systems(self):
        self.mount_handler = MountHandler(self.deployment_root, self.layout)
        self.mount_handler.mount_physical()

    @run_if_layout
    def mount_pseudo_file_systems(self):
        if self.mount_handler:
            self.mount_handler.mount_pseudo()

    @run_if_layout
    def teardown(self):
        if self.mount_handler and self.perform_teardown:
            self.mount_handler.teardown()

    @run_if_layout
    def write_fstab(self):
        log.info('Writing fstab')
        deployment.create_fstab(self.layout.generate_fstab(),
                                self.deployment_root)

    @property
    def has_imagefile(self):
        return bool(self.imagefile)

    @run_if_imagefile
    def fetch_image(self):
        if self.imagefile.image_exists:
            if self.imagefile.can_validate:
                log.info('Image is present on file system. Hashing image')
                self.imagefile.hash_file()
            return

        def our_callback(total, done):
            log.debug('Downloading: %.1f%%' %
                      (float(done) / float(total) * 100))

        log.info('Starting Download...')
        self.imagefile.download(our_callback)
        log.info('done')

    @run_if_imagefile
    def validate_image(self):
        if self.imagefile.can_validate:
            log.info('Validating image...')
            if not self.imagefile.validate():
                raise ImageValidationException(
                    'Error validating image checksum')
            log.info('done')

    @run_if_imagefile
    def extract_image(self):
        log.info('Extracting image')
        self.imagefile.extract()
        if not self.press_configuration['image'].get('keep'):
            self.imagefile.cleanup()

    @run_if_imagefile
    def run_image_ops(self):

        run_hooks('pre-image-acquire', self.press_configuration)
        self.fetch_image()
        run_hooks('post-image-acquire', self.press_configuration)

        run_hooks('pre-image-validate', self.press_configuration)
        self.validate_image()
        run_hooks('post-image-validate', self.press_configuration)

        run_hooks('pre-image-extract', self.press_configuration)
        self.extract_image()
        run_hooks('post-image-extract', self.press_configuration)

    @staticmethod
    def run_extensions(obj):
        if not target_extensions:
            log.debug('There are no extensions registered')
        for extension in target_extensions:
            if apply_extension(extension, obj):
                log.info('Running Extension: %s' % extension.__name__)
                extension(obj).run()

    def post_configuration(self):
        if not self.post_configuration_target:
            log.info('Target: {} is not supported'.format(
                self.press_configuration['target']))
            return

        log.info('Running post configuration target')
        obj = self.post_configuration_target(self.press_configuration,
                                             self.layout, self.deployment_root,
                                             self.staging_dir)
        self.write_fstab()
        self.mount_pseudo_file_systems()
        run_hooks("pre-create-staging", self.press_configuration)
        self.create_staging_dir()
        run_hooks("pre-target-run", self.press_configuration)
        obj.run()
        run_hooks("pre-extensions", self.press_configuration)
        self.run_extensions(obj)
        run_hooks("post-extensions", self.press_configuration)
        self.remove_staging_dir()
        if hasattr(obj, 'write_resolvconf'):
            obj.write_resolvconf()

    @property
    def full_staging_dir(self):
        return self.deployment_root + '/' + self.staging_dir.lstrip('/')

    def create_staging_dir(self):
        deployment.recursive_makedir(self.full_staging_dir)

    def remove_staging_dir(self):
        deployment.recursive_remove(self.full_staging_dir)

    def run(self):
        log.info('Installation is starting', extra={'press_event': 'deploying'})
        run_hooks("pre-apply-layout", self.press_configuration)
        self.apply_layout()

        if self.has_imagefile:
            run_hooks("pre-mount-fs", self.press_configuration)
            self.mount_file_systems()
            log.info(
                'Fetching image at %s' % self.imagefile.url,
                extra={'press_event': 'downloading'})
            run_hooks("pre-image-ops", self.press_configuration)
            self.run_image_ops()
            log.info('Configuring image', extra={'press_event': 'configuring'})
            run_hooks("pre-post-config", self.press_configuration)
        else:
            log.info('Press configured in layout only mode, finishing up.')

        if self.post_configuration_target:
            self.post_configuration()

        log.info('Finished', extra={'press_event': 'complete'})

        # Experimental
        kexec_config = self.press_configuration.get('kexec')
        if kexec_config:
            self.perform_teardown = False
            kexec(layout=self.layout, **kexec_config)
