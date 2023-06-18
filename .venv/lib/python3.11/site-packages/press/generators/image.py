import logging
from press import exceptions
from press.helpers.imagefile import ImageFile
from press.hooks.hooks import run_hooks

log = logging.getLogger(__name__)

_default_chunk_size = 1048576
_default_image_format = 'tgz'
_default_hash_method = 'sha1'


def imagefile_generator(image_config, target, proxy_info=None):
    checksum_details = image_config.get('checksum')
    if checksum_details:
        image_hash = checksum_details.get('hash')
        if not image_hash:
            raise exceptions.GeneratorError(
                'hash checksum is missing, check configuration')
        hash_method = checksum_details.get('method', _default_hash_method)
    else:
        image_hash = None
        hash_method = None

    dl = ImageFile(
        url=image_config['url'],
        target=target,
        hash_method=hash_method,
        expected_hash=image_hash,
        buffer_size=image_config.get('chunk_size', _default_chunk_size),
        proxy=proxy_info)

    return dl
