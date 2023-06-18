import hashlib
import logging
import os
import requests

from press.exceptions import PressCriticalException
from press.helpers.deployment import tar_extract

# noinspection PyUnresolvedReferences
from six.moves import urllib

log = logging.getLogger(__name__)


class ImageFile(object):

    def __init__(self,
                 url,
                 target,
                 hash_method=None,
                 expected_hash=None,
                 download_directory=None,
                 buffer_size=20480,
                 proxy=None):
        """
        Extending (or renaming really) Chad Catlett's Download class.

        Image download, detection (tar?,cpio?,qcow?), validation, extraction, and cleanup operations.

        :param url: extended url format, supporting http://,https://,file:// <and more?>
        :param target: a top-level directory where we extract/copy image files to
        :param hash_method: sha1, md5, sha256 or None
        :param expected_hash: pre-recorded hash
        :param download_directory: where to place the temporary file
        :param buffer_size: maximum size of buffer for stream operations
        :param proxy: a proxy server in host:port format
        """

        self.url = url
        self.target = target
        self.hash_method = hash_method
        self.expected_hash = expected_hash
        self.download_directory = download_directory
        self.buffer_size = buffer_size
        self.proxy = proxy
        self._hash_object = None
        self.image_exists = False
        self.url_scheme = self.full_filename = None

        if hash_method:
            self.hash_method = hash_method.lower()
            self._hash_object = hashlib.new(self.hash_method)

        if download_directory is None:
            self.download_directory = self.target

        self.filename_from_url()

    def filename_from_url(self):
        parsed_url = urllib.parse.urlparse(self.url)
        self.url_scheme = parsed_url.scheme or 'file'
        filename = parsed_url.path
        if self.url_scheme == 'file':
            if not os.path.isfile(filename):
                raise PressCriticalException(
                    'Specified image file is not present')
            self.image_exists = True
            self.full_filename = filename
        else:
            self.full_filename = os.path.join(self.download_directory,
                                              os.path.basename(filename))

    def hash_file(self):
        """
        If we are not downloading the file, we still need to hash it
        :return:
        """
        with open(self.full_filename, 'rb') as fp:
            while True:
                data = fp.read(self.buffer_size)
                if not data:
                    break
                self._hash_object.update(data)

    def download(self, callback_func):
        byte_count = 0
        if self.proxy:
            proxies = {'http': self.proxy, 'https': self.proxy}
        else:
            proxies = None

        res = requests.get(self.url, stream=True, proxies=proxies)
        res.raise_for_status()

        content_length = int(res.headers.get('content-length', '0'))

        with open(self.full_filename, 'wb') as download_file:
            for chunk in res.iter_content(self.buffer_size):
                byte_count += len(chunk)
                if self._hash_object is not None:
                    self._hash_object.update(chunk)
                download_file.write(chunk)
                if callback_func:
                    callback_func(content_length, byte_count)

    @property
    def can_validate(self):
        """Can validate() actually work?

        Returns True if it is safe to call self.validate, otherwise False
        """
        return True if self.hash_method and self.expected_hash else False

    def validate(self):
        """Validate the checksum matches the expected checksum

        returns True if the checksum is matches expected_hash, otherwise False
        """
        return self._hash_object.hexdigest() == self.expected_hash

    def prepare_for_extract(self):
        """Prepare for extraction.

        This doesn't do anything, subclasses(like qcow) or something could do something like mount an image
        """
        assert self  # stop pycharm from telling me this can be static
        pass

    def extract(self):
        """Extracts downloaded file to the target_path
        Returns an _AttributeString
        """

        # This is some ghetto crap to ensure that older versions of gnu-tar don't belly ache when it encounters a bz2.
        return tar_extract(self.full_filename, chdir=self.target)

    def cleanup(self):
        """Deletes downloaded file in this version
        """
        os.unlink(self.full_filename)
