import hashlib
import logging
import os
import requests
import time

from press.helpers.deployment import tar_extract

# noinspection PyUnresolvedReferences
from six.moves import urllib

log = logging.getLogger(__name__)


class Download(object):

    def __init__(self,
                 url,
                 hash_method=None,
                 expected_hash=None,
                 download_directory=None,
                 filename=None,
                 chunk_size=20480,
                 proxy=None):
        """A Download that also generates the checksum while downloading.

        Rarely will this clean be used directly.

        :param url: url of the file to Download
        :param hash_method: the actual hash algorithm used, sha1, md5, etc.
        :param expected_hash: expected hash of the file
        :param download_directory: directory to use to store the downloaded file
        :param filename: File name to store the download as, if None then it will be based on the original url, if
            that a name can't be determined, then a random name will be generated.
        :param chunk_size: how large of chunks to read from the stream.
        :param proxy: proxy server to use.
        """

        self.url = url
        self.hash_method = hash_method
        self.expected_hash = expected_hash
        self.download_directory = download_directory
        self.filename = filename
        self.chunk_size = chunk_size
        self.proxy = proxy
        self._hash = None

        if hash_method is not None:
            self.hash_method = hash_method.lower()
            self._hash = hashlib.new(self.hash_method)

        if expected_hash is not None:
            self.expected_hash = expected_hash.lower()

        if download_directory is None:
            self.download_directory = '/tmp'

        parsed_url = urllib.parse.urlparse(self.url)

        self.url_scheme = parsed_url.scheme

        if filename is None:
            new_filename = os.path.basename(parsed_url.path)
            if new_filename:
                self.filename = new_filename
            else:
                self.filename = '%d.tmp' % time.time()

        self.full_filename = os.path.join(self.download_directory,
                                          self.filename)

    def __repr__(self):
        output = []
        for attr_name in ('url', 'hash_method', 'expected_hash',
                          'download_directory', 'filename', 'chunk_size'):
            attr = getattr(self, attr_name)
            output.append('%s=%s' % (attr_name, attr))
        return 'Download(%s)' % ', '.join(output)

    def hash_file_url(self):
        """

        :return:
        """
        # urlparse doesn't support relative paths with file://
        self.full_filename = self.url.split('file://')[1]

    def download(self, callback_func=None):
        """Start the download

        Downloads the file located at self.url to self.download_directory

        :param callback_func: expected signature is callback_func(total_size, total_downloaded)

        Raises an exception on error, otherwise the download was completed(as far as http is concerned)

        """
        if self.url_scheme == 'file':
            self.hash_file_url()
            return

        byte_count = 0
        if self.proxy:
            proxies = {'http': self.proxy, 'https': self.proxy}
        else:
            proxies = None

        ret = requests.get(self.url, stream=True, proxies=proxies)
        ret.raise_for_status()

        content_length = int(ret.headers.get('content-length', '0'))

        with open(self.full_filename, 'wb') as download_file:
            for chunk in ret.iter_content(self.chunk_size):
                byte_count += len(chunk)
                if self._hash is not None:
                    self._hash.update(chunk)
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
        return self._hash.hexdigest() == self.expected_hash

    def prepare_for_extract(self):
        """Prepare for extraction.

        This doesn't do anything, subclasses(like qcow) or something could do something like mount an image
        """
        assert self  # stop pycharm from telling me this can be static
        pass

    def extract(self, target_path):
        """Extracts downloaded file to the target_path

        :param target_path: path to store extracted file in(must exist already)

        Returns an _AttributeString
        """

        # This is some ghetto crap to ensure that older versions of gnu-tar don't belly ache when it encounters a bz2.
        return tar_extract(self.full_filename, chdir=target_path)

    def cleanup(self):
        """Deletes downloaded file in this version
        """
        os.unlink(self.full_filename)
