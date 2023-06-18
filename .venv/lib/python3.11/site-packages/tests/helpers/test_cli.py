import mock
import pytest
import unittest

from press.helpers import cli


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.fake_subprocess = mock.patch('press.helpers.cli.subprocess')
        self.mock_subprocess = self.fake_subprocess.start()
        self.mock_subprocess.Popen = mock.Mock()
        self.mock_process = mock.Mock()
        self.mock_subprocess.Popen.return_value = self.mock_process

    def tearDown(self):
        self.fake_subprocess.stop()

    def test_run(self):
        self.mock_process.communicate = mock.Mock()
        self.mock_process.communicate.return_value = (b'hello', b'')
        self.mock_process.returncode = 0

        r = cli.run('nocmd')

        self.assertEqual(r.stdout, 'hello')

        # Test dry run path
        r = cli.run('nocmd', dry_run=True)
        assert r.returncode == 0

        # Make sure we raise an exception when expected
        with pytest.raises(cli.CLIException):
            self.mock_process.returncode = 99
            cli.run('nocmd', raise_exception=True)
