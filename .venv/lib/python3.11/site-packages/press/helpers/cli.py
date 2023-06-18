import os
import logging
import shlex
import subprocess
import tempfile

log = logging.getLogger(__name__)


class AttributeString(str):

    def __init__(self, x):
        """
        For introspection
        """
        str.__init__(x)
        self.stderr = ''
        self.returncode = None
        self.command = ''
        self.comments = ''

    @property
    def stdout(self):
        return self


class CLIException(Exception):
    """
    Exception used by cli.run function
    """
    pass


def run(command,
        bufsize=1048567,
        dry_run=False,
        raise_exception=False,
        ignore_error=False,
        quiet=False,
        env=None,
        _input=None):
    """Runs a command and stores the important bits in an attribute string.

    :param command: Command to execute.
    :type command: str.

    :param bufsize: Buffer line size.
    :type bufsize: int.

    :param dry_run: Should we perform a dry run of the command.
    :type dry_run: bool.
    :param raise_exception:
    :param ignore_error:
    :param quiet:
    :param env:
    :param _input:

    :returns: :func:`press.cli.AttributeString`.

    """
    if not quiet:
        log.debug('Running: %s' % command)
    our_env = os.environ.copy()
    our_env.update(env or dict())
    cmd = shlex.split(str(command))

    if not dry_run:
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if _input else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=bufsize,
            env=our_env)
        out, err = p.communicate(input=_input)
        ret = p.returncode
    else:
        out, err, ret = b'', b'', 0

    if not quiet:
        log.debug('Return Code: %d' % ret)
        if out:
            log.debug('stdout: \n%s' % out.strip().decode('utf-8'))
    if ret and not ignore_error:
        log.error('Return: %d running: %s stdout: %s\nstderr: \n%s' %
                  (ret, command, out.strip().decode('utf-8'),
                   err.strip().decode('utf-8')))
        if raise_exception:
            raise CLIException(err)

    attr_string = AttributeString(out.decode('utf-8'))
    attr_string.stderr = err.decode('utf-8')
    attr_string.returncode = ret
    attr_string.command = command
    return attr_string


def find_in_path(filename):
    if os.path.isabs(filename):
        if os.path.exists(filename):
            return filename
        else:
            return None

    for path in os.environ['PATH'].split(os.pathsep):
        abspath = os.path.join(path, filename)
        if os.path.exists(abspath):
            return abspath


def run_chroot(command,
               root='/mnt/press',
               staging_dir='/.press_staging',
               bufsize=1048567,
               dry_run=False,
               raise_exception=False,
               ignore_error=False,
               quiet=False,
               env=None,
               unlink=True,
               proxy=None):

    pre = \
        'export PATH=\"/bin:/usr/bin:/usr/local/bin:/sbin:/usr/sbin:' \
        '/usr/local/sbin\"\n'
    if proxy:
        pre += 'export HTTP_PROXY=http://%s\nexport HTTPS_PROXY=http://%s\n' \
               'export http_proxy=http://%s\n' % (proxy, proxy, proxy)
    abs_path = os.path.join(root, staging_dir.lstrip('/'))
    f = tempfile.NamedTemporaryFile(
        mode='w', suffix='.sh', prefix='press-', dir=abs_path, delete=False)
    f.write('#!/bin/bash\n%s%s\n' % (pre, command.strip()))
    f.flush()
    f.close()
    os.chmod(f.name, 0o700)
    script_path = os.path.join(staging_dir, os.path.split(f.name)[1])
    log.debug('chroot: %s' % command)
    cmd = 'chroot %s %s' % (root, script_path)
    r = run(cmd, bufsize, dry_run, raise_exception, ignore_error, quiet, env)
    if unlink:
        os.unlink(f.name)
    return r
