import glob
import logging
import os

from press.helpers import deployment, package, cli
from press.targets import Target
from press.targets import util

log = logging.getLogger(__name__)


class LinuxTarget(Target):
    name = 'linux'

    mdadm_conf = '/etc/mdadm.conf'

    ssh_protocol_2_key_types = ('rsa', 'ecdsa', 'ed25519', 'dsa')
    locale_command = "/usr/sbin/locale-gen"

    def set_language(self, language):
        _locale = 'LANG=%s\nLC_MESSAGES=C\n' % language
        deployment.write(self.join_root('/etc/locale.conf'), _locale)

    def generate_locales(self):
        language = self.press_configuration.get('localization', {}).get(
            'language', 'en_US.utf8')
        cmd = '%s %s' % (self.locale_command, language)
        self.chroot(cmd)

    def set_timezone(self, timezone):
        localtime_path = self.join_root('/etc/localtime')
        deployment.remove_file(localtime_path)
        zone_info = os.path.join('../usr/share/zoneinfo/', timezone)
        deployment.create_symlink(zone_info, localtime_path)

    def localization(self):
        configuration = self.press_configuration.get('localization', dict())

        language = configuration.get('language')
        if language:
            log.info('Setting LANG=%s' % language)
            self.set_language(language)

        timezone = configuration.get('timezone')
        if timezone:
            log.info('Setting localtime: %s' % timezone)
            self.set_timezone(timezone)

        ntp_server = configuration.get('ntp_server')
        if ntp_server:
            log.info('Syncing time with: %s' % ntp_server)
            self.set_time(ntp_server)

    @staticmethod
    def set_time(ntp_server):
        # TODO: Make utility function
        # TODO: --utc/--local or setting the hardware clock at all should be configurable
        time_cmds = ['ntpdate %s' % ntp_server, 'hwclock --utc --systohc']
        for cmd in time_cmds:
            result = cli.run(cmd)
            if result.returncode:
                log.error('Failed to run %s: %s' % (cmd, result.returncode))

    def __groupadd(self, group, gid=None, system=False):
        if not util.auth.group_exists(group, self.root):
            log.info('Creating group %s' % group)
            self.chroot(util.auth.format_groupadd(group, gid, system))
        else:
            log.warn('Group %s already exists' % group)

    def authentication(self):
        configuration = self.press_configuration.get('auth')
        if not configuration:
            log.warn('No authentication configuration found')
            return

        users = configuration.get('users')
        if not users:
            log.warn('No users have been defined')

        for user in users:
            _u = users[user]
            if user != 'root':
                home_dir = _u.get('home', '/home/%s' % user)
                # Add user groups

                if 'group' in _u:
                    self.__groupadd(_u['group'], _u.get('gid'))
                groups = _u.get('groups')
                if groups:
                    for group in groups:
                        self.__groupadd(group)

                # Add user

                if not util.auth.user_exists(user, self.root):
                    log.info('Creating user: %s' % user)
                    self.chroot(
                        util.auth.format_useradd(user, _u.get('uid'),
                                                 _u.get('group'),
                                                 _u.get('groups'), home_dir,
                                                 _u.get('shell'),
                                                 _u.get('create_home', True),
                                                 _u.get('system', False)))
                else:
                    log.warn('Defined user, %s, already exists' % user)

            # Set password
            else:
                home_dir = _u.get('home', '/root')

            password = _u.get('password')
            if password:
                password_options = _u.get('password_options', dict())
                is_encrypted = password_options.get('encrypted', True)
                log.info('Setting password for %s' % user)
                self.chroot(
                    util.auth.format_change_password(user, password,
                                                     is_encrypted))
            else:
                log.warn('User %s has no password defined' % user)

            authorized_keys = _u.get('authorized_keys', list())
            if authorized_keys:
                log.info('Adding authorized_keys for %s' % user)
                ssh_config_path = self.join_root('%s/.ssh' % home_dir)
                log.debug(ssh_config_path)
                if not os.path.exists(ssh_config_path):
                    log.info('Creating .ssh directory: %s' % ssh_config_path)
                    deployment.recursive_makedir(ssh_config_path, mode=0o700)

                if authorized_keys:
                    public_keys_string = '\n'.join(
                        authorized_keys).strip() + '\n'
                    log.debug('Adding public key: %s' % public_keys_string)
                    deployment.write(
                        os.path.join(ssh_config_path, 'authorized_keys'),
                        public_keys_string,
                        append=True)

        # Create system groups

        groups = configuration.get('groups')
        if groups:
            for group in groups:
                _g = groups[group] or dict(
                )  # a group with no options will be null
                self.__groupadd(group, _g.get('gid'), _g.get('system'))

    def set_hostname(self):
        network_configuration = self.press_configuration.get(
            'networking', dict())
        hostname = network_configuration.get('hostname')
        if not hostname:
            log.warn('Hostname not defined')
            return
        log.info('Setting hostname: %s' % hostname)
        deployment.write(self.join_root('/etc/hostname'), hostname + '\n')

    def write_resolvconf(self):
        network_configuration = self.press_configuration.get(
            'networking', dict())
        dns_config = network_configuration.get('dns')
        if not dns_config:
            log.warn('Static DNS configuration is missing')
            return
        header = '# Generated by Press v%s\n' % package.get_press_version()
        log.info('Writing resolv.conf')
        search_domains = dns_config.get('search')
        nameservers = dns_config.get('nameservers')

        if nameservers or search_domains:
            for domain in search_domains:
                header += 'search %s\n' % domain

            for nameserver in nameservers:
                header += 'nameserver %s\n' % nameserver

            deployment.write(self.join_root('/etc/resolv.conf'), header)
        else:
            log.warn('No nameservers or search domains are defined')

    @property
    def network_configuration(self):
        return self.press_configuration.get('networking', {})

    def update_etc_hosts(self):
        hostname = self.network_configuration.get('hostname')
        if not hostname:
            return

        networks = self.network_configuration.get('networks')
        if not networks:
            return

        for network in networks:
            default_route = network.get('default_route')
            if default_route:
                data = '%s %s\n' % (network.get('ip_address'), hostname)
                log.info('Adding %s to /etc/hosts' % data)
                deployment.write(
                    self.join_root('/etc/hosts'), data, append=True)
                break

    def ssh_keygen(self,
                   path,
                   key_type,
                   passphrase='',
                   comment='localhost.localdomain'):
        deployment.remove_file(self.join_root(path))
        command = 'ssh-keygen -f %s -t%s -Cpress@%s -N \"%s\"' % (path,
                                                                  key_type,
                                                                  comment,
                                                                  passphrase)
        self.chroot(command)

    def update_host_keys(self):
        log.info('Updating SSH host keys')
        hostname = self.network_configuration.get('hostname',
                                                  'localhost.localdomain')
        for f in glob.glob(self.join_root('/etc/ssh/ssh_host*')):
            log.info('Removing: %s' % f)
            deployment.remove_file(f)
        for key_type in self.ssh_protocol_2_key_types:
            path = '/etc/ssh/ssh_host_%s_key' % key_type
            log.info('Creating SSH host key %s' % path)
            self.ssh_keygen(path, key_type, comment=hostname)

    def copy_resolvconf(self):
        if not os.path.exists('/etc/resolv.conf'):
            log.warn('Host resolv.conf is missing')
            return
        deployment.write(
            self.join_root('/etc/resolv.conf'),
            deployment.read('/etc/resolv.conf'))

    def write_mdadm_configuration(self):
        mdraid_data = self.chroot('mdadm --detail --scan')
        log.info('Writing mdadm.conf')
        log.debug(mdraid_data)
        deployment.write(
            self.join_root(self.mdadm_conf), mdraid_data + '\n', append=True)

    # noinspection PyMethodMayBeStatic
    def get_product_name(self):
        res = cli.run('dmidecode -s system-product-name', raise_exception=True)
        for line in res.splitlines():
            if line.lstrip().startswith('#'):
                continue
            return line.strip()

    def run(self):
        self.authentication()
        self.set_hostname()
        self.update_etc_hosts()
        self.copy_resolvconf()
