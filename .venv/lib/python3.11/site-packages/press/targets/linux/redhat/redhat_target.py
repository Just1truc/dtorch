import os
import logging

from press.helpers import deployment
from press.targets.linux.linux_target import LinuxTarget
from press.hooks.hooks import add_hook

log = logging.getLogger(__name__)


class RedhatTarget(LinuxTarget):
    name = 'redhat'

    rpm_path = '/usr/bin/rpm'
    yum_path = '/usr/bin/yum'
    yum_config_file = '/etc/yum.conf'
    yum_config_backup = '/etc/yum.conf_bak'
    release_file = '/etc/redhat-release'

    def __init__(self, press_configuration, layout, root, chroot_staging_dir):
        super(RedhatTarget, self).__init__(press_configuration, layout, root,
                                           chroot_staging_dir)
        add_hook(self.add_repos, "pre-extensions", self)

    def get_package_list(self):
        command = \
            '%s --query --all --queryformat \"%%{NAME}\\n\"' % self.rpm_path
        out = self.chroot(command, quiet=True)
        return out.splitlines()

    def enable_yum_proxy(self, proxy):
        log.info('Enabling global yum proxy: %s' % proxy)
        self.chroot('/bin/cp {} {}'.format(self.yum_config_file,
                                           self.yum_config_backup))
        self.chroot('echo proxy=http://{} >> {}'.format(proxy,
                                                        self.yum_config_file))

    def disable_yum_proxy(self):
        log.info('Restoring original yum configuration')
        self.chroot('/bin/mv {} {}'.format(self.yum_config_backup,
                                           self.yum_config_file))

    def install_package(self, package):
        command = '{} install -y --quiet {}'.format(self.yum_path, package)
        res = self.chroot(command)
        if res.returncode:
            log.error('Failed to install package {}'.format(package))
        else:
            log.info('Installed: {}'.format(package))

    def install_packages(self, packages):
        command = '{} install -y --quiet {}'.format(self.yum_path,
                                                    ' '.join(packages))
        res = self.chroot(command)
        if res.returncode:
            log.error('Failed to install packages: {}'.format(
                ' '.join(packages)))
        else:
            log.info('Installed: {}'.format(' '.join(packages)))
        return res.returncode

    def add_repo(self, name, mirror, gpgkey):
        path_name = name.lower().replace(" ", "_")
        log.info('Creating repo file for "{name}"'.format(name=name))
        sources_path = self.join_root(
            '/etc/yum.repos.d/{name}.repo'.format(name=path_name))
        source = "[{lower_name}]\n" \
                 "name={formal_name}\n" \
                 "baseurl={mirror}\n" \
                 "enabled=1".format(lower_name=path_name,
                                    formal_name=name,
                                    mirror=mirror)
        if gpgkey:
            source += "\ngpgcheck=1"
            source += "\ngpgkey={gpgkey}".format(gpgkey=gpgkey)
        else:
            source += "\ngpgcheck=0"

        deployment.write(sources_path, source)

    def remove_repo(self, name):
        path_name = name.lower().replace(" ", "_")
        log.info('Removing repo file for "{name}"'.format(name=name))
        sources_path = self.join_root(
            '/etc/yum.repos.d/{name}.repo'.format(name=path_name))
        deployment.remove_file(sources_path)

    def add_repos(self, press_config):
        for repo in press_config.get('repos', []):
            self.add_repo(repo['name'], repo['mirror'], repo.get(
                'gpgkey', None))

    def package_exists(self, package_name):
        for package in self.get_package_list():
            if package_name == package.strip():
                return True
        return False

    def packages_exist(self, package_names):
        match = dict([(name, False) for name in package_names])
        for package in self.get_package_list():
            if package in match:
                match[package] = True
        if False in match.values():
            return False
        return True

    def packages_missing(self, packages):
        missing = list()
        installed_packages = self.get_package_list()
        for package in packages:
            if package not in installed_packages:
                missing.append(package)
        return missing

    @property
    def has_redhat_release(self):
        return os.path.exists(self.join_root('/etc/redhat-release'))

    def parse_el_release(self):
        """
        Heuristic garbage
        :return:
        """
        release_info = dict()
        if not self.has_redhat_release:
            return release_info
        data = deployment.read(self.join_root(self.release_file))
        data = data.strip()
        try:
            if 'oracle' in self.release_file:
                version = data.split()[-1]
            else:
                release_info['codename'] = data.split()[-1].strip('()')
                version = data.split()[-2]
            release_info['version'] = version
            # Sometimes need short version, not ones like '7.1.1503'
            # so splitting and then joining [0:2] to get '7.1'
            release_info['short_version'] = '.'.join(version.split('.')[:2])
            # Also need major version, like '7'
            # rather than short_version like '7.1'
            release_info['major_version'] = version.split('.')[0]
            release_info['os'] = data.split('release')[0].strip()
        except IndexError:
            log.error('Error parsing {} release file'.format(self.release_file))
        return release_info

    def parse_os_release(self):
        """
        reads /etc/os-release, excluding blank lines and returns dict()
        """
        with open(self.join_root("/etc/os-release")) as f:
            os_release = {}
            for line in f:
                if line.rstrip():
                    k, v = line.rstrip().split("=")
                    os_release[k] = v.strip('"')
        return os_release

    def get_el_release_value(self, key):
        el_release = self.parse_el_release()
        value = el_release.get(key)
        return value

    def get_os_release_value(self, key):
        """
        parses /etc/os_release and returns the key value passed in
        """
        os_release = self.parse_os_release()
        value = os_release.get(key)
        return value

    def baseline_yum(self, proxy):
        # TODO: Un-raxify this
        """
        Check major version an adjust for .eus or .z path on mirror
        Check to see if we need proxy, and enable in yum.conf
        Check if we are 'rhel' and if so add base repo
        """
        os_id = self.get_el_release_value('os')
        short_version = self.get_el_release_value('short_version')
        rhel_repo_name = 'rhel_base'

        if short_version[0] == '6':
            short_version = '6'
        elif short_version[0] == '7':
            short_version = str(short_version) + '.eus'

        rhel_repo_url = \
            'http://intra.mirror.rackspace.com/kickstart/'\
            'rhel-x86_64-server-{version}/'.format(version=short_version)

        if proxy:
            self.enable_yum_proxy(proxy)
        if 'Red Hat' in os_id:
            self.add_repo(rhel_repo_name, rhel_repo_url, gpgkey=None)

    def revert_yum(self, proxy):
        """
        Reverts changes from baseline yum:
        Disabled proxy
        If 'rhel' removes the base repo
        """
        os_id = self.get_el_release_value('os')
        rhel_repo_name = 'rhel_base'

        if proxy:
            self.disable_yum_proxy()
        if 'Red Hat' in os_id:
            self.remove_repo(rhel_repo_name)

    def service_control(self, service, action):
        log.info('Running: service {} {}'.format(service, action))
        command = 'service {} {}'.format(service, action)
        self.chroot(command)
