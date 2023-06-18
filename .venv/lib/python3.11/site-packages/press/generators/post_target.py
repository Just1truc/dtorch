from press.targets.linux.redhat.enterprise_linux.enterprise_linux_7.enterprise_linux_7_target \
    import EL7Target
from press.targets.linux.redhat.enterprise_linux.enterprise_linux_6.enterprise_linux_6_target \
    import EL6Target

target_mapping = {'rhel7': EL7Target, 'rhel6': EL6Target}
