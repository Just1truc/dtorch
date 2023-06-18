import argparse


def parse_args(arguments):
    """
    :param arguments: a list of arguments to parse
    :return: Argument namespace
    """
    parser = argparse.ArgumentParser(
        prog='press',
        description='Image installation utility that supports custom '
        'partitioning')
    parser.add_argument(
        '--parted-path',
        default='parted',
        help='The location of an alternative parted binary')
    parser.add_argument(
        '--deployment-root',
        default='/mnt/press',
        help='The location were we mount file systems')

    subparsers \
        = parser.add_subparsers(dest='command', help='<command> help')

    # apply command
    apply_parser = subparsers.add_parser(
        'apply', help='Apply a configuration to disk')
    apply_parser.add_argument(
        'configuration',
        help='The press configuration (state) file to apply (yaml/json)')
    apply_parser.add_argument(
        '--layout-only',
        action='store_true',
        help='Only apply the disk layout to disk, skipping image laydown')
    apply_parser.add_argument(
        '--quiet', action='store_true', help='Only log errors to console')
    apply_parser.add_argument(
        '--debug', action='store_true', help='Active verbose logging')
    apply_parser.add_argument(
        '--log-file',
        default=None,
        help='Enable file logging by specifying a file to log to')
    apply_parser.add_argument(
        '--plugin-dir',
        action='append',
        default=[],
        dest='plugin_dirs',
        help='A plugin directory to search. Can be used multiple times')
    apply_parser.add_argument(
        '--plugin',
        action='append',
        default=[],
        dest='plugins',
        help='A plugin that you would like to enable. Can be used multiple '
        'times')
    apply_parser.add_argument(
        '--staging-dir',
        default='/.press',
        help='This is the staging directory in the '
        'chroot')
    apply_parser.add_argument(
        '--use-fibre-channel',
        default=False,
        action='store_true',
        help='Enable fibre channel support')
    apply_parser.add_argument(
        '--loop-only',
        default=False,
        action='store_true',
        help='Only target loopback devices.')
    apply_parser.add_argument(
        '--cli-debug',
        action='store_true',
        help='Enabling debugging of raw CLI output')
    apply_parser.add_argument(
        '--partition-start',
        default='1MiB',
        help='Where the first partition should start. Default 1MiB')
    apply_parser.add_argument(
        '--alignment',
        default='1MiB',
        help='Alignment partitions to this boundary. Default 1MiB')
    apply_parser.add_argument(
        '--lvm-pe-size',
        default='4MiB',
        help='Physical extent size for logical volume groups')
    apply_parser.add_argument(
        '--proxy',
        default=None,
        help='Specify a proxy server to use when downloading image <host:port>')

    # info command
    # info_parser = subparsers.add_parser(
    #     'disks',
    #     help='Get information about the current drive configuration'
    # )
    # info_parser.add_argument('-l',
    #                          '--list',
    #                          action='store_true',
    #                          help='List valid disk targets')
    return parser.parse_args(args=arguments)
