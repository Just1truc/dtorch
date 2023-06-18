import crypt
import random

from press.targets.util.misc import join_root


def search_auth_db(db_path, term, column, delimiter=':'):
    with open(db_path) as fp:
        for line in fp:
            if line:
                items = line.split(delimiter)
                if column > len(items):
                    raise IndexError('Column index %d is out of range' % column)
                if term == items[column]:
                    return line


def group_exists(group, root='/mnt/press'):
    path = join_root(root, '/etc/group')
    return search_auth_db(path, group, 0) and True or False


def user_exists(user, root='/mnt/press'):
    path = join_root(root, '/etc/passwd')
    return search_auth_db(path, user, 0) and True or False


def format_groupadd(name, gid=None, system=False):
    options = list()
    if gid:
        options.append('--gid %d' % gid)
    if system:
        options.append('--system')

    return 'groupadd %s %s' % (' '.join(options), name)


def format_useradd(name,
                   uid=None,
                   group='',
                   groups=(),
                   home_dir='',
                   shell='',
                   create_home=True,
                   system=False):
    options = list()
    if uid is not None:
        options.append('--uid %d' % uid)
    if group:
        options.append('--gid %s' % group)
    if groups:
        options.append('--groups %s' % ','.join(groups))
    if home_dir:
        options.append('--home-dir %s' % home_dir)
    if shell:
        options.append('--shell %s' % shell)
    if create_home:
        options.append('--create-home')
    else:
        options.append('--no-create-home')
    if system:
        options.append('--system')

    return 'useradd %s %s' % (' '.join(options), name)


def generate_salt512(length=12):
    pool = [chr(x) for x in range(48, 122) if chr(x).isalnum()]
    chars = list()
    while len(chars) < length:
        chars.append(random.choice(pool))
    return '$6$%s$' % ''.join(chars)


def format_change_password(name, p, is_encrypted=True):
    if not is_encrypted:
        salt = generate_salt512()
        hashed_password = crypt.crypt(p, salt)
    else:
        hashed_password = p
    return 'usermod --password \'%s\' %s' % (hashed_password, name)
