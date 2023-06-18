import json
import yaml
import yaml.scanner

from press.exceptions import PressCriticalException


def configuration_from_yaml(data):
    return yaml.safe_load(data)


def configuration_from_json(data):
    return json.loads(data)


def deduce_from_extension(path):
    yaml_extensions = ['yml', 'yaml']
    extension = path.split('.')[-1].lower()

    if extension in yaml_extensions:
        return 'yaml'
    if extension == 'json':
        return 'json'


def configuration_from_file(path, config_type=None):
    if not config_type:
        config_type = deduce_from_extension(path)

    with open(path) as fp:
        if config_type == 'yaml':
            return configuration_from_yaml(fp.read())
        elif config_type == 'json':
            return configuration_from_json(fp.read())
        else:
            # Brute force
            try:
                return configuration_from_yaml(fp.read())
            except yaml.scanner.ScannerError:
                pass

            try:
                return configuration_from_json(fp.read())
            except ValueError:
                raise PressCriticalException(
                    'Could not parse configuration file at {}'.format(path))
