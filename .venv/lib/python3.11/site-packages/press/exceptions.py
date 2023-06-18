class PressException(Exception):
    pass


class FileSystemCreateException(PressException):

    def __init__(self, fs_type, fs_command, attr_str):
        self.fs_type = fs_type
        self.fs_command = fs_command
        self.attr_str = attr_str


class FileSystemFindCommandException(PressException):
    """
    Raised when FileSystem class cannot discover
    """


class PartitionValidationError(PressException):
    pass


class LayoutValidationError(PressException):
    pass


class LVMValidationError(PressException):
    pass


class PhysicalDiskException(PressException):
    pass


class GeneralValidationException(PressException):
    pass


class GeneratorError(PressException):
    pass


class ConfigurationError(PressException):
    pass


class PressCriticalException(PressException):
    pass


class ImageValidationException(PressException):
    pass


class NetworkConfigurationError(ConfigurationError):
    pass


class HookError(PressException):
    pass


class PressOrchestrationError(PressException):
    pass


class OSImageException(PressException):
    pass
