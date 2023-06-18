import logging

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def setup_logging(log_level=logging.ERROR,
                  console_logging=True,
                  log_file=None,
                  cli_debug=False):

    press_logger = logging.getLogger('press')
    press_cli_logger = logging.getLogger('press.helpers.cli')

    if console_logging:  # True unless explicitly untrue
        stream_handler = logging.StreamHandler()
        press_logger.addHandler(stream_handler)
        press_logger.setLevel(log_level)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt=FORMAT))
        press_logger.info('Setting log file: {}'.format(log_file))
        press_logger.addHandler(fh)

    if not cli_debug:
        press_cli_logger.setLevel(logging.ERROR)
