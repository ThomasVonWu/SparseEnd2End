import logging
import colorlog

log_colors_config = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}


def set_logger(log_path, save_file=False):
    logger = logging.getLogger("logger_name")
    console_handler = logging.StreamHandler()

    file_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d::%H:%M:%S"
    )

    console_formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s[%(asctime)s] [%(levelname)s]:%(reset)s %(message)s",
        datefmt="%Y-%m-%d::%H:%M:%S",
        log_colors=log_colors_config,
    )

    console_handler.setFormatter(console_formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)

    console_handler.close()

    file_handler = None
    if save_file:
        file_handler = logging.FileHandler(filename=log_path, mode="a", encoding="utf8")
        file_handler.setFormatter(file_formatter)
        if not logger.handlers:
            logger.addHandler(file_handler)
        file_handler.close()

    return logger, file_handler, console_handler


def logger_wrapper(
    log_path="",
    save_file=False,
    level1=logging.DEBUG,
    level2=logging.DEBUG,
    level3=logging.DEBUG,
):
    logger, file_handler, console_handler = set_logger(log_path, save_file)

    logger.setLevel(level1)
    console_handler.setLevel(level2)
    if save_file:
        file_handler.setLevel(level3)
    return logger, file_handler, console_handler
