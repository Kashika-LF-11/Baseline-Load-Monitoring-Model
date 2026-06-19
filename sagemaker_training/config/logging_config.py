import logging
import sys


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging for the training container.

    Messages will include level, module, and message. Keep format simple
    so it's compatible with most container log collectors.
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    if not logger.handlers:
        logger.addHandler(handler)
    return logging.getLogger("sagemaker_training")


def get_logger(name: str = __name__) -> logging.Logger:
    return logging.getLogger(name)
