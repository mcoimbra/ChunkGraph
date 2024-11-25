import logging

# Configure logging.
# logging.basicConfig(
#     level=logging.INFO,
#     format="[%(levelname)s] - %(message)s",
# )

class Logger:
    _logger: logging.Logger = None

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        if Logger._logger is None:
            Logger._logger = logging.getLogger(name)
            logging.basicConfig(
                #level=logging.DEBUG,
                #format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                level=logging.INFO,
                format="[%(levelname)s] - %(message)s"
            )
        return Logger._logger

# def get_logger(name: str) -> logging.Logger:
#     logger: logging.Logger = logging.getLogger(name)
#     return logger