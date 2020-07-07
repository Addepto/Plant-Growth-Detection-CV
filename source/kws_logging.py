"""
Logging module
"""
import logging


def config_logger(output_path, file_name):
    """
    Configure logger for the script
    :param output_path: where the logs should be saved
    :param file_name: name of the file for the log
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                        handlers=[
                            logging.FileHandler("{0}/{1}.log".format(output_path, file_name), mode='a'),
                            logging.StreamHandler()
                        ])
