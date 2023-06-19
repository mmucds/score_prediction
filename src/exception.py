import os
import sys
from src.logger import logging


def create_detail_error_message(error: str, error_detail: sys):
    _, _, exception_traceback = error_detail.exc_info()
    error_file_name = os.path.split(exception_traceback.tb_frame.f_code.co_filename)[1]
    error_line_number = exception_traceback.tb_lineno
    error_message = f'[{error}] error occurred in [{error_line_number}] number line from [{error_file_name}] script.'
    return error_message


class CustomException(Exception):
    def __init__(self, error, error_detail):
        super().__init__(error)
        self.error_message = create_detail_error_message(error=error,
                                                         error_detail=error_detail)

    def __str__(self):
        return self.error_message


if __name__ == '__main__':
    try:
        a = 1/0
    except Exception as ex:
        logging.info('Divide by Zero Error.')
        raise CustomException(ex, sys)


