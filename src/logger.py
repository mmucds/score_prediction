import os
import logging
from datetime import datetime

dir_name = f"{datetime.now().strftime('%d_%m_%Y_%S_%M_%H')}"
dir_path = os.path.join(os.getcwd(), '../../logs', dir_name)
os.makedirs(dir_path, exist_ok=True)
log_file_path = os.path.join(dir_path, f'{dir_name}.log')

logging.basicConfig(filename=log_file_path,
                    format='[ %(asctime)s ] Line : %(lineno)d, File Name : %(name)s, %(levelname)s - %(message)s',
                    level=logging.INFO)

if __name__ == '__main__':
    logging.info('Logging has started.')
