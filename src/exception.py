import sys
import logging
try:
    from . import logger
except ImportError:
    # When running directly, import from same directory
    import logger

def error_message_detail(error, error_detail):
    _,_,exc_tb = sys.exc_info() # exc_tb is the traceback object of the exception that was raised
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno # line number where exception occurred
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, line_number, str(error)
    )
    return error_message

class CustomException(Exception):

    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        logging.error(self.error_message)
         
    def __str__(self):
        return self.error_message
    

if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by zerooooo")
        raise CustomException(e, sys)