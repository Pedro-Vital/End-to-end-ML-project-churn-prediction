import sys
from typing import Any


def error_message_detail(error: Exception, error_detail: Any) -> str:
    _, _, exc_tb = sys.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "<unknown>"
        line_number = 0

    error_message = (
        f"Error occurred in python script [{file_name}] "
        f"line number [{line_number}] error message [{error}]"
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: Any):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self) -> str:
        return self.error_message
