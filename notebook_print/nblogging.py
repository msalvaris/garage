import sys


class Logger(object):
    """ Pushes the output to one or more outputs
    """
    def __init__(self, *outputs):
        self._outputs = outputs

    def write(self, message):
        for out in self._outputs:
            out.write(message)

    def flush(self):
        pass 
    
    
def error_to(output):
    """ Redirects stderr to a specified output location
    """
    orig_stderr = sys.stderr
    
    def reset_stderr():
        sys.stderr = orig_stdout
    
    sys.stderr = output
    return reset_stderr


def output_to(output):
    """ Redirects stdout to a specified output location
    """
    orig_stdout = sys.stdout
    
    def reset_stdout():
        sys.stdout = orig_stdout
    
    sys.stdout = output
    return reset_stdout