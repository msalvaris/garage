import sys
from ipykernel.iostream import OutStream
from toolz import curry


class NotebookLogger(OutStream):
    """ Will log output of notebook cells to specified outputs 
    
    >>> nb_logger = NotebookLogger(sys.stdout.session,
                                   sys.stdout.pub_thread,
                                   sys.stdout.name,
                                   sys.__stdout__)
    """
    def __init__(self, session, pub_thread, name, *outputs, pipe=None):
        self._outputs = outputs
        super(NotebookLogger, self).__init__(session, pub_thread, name, pipe=pipe)
        
    def write(self, message):
        for output in self._outputs:
            output.write(message)
        super(NotebookLogger, self).write(message)

        
@curry        
def redirect_to(orig_str, redirection):
    """ Redirects specified system attribute to redirection
    """
    orig_stdout = getattr(sys, orig_str)

    def reset_redirect():
        setattr(sys, orig_str, orig_stdout)

    setattr(sys, orig_str, redirection)
    return reset_redirect


error_to = redirect_to('stderr')
output_to = redirect_to('stdout')


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
    