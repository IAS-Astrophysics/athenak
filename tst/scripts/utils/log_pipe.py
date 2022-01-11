# Provides LogPipe class to pipe output from subprocess to a log.
# Adapted from https://codereview.stackexchange.com/questions/6567

# Modules
import logging
import threading
import os


class LogPipe(threading.Thread):
    # Setup object with logger and a loglevel and start the thread
    def __init__(self, logger, level):
        super(LogPipe, self).__init__()
        # threading.Thread.__init__(self)
        self.logger = logging.getLogger(logger)
        self.daemon = False
        self.level = level
        self.fdRead, self.fdWrite = os.pipe()
        self.pipeReader = os.fdopen(self.fdRead)
        self.start()

    # Return the write file descriptor of the pipe
    def fileno(self):
        return self.fdWrite

    # Run the thread, logging everything.
    def run(self):
        for line in iter(self.pipeReader.readline, ''):
            self.logger.log(self.level, line.strip('\n'))
        self.pipeReader.close()

    # Close the write end of the pipe."""
    def close(self):
        os.close(self.fdWrite)
