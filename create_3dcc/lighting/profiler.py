''' A simple profiler for logging '''
import logging
import time


class Profiler(object):
    def __init__(self, name, logger=None, level=logging.INFO):
        self.name = name
        self.logger = logger
        self.level = level

    def step(self, name):
        """ Returns the duration and stepname since last step/start """
        duration = self.summarize_step(start=self.step_start, step_name=name, level=self.level)
        now = time.time()
        self.step_start = now
        return duration

    def __enter__(self):
        self.start = time.time()
        self.step_start = time.time()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.summarize_step(self.start, step_name="complete")

    def summarize_step(self, start, step_name="", level=None):
        duration = time.time() - start
        step_semicolon = ':' if step_name else ""
        if self.logger:
            level = level or self.level
            self.logger.log(self.level,
                            "{name}{step}: {secs} seconds".format(name=self.name, step=step_semicolon + step_name,
                                                                  secs=duration))
        else:
            print("{name}{step}: {secs} seconds".format(name=self.name, step=step_semicolon + step_name, secs=duration))
        return duration
