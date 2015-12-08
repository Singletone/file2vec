import sys
from datetime import datetime


class Progress:
    def __init__(self, message, maxValue):
        self.message = message
        self.value = 0
        self.maxValue = maxValue
        self.percentage = 0
        self.startTime = None


    def __enter__(self):
        self.startTime = datetime.now()
        return self.update


    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.write('\n')
        return self


    @staticmethod
    def percentage(value, maxValue):
        return 100 * float(value) / float(maxValue)


    @staticmethod
    def delta(currentTime, startTime):
        deltaString = str(currentTime - startTime)
        deltaString = deltaString.split('.')[0]

        return deltaString


    def update(self, value, **kwargs):
        params = dict(self.__dict__, **kwargs)

        params['value'] = value
        params['percentage'] = Progress.percentage(value, self.maxValue)
        params['elapsed'] =  Progress.delta(datetime.now(), self.startTime)

        message = self.message % params
        sys.stdout.write('\r')
        sys.stdout.write(message)
        sys.stdout.flush()


def start(message, maxValue):
    return Progress(message, maxValue)