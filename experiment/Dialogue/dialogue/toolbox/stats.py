import sys
import time
import numpy as np


class Statistics(object):
    """
    Train/validate loss statistics.
    """
    def __init__(self, logger, num=0, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.start_time = time.time()
        self.num = num
        self.logger = logger

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.num += stat.num

    def accuracy(self):
        return 100.0 * self.n_correct / (self.num + 1e-5)

    def ppl(self):
        return min(np.exp(min(self.get_loss(), 100)), 1000)

    def get_loss(self):
        return self.loss * 1.0 / (self.num + 1e-5)

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, step, total_step):
        t = self.elapsed_time()
        self.logger.info("Step %5d/%5d; acc: %6.2f; loss: %6.2f; "
                         "ppl: %6.2f; %3.0f tok/s; %6.0f s elapsed" %
                          (step, total_step,
                           self.accuracy(),
                           self.get_loss(),
                           self.ppl(),
                           self.n_words / (t + 1e-5),
                           self.elapsed_time()))
        sys.stdout.flush()