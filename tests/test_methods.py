import numpy as np
import unittest

from stepy.methods import rolling_ttest,fit_signal

class TestMethods(unittest.TestCase):
    def setUp(self) -> None:
        #dt = 0.01
        step_size = 2
        np.random.seed(1)
        trace = np.random.normal(size=1000)
        trace[500:] += step_size
        self.trace = trace
    def test_rolling_ttest(self):
        score,pvalue = rolling_ttest(self.trace)
        np.testing.assert_all_close(score,self.trace)
        np.testing.assert_all_close(pvalue,self.trace)
    def test_fit_signal(self):
        fs = fit_signal(self.trace,dt=0.01)
        fs.fit()
        fs.plot()
        