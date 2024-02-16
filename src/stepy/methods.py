import matplotlib.pyplot as plt
import numpy as np

from lmfit.models import ConstantModel, StepModel
from matplotlib.cbook import contiguous_regions
from scipy.stats import ttest_ind


def rolling_ttest(trace,
                  window_length=100,
                  **kwargs):
    """Calculate a rolling ttest metric.

    Args:
        trace (array_like): A one-dimensional array of a single-molecule signal.
        window_length (int, optional): Length of window. Defaults to 100.

    Returns:
        (tscore,pvalue): Tscore and pvalue for each point from trace.
    """
    # Adds reflected values on either side of trace to avoid edge effects
    a = np.pad(trace, window_length, 'reflect')
    # Creates a sliding window view
    a = np.lib.stride_tricks.sliding_window_view(a, window_length)[:-1]
    tscore, pvalue = ttest_ind(a[window_length:].T,  # all windows before point
                               a[:-window_length].T,  # all windows after point
                               **kwargs)
    return tscore, pvalue


class fit_signal:
    def __init__(self,
                 trace,
                 dt=1,
                 window_length=100,
                 method='ttest',
                 min_threshold=0.005,
                 max_threshold=0.05,
                 **kwargs) -> None:
        """Calculate fit signal

        Args:
            trace (array-like): Time series of measurement values.
            dt (int, optional): Time step between measurement values. Defaults to 1.
            window_length (int, optional): Length of window to use. Defaults to 100.
            method (str, optional): Method to identify steps. Defaults to 'ttest'.
            min_threshold (float, optional): Minimum pvalue threshold. Defaults to 0.005.
            max_threshold (float, optional): Maximum pvalue threshold. Defaults to 0.05.
        """
        self.trace = trace
        time = np.arange(dt, trace.size*dt+dt, dt)
        self.time = time
        if method == 'ttest':
            self.score, self.pvalue = rolling_ttest(trace,
                                                    window_length,
                                                    **kwargs)
            regions = contiguous_regions(self.pvalue < max_threshold)
            # Filter out regions that are shorter than window_length
            msk = np.ravel(np.diff(regions) >= window_length)
            regions = np.array(regions)[msk]
            # Filter out regions with no pvalues greater than min_threshold
            regions = np.array([r for r in regions if any(
                self.pvalue[slice(*r)] < min_threshold)])
            self.regions = regions

    def plot(self):
        fig, ax = plt.subplots(2, sharex=True,
                               gridspec_kw={'hspace': 0.1,
                                            'height_ratios': [1, 3]})
        ax[0].semilogy(self.time, self.pvalue)
        for r in self.regions:
            ax[0].semilogy(self.time[slice(*r)], self.pvalue[slice(*r)])
        ax[1].plot(self.time, self.trace)
        if hasattr(self, "output"):
            ax[1].plot(self.time, self.trace + self.output.residual)
        ax[1].set_xlabel("Time")
        ax[0].set_ylabel("P-value")
        ax[1].set_ylabel("Signal")
        fig.align_labels()

    def fit(self,
            form='linear',
            min_step_size=-np.inf,
            max_step_size=np.inf,
            fixed_step_width=None,
            max_step_width=None):
        """Fit the trace based on identified step regions.

        Args:
            form (str, optional): Type of step function to send to lmfits StepModel.
            Options are 'linear', 'atan', 'erf', and 'logistic'. Defaults to 'linear'.
            min_step_size (float, optional): Minimum step size when fitting. Default is -inf.
            max_step_size (number, optional): Maximum step size when fitting. Default is inf.
            fixed_step_width (float, optional): Fixed step width when fitting. Default is None.
            max_step_width (float, optional): Maximum step width when fitting. Default is None.
        """
        time, trace = self.time, self.trace
        model = ConstantModel()
        params = model.make_params(c=trace[:10].mean())
        for i, r in enumerate(self.regions):
            stime = time[slice(*r)]
            strace = trace[slice(*r)]
            xmax, xmin = max(stime), min(stime)
            ymax, ymin = max(strace), min(strace)
            sigma = (xmax-xmin)/2
            step = StepModel(prefix=f"s{i}_", form=form)
            params.add(f"s{i}_amplitude",
                       value=ymax-ymin,
                       min=min_step_size,
                       max=max_step_size)
            params.add(f"s{i}_center", value=xmin + sigma)
            if fixed_step_width is not None:
                params.add(f"s{i}_sigma",
                           value=fixed_step_width,
                           vary=False)
            elif max_step_width is not None:
                params.add(f"s{i}_sigma",
                           value=sigma,
                           min=0,
                           max=max_step_width)
            else:
                params.add(f"s{i}_sigma",
                           value=sigma,
                           min=0,
                           max=2*sigma)
            model += step
        self.output = model.fit(trace, params, x=time)

    @property
    def results(self):
        params = self.output.params.valuesdict()
        results = {"step_height": [], "step_width": [], "dwell_time": [],
                   "step_rate": [], "step_start": []}
        for i, _ in enumerate(self.regions):
            results["step_height"].append(params[f"s{i}_amplitude"])
            results["step_width"].append(params[f"s{i}_sigma"])
            if i == 0:
                results["dwell_time"].append(0)
            else:
                dwell_time = params[f"s{i}_center"] -  \
                    params[f"s{i-1}_center"] - \
                    params[f"s{i-1}_sigma"]
                results["dwell_time"].append(dwell_time)
            results["step_rate"].append(params[f"s{i}_amplitude"] /
                                        params[f"s{i}_sigma"])
            results["step_start"].append(params[f"s{i}_center"])
        return results
