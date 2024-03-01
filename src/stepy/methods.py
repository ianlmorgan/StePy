import copy
import matplotlib.pyplot as plt
import numpy as np

from lmfit import minimize, Parameters, report_fit
from matplotlib.cbook import contiguous_regions
from scipy.signal import find_peaks, peak_widths
from scipy.stats import ttest_ind
from lmfit.lineshapes import step

# set tiny to a small number
tiny = 1.0e-15


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


def objective(params, xdata, ydata, form='linear'):
    """ Calculate total residual for fits to several data sets held
    in a 2-D array"""
    ndata = len(xdata)
    resids = []
    for i in range(ndata):
        resids.append(ydata[i] - params[f's{i}_c'] - step(xdata[i],
                                                          amplitude=params[f's{i}_amplitude'],
                                                          center=params[f's{i}_center'],
                                                          sigma=params[f's{i}_sigma'],
                                                          form=form))
    return np.concatenate(resids)


class fit_signal:
    def __init__(self,
                 trace,
                 dt=1,
                 method='ttest',
                 window_length=100,
                 min_threshold=3,
                 max_threshold=5,
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
        # Save initial parameters to use later
        trace = np.asarray(trace)
        self.trace = trace
        self.dt = dt
        self.window_length = window_length
        time = np.arange(dt, trace.size*dt+dt, dt)
        self.time = time
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        # Leaving this here in case I want to add other methods in later
        if method == 'ttest':
            self.score, self.pvalue = rolling_ttest(trace,
                                                    window_length,
                                                    **kwargs)
            # Use the negative log likelihood (p-value)
            self.nlp = -np.log10(self.pvalue)
            # Find contiguous regions greater than min_threshold
            regions = contiguous_regions(self.nlp >= min_threshold)
            # Filter out regions that are shorter than window_length/2
            msk = np.ravel(np.diff(regions) >= window_length/2)
            regions = np.array(regions)[msk]
            # Filter out regions with no pvalues greater than max_threshold
            regions = np.array([r for r in regions if any(
                self.nlp[slice(*r)] >= max_threshold)])
            regions2 = []
            for r in regions:
                r[0] -= int(window_length/2)
                r[1] += int(window_length/2)
                rnlp = self.nlp[slice(*r)]
                # Find peaks that are greater than max threshold
                # and more prominent than min_threshold
                peaks, _ = find_peaks(rnlp,
                                      prominence=min_threshold,
                                      height=max_threshold)
                npeaks = len(peaks)
                # Find split between peaks if there is more than one peak
                if npeaks > 1:
                    # Find valley(s) between first and last peaks
                    # Subtract min_threshold by one
                    peaks2, _ = find_peaks(-rnlp,
                                           prominence=min_threshold)
                    # Add one to ensure we reach the threshold
                    threshold = rnlp[peaks2].min() + 1
                    # Find contiguous regions greater than threshold
                    r2 = contiguous_regions(
                        rnlp >= threshold)
                    # Filter out any regions without peaks greater than max_threshold
                    r2 = np.array([rinner for rinner in r2 if any(
                        rnlp[slice(*rinner)] >= max_threshold)])
                    # msk = np.ravel(np.diff(r2) >= window_length/10)
                    # r2 = np.array(r2)[msk]
                    # Adjust first and last region to have same start and end point
                    # as the original region
                    r2[0][0] = 0
                    r2[-1][1] = len(rnlp)
                    for r3 in r2:
                        regions2.append(r[0]+r3)
                else:
                    regions2.append(r)
            self.regions = regions2

    def plot(self):
        fig, ax = plt.subplots(2, sharex=True,
                               gridspec_kw={'hspace': 0.1,
                                            'height_ratios': [1, 3]})
        ax[0].plot(self.time, self.nlp)
        ax[0].axhline(self.max_threshold, ls='--')
        ax[0].axhline(self.min_threshold, ls='--')
        ax[1].plot(self.time, self.trace, alpha=0.6)
        for i, r in enumerate(self.regions):
            stime = self.time[slice(*r)]
            ax[0].plot(self.time[slice(*r)], self.nlp[slice(*r)])
            if hasattr(self, 'output'):
                params = self.output.params
                yfit = params[f's{i}_c'] + step(stime,
                                                params[f's{i}_amplitude'],
                                                params[f's{i}_center'],
                                                params[f's{i}_sigma'],
                                                form=self.form)
                ax[1].plot(stime, yfit, lw=2)
        ax[1].set_xlabel("Time")
        ax[0].set_ylabel("-$\log_{10}(p)$")
        ax[1].set_ylabel("Signal")
        fig.align_labels()

    def fit(self,
            form='linear',
            min_step_size=-np.inf,
            max_step_size=np.inf,
            fixed_step_width=None,
            max_step_width=None):
        self.form = form
        params = Parameters()
        xdata, ydata = [], []
        nregions = len(self.regions)
        if nregions == 0:
            return self
        for i, r in enumerate(self.regions):
            stime = self.time[slice(*r)]
            strace = self.trace[slice(*r)]
            snlp = self.nlp[slice(*r)]
            peak = np.argmax(snlp)
            xdata.append(stime)
            ydata.append(strace)
            xmin = stime.min()
            xmax = stime.max()
            sigma = xmax-xmin

            # Use data before region to find ymin
            if i == 0:
                y = self.trace[:self.regions[i][0]]
            else:
                y = self.trace[self.regions[i-1][1]:self.regions[i][0]]
            # Handle split peaks by assigning ymin to first value
            if len(y) == 0:
                ymin = strace[0]
            else:
                ymin = np.nanmedian(y)
            # Used data after region to find ymax
            if i == len(self.regions)-1:
                y = self.trace[self.regions[i][1]:]
            else:
                y = self.trace[self.regions[i][1]:self.regions[i+1][0]]
            # Handle split peaks by assigning ymax to last value
            if len(y) == 0:
                ymax = strace[-1]
            else:
                ymax = np.nanmedian(y)
            # Assign starting yvalue for fit
            if i == 0:  # Use ymin for first regions starting yvalue
                params.add(f"s{i}_c", value=ymin)
            else:  # Use previous regions ending position as starting yvalue
                params.add(f"s{i}_c",
                           expr=f"s{i-1}_c+s{i-1}_amplitude",
                           vary=False)
            # Assign amplitude initial value
            params.add(f"s{i}_amplitude",
                       value=ymax-ymin,
                       min=min_step_size,
                       max=max_step_size)
            params.add(f"s{i}_center",
                       value=stime[peak],
                       min=xmin,
                       max=xmax)
            # Assign step widths
            if fixed_step_width is not None:
                params.add(f"s{i}_sigma",
                           value=fixed_step_width,
                           vary=False)
            elif max_step_width is not None:
                params.add(f"s{i}_sigma",
                           value=sigma/2,
                           min=0,
                           max=max_step_width)
            else:
                params.add(f"s{i}_sigma",
                           value=1,
                           min=0,
                           max=sigma)
        output = minimize(objective, params, args=(xdata, ydata, form))
        self.output = output
        # Get parameters from fit
        params = self.output.params.valuesdict()
        # Save parameters to dictionary
        results = {'dwell_time': [],
                   'step_sizes': []}
        for i in range(nregions):
            results['step_sizes'].append(params[f"s{i}_amplitude"])
            if i == nregions-1:
                pass
            else:
                results['dwell_times'].append(
                    params[f"s{i+1}_center"]-params[f"s{i}_center"]-params[f"s{i}_sigma"])
        self.results = results
        return self

    def print_fit_report(self):
        if hasattr(self, "output"):
            report_fit(self.output)


def original_fit(self,
                 form='linear',
                 min_step_size=-np.inf,
                 max_step_size=np.inf,
                 fixed_step_width=None,
                 max_step_width=np.inf):
    """Fit the trace based on identified step regions.

    Args:
        form (str, optional): Type of step function to send to lmfits StepModel.
        Options are 'linear', 'atan', 'erf', and 'logistic'. Defaults to 'linear'.
        min_step_size (float, optional): Minimum step size when fitting. Default is -inf.
        max_step_size (number, optional): Maximum step size when fitting. Default is inf.
        fixed_step_width (float, optional): Fixed step width when fitting. Default is None.
        max_step_width (float, optional): Maximum step width when fitting. Default is None.
    """
    self.form = form
    params = Parameters()
    xdata = []
    ydata = []
    for i, r in enumerate(self.regions):
        stime = self.time[slice(*r)]
        strace = self.trace[slice(*r)]
        xdata.append(stime)
        ydata.append(strace)
        regions = contiguous_regions(
            self.pvalue[r[0]:r[1]] > self.min_threshold)
        xmin = stime[regions[0][-1]]
        xmax = stime[regions[-1][0]]
        ymin = np.median(strace[slice(*regions[0])])
        ymax = np.median(strace[slice(*regions[-1])])
        sigma = (xmax-xmin)/2
        if i == 0:
            params.add(f"s{i}_c", value=ymin)
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
                       value=sigma/2,
                       min=0,
                       max=2*sigma)
        model += step
    self.output = model.fit(self.trace, params, x=self.time)
    return self


""" old code not used 
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
"""
