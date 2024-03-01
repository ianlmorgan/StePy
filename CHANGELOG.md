0.1.0
* switched to new faster fitting method
* updated how step regions are differentiated
* implemented global fits
* updated how initial parameters are chosen for fits
* annotated the code a bit better
0.0.4
* updated default initial guesses to use the first and last regions between the max and minimum thresholds for each step
* updated the fit2 method to fix the starting height of each step to the final height from the previous step
0.0.3
* added a new fitting method that only fits the region around steps
* updated the initial guesses to use the mean of the first and last 10 points rather than the maximum and minimum heights to avoid spikes that can throw off fits
* updated the plotting to include the new fitting method
* updated the results to include the new fitting method
