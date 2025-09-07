# Changelog
## unreleased (2025-09-06)
### Added
### Changed
### Fixed
### Deprecated
### Removed
### Security

## 3.5.1 (2025-09-06)
### Added
### Changed
### Fixed
- clear queue before exiting
- clear queue after pausing via ctrl-x
- upper frequency limit for fundamental
- smaller corrections
- optimized imports
- most recent modules in requirements.txt tested
### Deprecated
### Removed
### Security

## 3.5.0 (2024-07-31)
### Added
- CHANGELOG.md 
- new parameter in parameters.py
- plot figure cannot be closed by MATPLOTLIB standard hot keys 
- conda env list provided
### Changed
- setup.py reads from requirements.txt
- typing modified
- noise measurement: peak height > average background + factor * standard deviation
- hot-keys shifted
### Fixed
- updated to current module versions (see requirements.txt)
### Deprecated
### Removed
### Security

## Old Format
2021/08/14 - Ralf A. Timmermann
- Update to version 2.1
    * new hotkeys to change minimum frequency in Fourier spectrum
2021/08/26 - Ralf A. Timmermann
- version 2.2
    * nested pie to display deviation of key from target value, parameter.PIE
    to toggle to previous setup
    * hotkey to reset to initial values
    * inner pie filled with color to indicate correct tuning
    * DEBUG: order peak list by frequency (ascending)
    * DEBUG: utilized time in ms
    * INFO: best results with f_1 instead of f_0
    * import solely modules to be used  
2022/02/26 - Ralf A. Timmermann
- version 2.3
    * f0 and b are derived from combinations of partitials with no common 
    divisor. In ambiguous cases the calculated partials' frequencies  
    (as derived from f0 and b) are compared with those measured and the cost 
    function (l1-norm) for a LASSO regression is minimized.
    * hotkey 'x' to toggle between halt and resume.
    * matplotlib commands are swapped to new superclass MPmatplot in 
    multiProcess_matplot.py that will be started in a proprietary process, 
    variables in dict passed through queue.
    * new plot parameters moved to parameters.py 
    * consumed time per module in DEBUG mode measured in wrapper mytimer
    * FFTaux.py added
2022/04/10 - Ralf A. Timmermann
- version 3.0 (productive)
    * Finding the NMAX highes peaks in the frequency spectrum: prominence 
    for find_peaks disabled, peak heights minus background calculated by 
    utilizing the average interpolated positions of left and right intersection 
    points of a horizontal line at the respective evaluation height.
    * modified global hot keys, such as q is disabled.
2022/07/17 - Ralf A. Timmermann
- version 3.1.0 (productive)
    * slice shift is fixed value in parameter file
    * import absolute paths
    * final L1 minimization on difference between measured and calculated 
    partials utilizing scipy.optimize.minimize through brute
    to determine base-frequency and inharmonicity (toggled in parameters)
2022/07/19 - Ralf A. Timmermann
- version 3.1.1 (productive)
    * Noise level can be adjusted through global hot keys
    * L1 minimization called only if more than 2 measured peaks
    * catching if erroneous input from the keyboard
2022/07/20 - Ralf A. Timmermann
- version 3.1.2 (productive)
    * error correction: Jacobian sign of derivative toggled
    * L1_contours.py added to demonstrate L1 cost function and its Jacobian (
    not needed with brute force) 
2022/07/26 - Ralf A. Timmermann
- version 3.1.3 (productive)
    * slice of inharmonicity factor b is on log10 scale for equidistant grids
    for brute force minimizer
2022/08/07 - Ralf A. Timmermann
- version 3.2 (productive)
    * updated for Python v3.9
    * typing
2022/08/07 - Ralf A. Timmermann
- version 3.2.1 (productive)
    * SLSQP minimizer
2022/08/12 - Ralf A. Timmermann
- version 3.2.2 
    * absoute pitch in pie
2022/08/17 - Ralf A. Timmermann
- version 3.3.1 
    * dublicates removed from the partials list
    * for the minimizer: list of found frequencies is now tagged with 
    the appropriate partials, that makes l1 computation unambiguous and avoids
    local minima
    * code cleansing
2022/09/27 - Ralf A. Timmermann
- version 3.4.1
    * highpass filter parameters outsourced as global parameters
    * Noise measurement with no audio signal (silence) comprising mean and 
    standard deviation per bin - need to be toggled on/off per global hotkey
    (in experimental stage) 
