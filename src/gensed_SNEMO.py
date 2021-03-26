"""gensed_SNEMO.py - Use the BYOSED framework to use SNEMO models for SNANA sims.

Written by Benjamin Rose, benjamin.rose@duke.edu
Fall 2020 through Spring 2021.

Developed with: 
* Python 3.7.8
* sklearn 0.23.2
* available in /project2/rkessler/PRODUCTS/miniconda/envs/snemo

Built on top of: 
BOYSED Paper - https://ui.adsabs.harvard.edu/abs/2020arXiv201207811P/abstract
BOYSED Docs - https://byosed.readthedocs.io/en/latest/
SNEMO paper - https://ui.adsabs.harvard.edu/abs/2018ApJ...869..167S/abstract
SNEMO parameter distributions paper - Dixon 2021, Dissertation Ch. 4 - Contact Sam Dixon or Ben Rose

Dev tips:
* Need to flush stdout to work with batch jobs/slurm, `print(..., flush=True)`.

TODO list:
- [ ] document params file
- [ ] can I make params file optional?
- [ ] find a default home for the KDE files on Midway
- [ ] find a way to get SNEMO model name into SNANA metadata.
"""

import sys
import os
import pickle
import copy


def raise_error(e):
    """Errors need to fail and print messages rather than causing an SNANA failure.

    If SNANA fails, no python error is printed. Justin Pierel set up this
    method of error handling and it works. There may or may not be a better
    way.
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()
    # Need to flush stdout to work with batch jobs/slurm
    print("Python Error :", e, flush=True)
    print("gensed_SNEMO.py, line number: %i" % exc_tb.tb_lineno, flush=True)
    print_err()


def print_err():
    print(
        """
			   ______
			 /		x	  \\
			/	 --------<	ABORT Python on Fatal Error.
		__ /  _______/
/^^^^^^^^^^^^^^/  __/
\________________/
				""",
        flush=True,
    )
    raise RuntimeError


try:
    import yaml
    import sncosmo
    import numpy as np
except Exception as e:
    # Bad practice, but we need to catch all python errors. See `raise_error()` documentation.
    raise_error(e)


mask_bit_locations = {"verbose": 1, "dump": 2}


# Variable names fit SNANA conventions rather than Python.
class gensed_SNEMO:
    """Generate a SNEMO SNIa SED for SNANA via the BYOSED framework.

    SNANA required methods: `fetchSED_NLAM`, `fetchSED_LAM`, `fetchSED_SNEMO`, `fetchParNames_SNEMO`,
        `fetchNParNames_SNEMO`, `fetchParVals_SNEMO_4SNANA`

    Class specific helper methods: `_update_params`, `_reset_cache`, `_update_cache`, `_normalize_phase`

    Most of the work is done in `fetchSED_SNEMO`.

    Parameters
    ----------
    PATH_VERSION: str
        Path to settings/parameter file. Path needs to point to a folder
        containing "SNEMO.params" with approprate yaml settings/options.
    OPTMASK: int
        Bit flag options, following SNANA conventions.
    ARGLIST: str
        a comma-separated list. Typically empty but also can contrain items like `RANSEED 12945`.
        CURRENTLY UNUSED.
    HOST_PARAM_NAMES
        CURRENTLY UNUSED. Names of host paramters.


    Attributes
    ---------
    # Settings
    verbose: int or bool
        Internal status reflecting options, regarding amount of output, given to SNANA.

    dump
        CURRENTLY UNUSED. Internal status reflecting options given to SNANA.
    host_param_names
        CURRENTLY UNUSED. Internal atribute, derived from parameter `HOST_PARAM_NAME`.

    PATH_VERSION: str
        Interanl storage of path to settings file. Stores values passed in via `PATH_VERSION` parameters.
    paramfile: str
        Path+filename of settings file.
    params_file_contents: dict

    # Model values
    model:
        The SNCosmo model instance. Defaults to SNCosmo.source, class=SNEMOSource, name='snemo15'.
        Model can be selected via `paramfile`.
    parameter_values: dict
        Dictionary of parameter names (as keys) and values:
        `dict(zip(self.model.param_names[2:], self.model.parameters[2:]))`

    # SNEMO parameter KDE model
    KDE
        KDE of SNEMO parameter correlations.
    rot
        Part of KDE data. Read KDE paper for details.
    eigenvals
        Part of KDE data. Read KDE paper for details.

    # Information from SNANA
    self.external_id: int, default=-1
        A copy of the latest SNANA `external_id` used with `fetchSED_SNEMO()`. A negative value
        is stored prior if SNANA has not yet called `fetchSED_SNEMO()`.

    # Internal data
    cache: dict
        A cache of SED at a give phase. `gensed_SNEMO._normalize_phase()` normalize float value phases.
        Therefore evolution on small time scales, < 15 mins, is ignored. Read documentation for
        `_normalize_phase` for more details.

    # Attributes used from SNANA required getter-methods
    wave: list
        Wavelengths of the SED model: `model.source._wave`
    wavelen: int
        Length of `wave` array: `len(self.wave)`
    parameter_names: list
        List of model parameter names. Not the same as SNCosmo's `model.param_names` (no need
        from z or t0): `model.param_names[2:]`
    """

    def __init__(self, PATH_VERSION, OPTMASK, ARGLIST, HOST_PARAM_NAMES):
        try:
            # Setup infromation from SNANA
            ##############################

            # Process SNANA options bit flags.
            self.verbose = OPTMASK & (1 << mask_bit_locations["verbose"]) > 0
            self.verbose = True
            self.dump = OPTMASK & (1 << mask_bit_locations["dump"]) > 0

            self.host_param_names = [x.upper() for x in HOST_PARAM_NAMES.split(",")]

            self.PATH_VERSION = os.path.expandvars(os.path.dirname(PATH_VERSION))

            if os.path.exists(os.path.join(self.PATH_VERSION, "SNEMO.params")):
                self.paramfile = os.path.join(self.PATH_VERSION, "SNEMO.params")
            elif os.path.exists(os.path.join(self.PATH_VERSION, "SNEMO.PARAMS")):
                self.paramfile = os.path.join(self.PATH_VERSION, "SNEMO.PARAMS")
            else:
                raise RuntimeError(
                    "param file %s not found!"
                    % os.path.join(self.PATH_VERSION, "SNEMO.params")
                )

            self.params_file_contents = yaml.load(
                open(self.paramfile), Loader=yaml.FullLoader
            )

            # Setup SNEMO & SNCOSMO
            #######################
            if self.params_file_contents["SNEMO_model"] in [2, 7, 15]:
                source_name = "snemo{}".format(self.params_file_contents["SNEMO_model"])
                self.model = sncosmo.Model(source=source_name)
                if self.verbose:
                    # Need to flush stdout to work with batch jobs/slurm
                    print(
                        "Running with SNCosmo model {}.".format(source_name), flush=True
                    )
            else:
                if self.verbose:
                    print(
                        "SNEMO_model in params file (",
                        self.params_file_contents["SNEMO_model"],
                        ") is not an SNEMO model (2, 7, 15). Defaulting to SNEMO15.",
                        flush=True,
                    )
                self.model = sncosmo.Model(source="snemo15")

            kde_path = os.path.expandvars(
                os.path.join(
                    self.params_file_contents["KDE_FOLDER"],
                    "{}_KDE_published.pkl".format(self.model.source.name),
                )
            )

            with open(kde_path, "rb") as f:
                self.KDE, self.rot, self.eigenvals, _ = pickle.load(f)

            # set a non-positive integer, SNANA uses this as a counting number
            # this works around the buggy nature of `new_event` in `fetchSED_SNEMO()`
            self.external_id = -1
            # initiate an empty cache.
            # SNANA requests the full SED per filter it looks at. Don't regenerate the SED for
            # the same SN at the same phase just because SNANA is looking with a new filter
            self.cache = {}

            # SNANA required variables
            self.wave = self.model.source._wave
            # https://stackoverflow.com/questions/7271385/how-do-i-combine-two-lists-into-a-dictionary-in-python
            self.parameter_values = dict(
                zip(self.model.param_names[2:], self.model.parameters[2:])
            )

            # Not being used outside of SNANA method.
            # Currently being used like a c-programmer
            # Is this faster then recomputing each time the methods are called?
            self.wavelen = len(self.wave)
            # don't need SNCosmo's z and t0
            self.parameter_names = self.model.param_names[2:]

            if self.verbose:
                # Need to flush stdout to work with batch jobs/slurm
                print(
                    "Initializing SNComso model parameters with",
                    self.parameter_values,
                    flush=True,
                )

        except Exception as e:
            raise_error(e)

    # MAIN METHOD
    # name set by BOYSED
    ###################
    def fetchSED_SNEMO(
        self, trest, maxlam=5000, external_id=1, new_event=1, hostpars=""
    ):
        """
        Returns the flux at every wavelength, for a given phase.

        Parameters
        ----------
        trest : float
            The rest frame phase at which to calculate the flux
        maxlam : int
            A maximum number of wavelength bins. If your wavelength
            vector is longer than this number (default is arbitrary),
            program should abort
        external_id : int
            ID for SN
        new_event : int
            BUGGY (December 2020)!! Should be: 1 if new event, 0 if same SN.
            This param is convienent but redudent to a change in external_id.
            Don't use, even though I hear it was fixed by Justin Pierel.
        hostpars : str
            Comma separated list of host parameters

        Returns
        -------
        A list of length self.wavelen containing the flux at
        every wavelength in self.wave, at the phase trest
        """

        try:
            print("PHASE: ", trest, flush=True)
            if len(self.wave) > maxlam:
                raise RuntimeError(
                    "Your wavelength array cannot be larger than %i but is %i"
                    % (maxlam, len(self.wave))
                )

            # Update parmaters if a new SN
            # `new_event` can be buggy and unreliable need to use a change in exertnal_id
            if self.external_id != external_id:
                if self.verbose:
                    # Need to flush stdout to work with batch jobs/slurm
                    print("New SN. Now simulating SN ", external_id, flush=True)
                self.external_id = external_id
                self._reset_cache()
                self._update_params()
                self.model.set(**self.parameter_values)
                if self.verbose:
                    print("Updated params to:", self.parameter_values, flush=True)

            # Get flux
            ##########
            try:
                # skipping cache, to test a bug
                raise KeyError
                # Check cache first
                # Read docs for `_update_cache()` for why we are using a cache
                flux_at_all_wavelengths_for_trest = self.cache[
                    self._normalize_phase(trest)
                ]
                if self.verbose:
                    # Need to flush stdout to work with batch jobs/slurm
                    print("Got SED at phase", trest, " from cache.", flush=True)
            except KeyError:
                # Calculation for flux at self.wave wavelengths for phase trest
                # SNCosmo is observer frame and SNANA gives restframe.
                # However, we are using SNCosmo at z=0 so observer is rest!
                # SNCosmo returns ergs / s / cm^2 / Angstrom, same as expected by SNANA (`grep erg genmag_PySEDMODEL.c`).
                flux_at_all_wavelengths_for_trest = self.model.flux(
                    wave=self.wave, time=trest
                )
                self._update_cache(trest, flux_at_all_wavelengths_for_trest)
            if flux_at_all_wavelengths_for_trest is not None:
                # Need to flush stdout to work with batch jobs/slurm
                print(
                    "SNEMO INFO: ",
                    np.max(flux_at_all_wavelengths_for_trest),
                    np.min(flux_at_all_wavelengths_for_trest),
                    trest,
                    self.parameter_values,
                    flush=True,
                )
            else:
                # We are hitting a bug sometimes. Returning None results in SNANA/BOYSED to seg-fault
                print("FLUX IS NONE", flush=True)
            if np.isnan(
                np.dot(
                    flux_at_all_wavelengths_for_trest, flux_at_all_wavelengths_for_trest
                )
            ):
                # fast way to check if any is np.nan
                # https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy
                print("FLUX HAS NaN", flush=True)
            return list(flux_at_all_wavelengths_for_trest)

        except Exception as e:
            raise_error(e)

    # HELPER METHODS
    ################
    def _update_params(self):
        """Update SNEMO parameters (create a new SN) from a KDE.

        KDE location is defined in SNEMO.params file via an absolute
        path via the `KDE_FOLDER` keyword.

        Derived from work by Sam Dixon,
        https://github.com/sam-dixon/snemo_generator/blob/master/snemo_gen/kde.py
        """
        if self.verbose:
            # Need to flush stdout to work with batch jobs/slurm
            print("Updating paramters", flush=True)

        unscaled_samples = self.KDE.sample(1)
        samples = (self.rot @ np.diag(np.sqrt(self.eigenvals)) @ unscaled_samples.T).T
        if self.verbose:
            # Need to flush stdout to work with batch jobs/slurm
            print("KDE sample values", samples, flush=True)

        # KDE.sample returns a list, remove the outer list here with samples[0]

        # convert MB from KDE to c
        if self.params_file_contents["MB_KDE"]:
            # the default KDE is detla-MB, As, c1, .... self.parameter_names has c0 not MB.
            kde_param_names = ["MB", *self.model.source.param_names[1:]]

            # if KDE is givin in MB, use SNCosmo to convert to c0: then update self.paramer_values
            # need to copy, but temp_model.set() is leaking back to original
            temp_model = copy.copy(self.model)
            # Set all the parameters sampled above in the temp model.
            # Don't set SNCosmo's c0, since KDE has delta-MB.
            temp_model.set(**dict(zip(kde_param_names[1:], samples[0, 1:])))
            # use `model.set_source_peakabsmag` to go from KDE M_B to SNCosmo c0.
            # `set_source_peakabsmag:`` 'absolute magnitude undefined when z<=0.'
            # set z to 10 pc. Magic SNCosmo number.
            temp_model.set(z=0.000000002369)  # needs to be >10 parsecs
            # NOTE: -19.1 is semi-arbitrary, it centers MB around 0.
            # It is a convention of the KDE, nothing more.
            # KDE M_B is actually a delta-M_B (ie M_B + 19.1)
            temp_model.set_source_peakabsmag(samples[0, 0] - 19.1, "bessellb", "ab")
            if self.verbose:
                # Need to flush stdout to work with batch jobs/slurm
                print("M_B set to", samples[0, 0] - 19.1, flush=True)
            temp_model.set(z=0)  # no idea if this is needed.
            # model.param_names starts is z, t0, c0, As ...., samples is MB, As, ...
            # pull values from temp_model_values since `samples` uses delta-MB not c0.
            # `self.parameter_values` does not contain z or t0.
            for key, val in zip(temp_model.param_names[2:], temp_model.parameters[2:]):
                self.parameter_values[key] = val
        else:
            # Need to update self.paramer_values and not overwrite
            kde_param_names = ["c0", *self.model.source.param_names[1:]]
            for key, val in zip(kde_param_names, samples[0]):
                self.parameter_values[key] = val

        if self.verbose:
            # Need to flush stdout to work with batch jobs/slurm
            print("Parameters updated to: ", self.parameter_values, flush=True)

    def _reset_cache(self):
        """Reset the SED cache"""
        if self.verbose:
            # Need to flush stdout to work with batch jobs/slurm
            print("Resetting cache", flush=True)
        self.cache = {}

    def _update_cache(self, phase, sed):
        """Update the SED cache.

        Justin Pierel, Mar 15, 2021 1:07pm WFIRST Slack Message
        SNANA basically does a double for loop. The outer loop is filters, the
        inner loop is phases. That means say it was B-band phase -5, 0, 5, ... then V-band
        phase -5, 0, 5 ... then R-band phase -5, 0, 5 each time phase==0 you’re passing back
        the exact same thing (SNEMO at all wavelengths at phase 0).
        For example, it builds all the B-band LC then the V-band.

        Parameters
        ----------
        phase: float
            Restframe phase. Expected to be `trest` that SNANA past to `fetchSED_SNEMO()`
        sed: self.model.flux
            Expected to be the returned value of a `self.model.flux()` call.
        """
        if self.verbose:
            # Need to flush stdout to work with batch jobs/slurm
            print("Adding phase=", phase, " to cache", flush=True)
        self.cache[self._normalize_phase(phase)] = sed

    def _normalize_phase(self, phase):
        """normalize the phase float value, mostly for cache.

        1.0e-2 days is 14.4 mins. This normaliziation process may add/subtract up to 14.4 mins.

        Parameters
        ----------
        phase: float
                        Restframe phase. Expected to be `trest` that SNANA past to `fetchSED_SNEMO()`

        Returns
        -------
        float
                        a normalized verions of the float
        """
        return round(phase, 2)

    # SNANAN GETTER METHODS
    # don't change names
    ########################
    def fetchSED_NLAM(self):
        """
        Returns the length of the wavelength vector
        """
        try:
            if self.verbose:
                # Need to flush stdout to work with batch jobs/slurm
                print(
                    "Retreaved length of wavelength vector: ", self.wavelen, flush=True
                )
            return self.wavelen
        except Exception as e:
            # Bad practice, but we need to catch all python errors. See `raise_error()` documentation.
            raise_error(e)

    def fetchSED_LAM(self):
        """
        Returns the wavelength vector in erg/s/cm^2/A
        """
        try:
            if self.verbose:
                # Need to flush stdout to work with batch jobs/slurm
                print("Retreaved wavelength vecotr: ", list(self.wave), flush=True)
            return list(self.wave)
        except Exception as e:
            # Bad practice, but we need to catch all python errors. See `raise_error()` documentation.
            raise_error(e)

    def fetchParNames_SNEMO(self):
        """
        Returns the names of model parameters
        """
        try:
            if self.verbose:
                # Need to flush stdout to work with batch jobs/slurm
                print("Retreaved param names: ", self.parameter_names, flush=True)
            return list(self.parameter_names)
        except Exception as e:
            # Bad practice, but we need to catch all python errors. See `raise_error()` documentation.
            raise_error(e)

    def fetchNParNames_SNEMO(self):
        """
        Returns the number of model parameters
        """
        try:
            if self.verbose:
                # Need to flush stdout to work with batch jobs/slurm
                print(
                    "Retreaved number of model parameters: ",
                    len(self.parameter_names),
                    flush=True,
                )
            return len(self.parameter_names)
        except Exception as e:
            # Bad practice, but we need to catch all python errors. See `raise_error()` documentation.
            raise_error(e)

    def fetchParVals_SNEMO_4SNANA(self, varname):
        """
        Returns the value of parameter 'varname'

        Parameters
        ----------
        varname : str
                                        A parameter name from self.parameter_names
        """
        try:
            if self.verbose:
                # Need to flush stdout to work with batch jobs/slurm
                print(
                    "Retreaved param values: ",
                    self.parameter_values[varname],
                    flush=True,
                )
            return self.parameter_values[varname]
        except Exception as e:
            # Bad practice, but we need to catch all python errors. See `raise_error()` documentation.
            raise_error(e)


if __name__ == "__main__":
    # Test with Midway param file
    mySED = gensed_SNEMO(
        "$WFIRST_USERS/brose3/SNEMO_test/", 2, [], "z,AGE,ZCMB,METALLICITY"
    )
    # Test locally
    # mySED = gensed_SNEMO("./data/", 2, [], "z,AGE,ZCMB,METALLICITY")
