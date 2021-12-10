'''Class structure describing external importations'''
from abc import ABC
from numpy import exp, ones, zeros
from scipy.interpolate import interp1d

def import_model_from_spec(spec, det):
    text_to_type = {
        'fixed': FixedImportModel,
        'step': StepImportModel,
        'exponential': ExponentialImportModel,
        'care_home': CareHomeImportModel,
    }
    return text_to_type[spec['external_importation']['type']].make_from_spec(
        spec['external_importation'], det)


class ImportModel(ABC):
    '''Abstract class for importation models'''
    def __init__(self,
                no_inf_compartments,
                no_age_classes):
        self.no_inf_compartments = no_inf_compartments
        self.no_age_classes = no_age_classes
        self.no_entries = no_inf_compartments * no_age_classes
    def cases(self, t):     # Cases is a list of import functions
        pass


class NoImportModel(ImportModel):
    def cases(self, t):
        return zeros(self.no_age_classes,)

    @classmethod
    def make_from_spec(cls, spec, det):
        return cls()


class FixedImportModel(ImportModel):
    def __init__(
            self,
            no_inf_compartments,
            no_age_classes,
            import_array):
        '''import_arrays should be a list of arrays. The number of arrays is
        no_inf_compartments and each has length no_age_classes. The jth element of
        the ith array is the rate at which individuals in age class j are infected
        by external cases in infectious compartment i.'''
        super().__init__(no_inf_compartments, no_age_classes)
        self.import_array = import_array

    def cases(self, t):
        return self.import_array

class StepImportModel(ImportModel):
    def __init__(
            self,
            time,
            external_prevalance):       # External prevalence is now a age classes by inf compartments array
        self.prevalence_interpolant = []
        for i in range(self.no_entries):
            self.prevalence_interpolant.append(interp1d(
                time, external_prevalance[i,:],
                kind='nearest',
                bounds_error=False,
                fill_value='extrapolate',
                assume_sorted=True))

    def cases(self, t):
        imports = zeros(no_entries,)
        for i in range(self.no_entries):
            imports[i] = self.prevalence_interpolant[i](t)
        return imports


class ExponentialImportModel(ImportModel):
    def __init__(self, r, det_profile, undet_profile):
        self.r = r
        self.det_profile = det_profile
        self.undet_profile = undet_profile

    @classmethod
    def make_from_spec(cls, spec, det):
        r = float(spec['exponent'])
        alpha = float(spec['alpha'])
        det_profile = alpha * det
        undet_profile = alpha * (ones((10,)) - det)
        return cls(r, det_profile, undet_profile)

    def detected(self, t):
        return exp(self.r * t) * self.det_profile

    def undetected(self, t):
        return exp(self.r * t) * self.undet_profile

class CareHomeImportModel(ImportModel):
    def __init__(
            self,
            time,
            prodromal_prev,
            infected_prev):
        self.prodromal_interpolant = interp1d(
            time, prodromal_prev,
            kind='nearest',
            bounds_error=False,
            fill_value='extrapolate',
            assume_sorted=True)
        self.infected_interpolant = interp1d(
            time, infected_prev,
            kind='nearest',
            bounds_error=False,
            fill_value='extrapolate',
            assume_sorted=True)

    def prodromal(self, t):
        return self.prodromal_interpolant(t)

    def infected(self, t):
        return self.infected_interpolant(t)
