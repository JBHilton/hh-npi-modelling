'''Module for additional computations required by the model'''
from numpy import (
    arange, diag, isnan, ix_,
    shape, sum, where, zeros)
from numpy import int64 as my_int
import pdb
from scipy.sparse import csc_matrix as sparse
from model.subsystems import subsystem_key


def build_external_import_matrix(household_population, FOI):
    '''Gets sparse matrices containing rates of external infection in a
    household of a given type'''

    row = household_population.inf_event_row
    col = household_population.inf_event_col
    inf_class = household_population.inf_event_class
    total_size = len(household_population.which_composition)

    matrix_shape = (total_size, total_size)

    Q_ext = sparse(matrix_shape,)

    vals = FOI[row, inf_class]
    Q_ext += sparse((vals, (row, col)), shape=matrix_shape)

    diagonal_idexes = (arange(total_size), arange(total_size))
    S = Q_ext.sum(axis=1).getA().squeeze()
    Q_ext += sparse((-S, diagonal_idexes))

    return Q_ext


def build_external_import_matrix_SEPIRQ(
        household_population, FOI_pro, FOI_inf):
    '''Gets sparse matrices containing rates of external infection in a
    household of a given type'''

    row = household_population.inf_event_row
    col = household_population.inf_event_col
    inf_class = household_population.inf_event_class
    total_size = len(household_population.which_composition)

    # Figure out which class gets infected in this transition
    p_vals = FOI_pro[row, inf_class]
    i_vals = FOI_inf[row, inf_class]

    matrix_shape = (total_size, total_size)
    Q_ext_p = sparse(
        (p_vals, (row, col)),
        shape=matrix_shape)
    Q_ext_i = sparse(
        (i_vals, (row, col)),
        shape=matrix_shape)

    diagonal_idexes = (arange(total_size), arange(total_size))
    S = Q_ext_p.sum(axis=1).getA().squeeze()
    Q_ext_p += sparse((-S, diagonal_idexes))
    S = Q_ext_i.sum(axis=1).getA().squeeze()
    Q_ext_i += sparse((-S, diagonal_idexes))

    return Q_ext_p, Q_ext_i


class RateEquations:
    '''This class represents a functor for evaluating the rate equations for
    the model with no imports of infection from outside the population. The
    state of the class contains all essential variables'''
    # pylint: disable=invalid-name
    def __init__(self,
                 model_input,
                 household_population,
                 import_model,
                 epsilon=1.0):

        self.compartmental_structure = \
            household_population.compartmental_structure
        self.no_compartments = subsystem_key[self.compartmental_structure][1]
        self.household_population = household_population
        self.epsilon = epsilon
        self.Q_int = household_population.Q_int
        self.composition_by_state = household_population.composition_by_state
        self.states_sus_only = \
            household_population.states[:, ::self.no_compartments]
        self.s_present = where(self.states_sus_only.sum(axis=1) > 0)[0]
        self.states_new_cases_only = \
            household_population.states[
                :, model_input.new_case_compartment::self.no_compartments]
        self.epsilon = epsilon
        self.inf_compartment_list = \
            subsystem_key[self.compartmental_structure][2]
        self.no_inf_compartments = len(self.inf_compartment_list)
        self.import_model = import_model
        self.ext_matrix_list = []
        self.inf_by_state_list = []
        for ic in range(self.no_inf_compartments):
            self.ext_matrix_list.append(
                diag(model_input.sus).dot(
                    model_input.k_ext).dot(
                        diag(model_input.inf_scales[ic])))
            self.inf_by_state_list.append(household_population.states[
                :, self.inf_compartment_list[ic]::self.no_compartments])

    def __call__(self, t, H):
        '''hh_ODE_rates calculates the rates of the ODE system describing the
        household ODE model'''
        Q_ext = self.external_matrices(t, H)
        if (H < 0).any():
            # pdb.set_trace()
            H[where(H < 0)[0]] = 0
        if isnan(H).any():
            # pdb.set_trace()
            raise ValueError('State vector contains NaNs at t={0}'.format(t))
        dH = (H.T * (self.Q_int + Q_ext)).T
        return dH

    def external_matrices(self, t, H):
        FOI = self.get_FOI_by_class(t, H)
        return build_external_import_matrix(
            self.household_population,
            FOI)

    def get_FOI_by_class(self, t, H):
        '''This calculates the age-stratified force-of-infection (FOI) on each
        household composition'''
        # Average number of each class by household
        denom = H.T.dot(self.composition_by_state)

        FOI = self.states_sus_only.dot(diag(self.import_model.cases(t)))

        for ic in range(self.no_inf_compartments):
            states_inf_only = self.inf_by_state_list[ic]
            inf_by_class = zeros(shape(denom))
            inf_by_class[denom > 0] = (
                H.T.dot(states_inf_only)[denom > 0]
                / denom[denom > 0]).squeeze()
            FOI += self.states_sus_only.dot(
                    diag(self.ext_matrix_list[ic].dot(
                        self.epsilon * inf_by_class.T)))

        return FOI


class SIRRateEquations(RateEquations):
    @property
    def states_inf_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 2::self.no_compartments]


class SEIRRateEquations(RateEquations):
    @property
    def states_exp_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_inf_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 3::self.no_compartments]


class SEPIRRateEquations(RateEquations):
    @property
    def states_exp_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_pro_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_inf_only(self):
        return self.household_population.states[:, 3::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 4::self.no_compartments]


class SEPIRQRateEquations(RateEquations):
    def __init__(self,
                 model_input,
                 household_population,
                 import_model,
                 epsilon=1.0):
        super().__init__(
            model_input,
            household_population,
            import_model,
            epsilon)

        self.iso_pos = 5 + self.no_compartments * \
            arange(model_input.no_age_classes)
        self.states_iso_only = \
            household_population.states[:, self.iso_pos]

        self.iso_method = model_input.iso_method

        if self.iso_method == "int":
            total_iso_by_state = self.states_iso_only.sum(axis=1)
            self.no_isos = (total_iso_by_state == 0)
            self.isos_present = (total_iso_by_state > 0)
            self.ad_prob = model_input.ad_prob
        else:
            # If we are isolating externally, remove quarantined from inf
            # compartment list
            self.inf_compartment_list = [2, 3]
            self.no_inf_compartments = len(self.inf_compartment_list)
            self.ext_matrix_list = []
            self.inf_by_state_list = []
            for ic in range(self.no_inf_compartments):
                self.ext_matrix_list.append(
                    diag(model_input.sus).dot(
                        model_input.k_ext).dot(
                            diag(model_input.inf_scales[ic])))
                self.inf_by_state_list.append(household_population.states[
                    :, self.inf_compartment_list[ic]::self.no_compartments])

    def get_FOI_by_class(self, t, H):
        '''This calculates the age-stratified force-of-infection (FOI) on each
        household composition'''

        FOI = self.states_sus_only.dot(diag(self.import_model.cases(t)))

        if self.iso_method == 'ext':
            # Under ext. isolation, we need to take iso's away from total
            # household size
            denom = H.T.dot(
                self.composition_by_state - self.states_iso_only)
            for ic in range(self.no_inf_compartments):
                states_inf_only = self.inf_by_state_list[ic]
                inf_by_class = zeros(shape(denom))
                inf_by_class[denom > 0] = (
                    H.T.dot(states_inf_only)[denom > 0]
                    / denom[denom > 0]).squeeze()
                FOI += self.states_sus_only.dot(
                        diag(self.ext_matrix_list[ic].dot(
                            self.epsilon * inf_by_class.T)))
        else:
            # Under internal isoltion, we scale down contribution to infections
            # of any houshold containing Q individuals
            denom = H.T.dot(self.composition_by_state)

            for ic in range(self.no_inf_compartments):
                states_inf_only = self.inf_by_state_list[ic]
                inf_by_class = zeros(shape(denom))
                index = (denom > 0)
                inf_by_class[index] = (
                    (
                        H[where(self.no_isos)[0]].T.dot(
                            states_inf_only[where(self.no_isos)[0], :])[index]
                        + (1 - self.ad_prob)
                        * H[where(self.isos_present)[0]].T.dot(
                            states_inf_only[where(self.isos_present)[0], :])
                    )[index]
                    / denom[index]).squeeze()
                FOI += self.states_sus_only.dot(
                    diag(self.ext_matrix_list[ic].dot(
                        self.epsilon * inf_by_class.T)))

        return FOI

    @property
    def states_exp_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_pro_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_inf_only(self):
        return self.household_population.states[:, 3::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 4::self.no_compartments]


class SEDURRateEquations(RateEquations):
    @property
    def states_exp_only(self):
        return self.household_population.states[:, 1::self.no_compartments]

    @property
    def states_det_only(self):
        return self.household_population.states[:, 2::self.no_compartments]

    @property
    def states_undet_only(self):
        return self.household_population.states[:, 3::self.no_compartments]

    @property
    def states_rec_only(self):
        return self.household_population.states[:, 4::self.no_compartments]
