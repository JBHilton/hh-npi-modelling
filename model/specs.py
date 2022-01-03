'''Module containing model specifications'''

from copy import deepcopy
from numpy import arange, array, multiply, ones, zeros
from numpy.random import rand

def draw_random_two_age_SEPIR_specs(spec,
                      AR_pars = [0.25, 0.75, 6],
                      rec_pars = [1, 7],
                      inc_pars = [1, 10],
                      pro_pars = [1, 5],
                      pro_trans_pars = [1, 1],
                      sus_pars = [1, 1],
                      dens_pars = [0, 1],
                      R_pars = [1, 2]):

    rand_spec = deepcopy(spec)

    ''' Start by drawing secondary inf probabilities from uniform distribution,
    with each SITP bounded above by previous one, starting default [0.25, 0.75]'''
    rand_spec['SITP'] = zeros(AR_pars[2],) # AR_pars[2] is max hh size
    prev_AR = AR_pars[1]
    for n in range(AR_pars[2] - 1): # SITP is indexed by hh size minus one
        new_AR = AR_pars[0] + (prev_AR - AR_pars[0]) * rand(1,)
        rand_spec['SITP'][n] = new_AR
        prev_AR = new_AR

     # Rec rate unif, default [1, 7]
    rand_spec['recovery_rate'] = 1 / (
                                        rec_pars[0] +
                                        (rec_pars[1] - rec_pars[0]) *
                                        rand(1,) )
    # Inc rate unif, default [1, 7]
    rand_spec['incubation_rate'] = 1 / (
                                        inc_pars[0] +
                                        (inc_pars[1] - inc_pars[0]) *
                                        rand(1,) )
    # Onset rate rate unif, default [1, 5]
    rand_spec['symp_onset_rate'] = 1 / (
                                        pro_pars[0] +
                                        (pro_pars[1] - pro_pars[0]) *
                                        rand(1,) )
    # Unif prodrome scalings, default <1
    rand_spec['prodromal_trans_scaling'] = multiply( array(pro_trans_pars),
                                            rand(len(pro_trans_pars),) )
    # Unif sus scalings
    unscaled_sus = multiply( array(sus_pars), rand(len(sus_pars),) )
    # Set sus of most sus class to 1, all others <1
    rand_spec['sus'] = unscaled_sus/unscaled_sus.max()
    # Unif density expo, default [0, 1]
    rand_spec['density_expo'] = \
        dens_pars[0] +(dens_pars[1] - dens_pars[0]) * rand(1,)
    if spec['fit_method'] == 'R*':
        # Unif R*, default [1, 2]
        rand_spec['R*'] = \
            R_pars[0] +(R_pars[1] - R_pars[0]) * rand(1,)

    return rand_spec

TRANCHE1_SITP = array([0.184,0.162,0.148,0.137,0.129])
TRANCHE2_SITP = array([0.345,0.274,0.230,0.200,0.177])
TRANCHE3_SITP = array([0.302,0.238,0.198,0.171,0.151])
TRANCHE4_SITP = array([0.290,0.233,0.197,0.173,0.155])

LATENT_PERIOD = 0.2 * 5.8   # Time in days from infection to infectiousness
PRODROME_PERIOD = 0.8 * 5.8 # Time in days from infectiousness to symptom onset
SYMPTOM_PERIOD = 5          # Time in days from symptom onset to recovery

PRODROME_SCALING = 3        # Relative intensity of transmission during prodrome

TWO_AGE_SIR_SPEC = {
    'compartmental_structure': 'SIR', # This is which subsystem key to use
    'SITP': TRANCHE2_SITP,                     # Secondary inf probability
    'R*': 1.1,                      # Household-level reproduction number
    'recovery_rate': 1 / (LATENT_PERIOD +
                          PRODROME_PERIOD +
                          SYMPTOM_PERIOD),           # Recovery rate
    'sus': array([1,1]),          # Relative susceptibility by
                                  # age/vulnerability class
    'fit_method' : 'R*'
}

TWO_AGE_SIR_SPEC_FOR_FITTING = {
    'compartmental_structure': 'SIR', # This is which subsystem key to use
    'SITP': TRANCHE2_SITP,                     # Secondary inf probability
    'recovery_rate': 1 / (LATENT_PERIOD +
                          PRODROME_PERIOD +
                          SYMPTOM_PERIOD),           # Recovery rate
    'sus': array([1,1]),          # Relative susceptibility by
                                  # age/vulnerability class
    'fit_method' : 'EL'
}

SINGLE_AGE_SEIR_SPEC = {
    'compartmental_structure': 'SEIR', # This is which subsystem key to use
    'SITP': TRANCHE2_SITP,                     # Secondary inf probability
    'R*': 1.1,                      # Household-level reproduction number
    'recovery_rate': 1 / (PRODROME_PERIOD +
                          SYMPTOM_PERIOD),           # Recovery rate
    'incubation_rate': 1 / LATENT_PERIOD,         # E->I incubation rate
    'sus': array([1]),          # Relative susceptibility by
                                  # age/vulnerability class
    'fit_method' : 'R*'
}

SINGLE_AGE_SEIR_SPEC_FOR_FITTING = {
    'compartmental_structure': 'SEIR', # This is which subsystem key to use
    'SITP': TRANCHE2_SITP,                     # Secondary inf probability
    'recovery_rate': 1 / (PRODROME_PERIOD +
                          SYMPTOM_PERIOD),           # Recovery rate
    'incubation_rate': 1 / LATENT_PERIOD,         # E->I incubation rate
    'sus': array([1]),          # Relative susceptibility by
                                  # age/vulnerability class
    'fit_method' : 'EL'
}

TWO_AGE_SEIR_SPEC = {
    'compartmental_structure': 'SEIR', # This is which subsystem key to use
    'SITP': TRANCHE2_SITP,                     # Secondary inf probability
    'R*': 1.1,                      # Household-level reproduction number
    'recovery_rate': 1 / (PRODROME_PERIOD +
                          SYMPTOM_PERIOD),           # Recovery rate
    'incubation_rate': 1 / LATENT_PERIOD,         # E->I incubation rate
    'sus': array([1,1]),          # Relative susceptibility by
                                  # age/vulnerability class
    'fit_method' : 'R*'
}

TWO_AGE_SEIR_SPEC_FOR_FITTING = {
    'compartmental_structure': 'SEIR', # This is which subsystem key to use
    'SITP': TRANCHE2_SITP,                     # Secondary inf probability
    'recovery_rate': 1 / (PRODROME_PERIOD +
                          SYMPTOM_PERIOD),           # Recovery rate
    'incubation_rate': 1 / LATENT_PERIOD,         # E->I incubation rate
    'sus': array([1,1]),          # Relative susceptibility by
                                  # age/vulnerability class
    'fit_method' : 'EL'
}

TWO_AGE_SEPIR_SPEC = {
    'compartmental_structure': 'SEPIR', # This is which subsystem key to use
    'SITP': TRANCHE2_SITP,                     # Secondary inf probability
    'R*': 1.1,                      # Household-level reproduction number
    'recovery_rate': 1 / SYMPTOM_PERIOD,           # Recovery rate
    'incubation_rate': 1 / LATENT_PERIOD,         # E->P incubation rate
    'symp_onset_rate': 1 / PRODROME_PERIOD,         # P->I prodromal to symptomatic rate
    'prodromal_trans_scaling':
     PRODROME_SCALING * ones(2,),          # Prodromal transmission intensity relative to
                                # full inf transmission
    'sus': array([1,1]),          # Relative susceptibility by
                                  # age/vulnerability class
    'fit_method' : 'R*'
}

TWO_AGE_SEPIR_SPEC_FOR_FITTING = {
    'compartmental_structure': 'SEPIR', # This is which subsystem key to use
    'SITP': TRANCHE2_SITP,                     # Secondary inf probability
    'recovery_rate': 1 / SYMPTOM_PERIOD,           # Recovery rate
    'incubation_rate': 1 / LATENT_PERIOD,         # E->P incubation rate
    'symp_onset_rate': 1 / PRODROME_PERIOD,         # P->I prodromal to symptomatic rate
    'prodromal_trans_scaling':
     PRODROME_SCALING * ones(2,),          # Prodromal transmission intensity relative to
                                # full inf transmission
    'sus': array([1,1]),          # Relative susceptibility by
                                  # age/vulnerability class
    'fit_method' : 'EL'
}

TWO_AGE_INT_SEPIRQ_SPEC = {
    'compartmental_structure': 'SEPIRQ', # This is which subsystem key to use
    'SITP': TRANCHE2_SITP,                     # Secondary inf probability
    'recovery_rate': 1 / SYMPTOM_PERIOD,           # Recovery rate
    'incubation_rate': 1 / LATENT_PERIOD,         # E->P incubation rate
    'symp_onset_rate': 1 / PRODROME_PERIOD,         # P->I prodromal to symptomatic rate
    'exp_iso_rate': 1/1 * ones(2,),  # Ave. time in days to detection by class
    'pro_iso_rate': 1/1 * ones(2,),
    'inf_iso_rate': 1/1 * ones(2,),
    'discharge_rate': 1/14,         # 1 / ave time in isolation
    'iso_method': "int",            # This is either "int" or "ext"
    'ad_prob': 0.2,                   # Probability under internal isolation
                                      # that household members actually isolate
    'class_is_isolating':
    array([[True, True, True],
           [True, True, True],
           [True, True, True]]), # Element (i,j) is "If someone of class j is
                                 # present, class i will isolate internally"
    'prodromal_trans_scaling':
     PRODROME_SCALING * ones(2,),          # Prodromal transmission intensity relative to
                                # full inf transmission
    'iso_trans_scaling':
     array([1,1]),          # Prodromal transmission intensity relative to full
                            # inf transmission
    'sus': array([1,1]),    # Relative susceptibility by age/vulnerability class
    'fit_method' : 'EL'
}

TWO_AGE_EXT_SEPIRQ_SPEC = {
    'compartmental_structure': 'SEPIRQ', # This is which subsystem key to use
    'SITP': TRANCHE2_SITP,                     # Secondary inf probability
    'recovery_rate': 1 / SYMPTOM_PERIOD,           # Recovery rate
    'incubation_rate': 1 / LATENT_PERIOD,         # E->P incubation rate
    'symp_onset_rate': 1 / PRODROME_PERIOD,         # P->I prodromal to symptomatic rate
    'exp_iso_rate': 1/1 * ones(2,),  # Ave. time in days to detection by class
    'pro_iso_rate': 1/1 * ones(2,),
    'inf_iso_rate': 1/1 * ones(2,),
    'discharge_rate': 1/14,         # 1 / ave time in isolation
    'iso_method': "ext",            # This is either "int" or "ext"
    'ad_prob': 0.2,                   # Probability under OOHI that household
                                      # members actually isolate
    'class_is_isolating':
    array([[False, False, False],
           [False, False, True],
           [False, False, False]]), # Element (i,j) is "If someone of class j is
                                    # present, class i will isolate externally"
    'prodromal_trans_scaling':
     PRODROME_SCALING * ones(2,),          # Prodromal transmission intensity relative to
                                # full inf transmission
    'iso_trans_scaling':
     array([0,0]),          # Isolated transmission intensity relative to full
                            # inf transmission
    'sus': array([1,1]),    # Relative susceptibility by age/vulnerability class
    'fit_method' : 'EL'
}

SINGLE_AGE_UK_SPEC = {
    # Load in within-hh and pop-level contact matrices:
    'k_home': {
        'file_name': 'inputs/MUestimates_home_2.xlsx',
        'sheet_name':'United Kingdom of Great Britain'
    },
    'k_all': {
        'file_name': 'inputs/MUestimates_all_locations_2.xlsx',
        'sheet_name': 'United Kingdom of Great Britain'
    },
    # Load in age pyramid:
    'pop_pyramid_file_name': 'inputs/United Kingdom-2019.csv',
            'fine_bds' : arange(0,81,5),    # Boundaries used in contact data
    'coarse_bds' : array([0]),  # Desired boundaries for model population
    'adult_bd' : 1
}

TWO_AGE_UK_SPEC = {
    'k_home': {
        'file_name': 'inputs/MUestimates_home_2.xlsx',
        'sheet_name':'United Kingdom of Great Britain'
    },
    'k_all': {
        'file_name': 'inputs/MUestimates_all_locations_2.xlsx',
        'sheet_name': 'United Kingdom of Great Britain'
    },
    'pop_pyramid_file_name': 'inputs/United Kingdom-2019.csv',
    'fine_bds' : arange(0,81,5),
    'coarse_bds' : array([0,20]),
    'adult_bd' : 1
}
