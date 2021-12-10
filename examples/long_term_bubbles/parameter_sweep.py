''' In this script we do projections  of the impact of support bubble policies
 by doing a 2D parameter sweep'''

from argparse import ArgumentParser
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from copy import deepcopy
from multiprocessing import Pool
from numpy import append, arange, array, exp, log, sum, vstack, where
from numpy.linalg import eig
from numpy.random import rand
from pandas import read_csv
from time import time as get_time
from scipy.integrate import solve_ivp
from model.preprocessing import ( build_support_bubbles, estimate_beta_ext,
        estimate_growth_rate, SEPIRInput, HouseholdPopulation,
        make_initial_condition_by_eigenvector)
from model.specs import TWO_AGE_SEPIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC
from model.common import SEPIRRateEquations
from model.imports import NoImportModel

if isdir('outputs/long_term_bubbles') is False:
    mkdir('outputs/long_term_bubbles')

MAX_ADULTS = 1 # In this example we assume only single-adult households can join bubbles
MAX_BUBBLE_SIZE = 10
SPEC = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}
DOUBLING_TIME = 3
X0 = log(2) / DOUBLING_TIME


composition_list = read_csv(
    'inputs/eng_and_wales_adult_child_composition_list.csv',
    header=0).to_numpy()
comp_dist = read_csv(
    'inputs/eng_and_wales_adult_child_composition_dist.csv',
    header=0).to_numpy().squeeze()

if isfile('outputs/long_term_bubbles/beta_ext.pkl') is True:
    with open('outputs/long_term_bubbles/beta_ext.pkl', 'rb') as f:
        beta_ext = load(f)
else:
    growth_rate = log(2) / DOUBLING_TIME # Doubling time of 3 days
    model_input_to_fit = SEPIRInput(SPEC, composition_list, comp_dist)
    household_population_to_fit = HouseholdPopulation(
        composition_list, comp_dist, model_input_to_fit)
    rhs_to_fit = SEPIRRateEquations(model_input_to_fit,
                                    household_population_to_fit,
                                    NoImportModel(5,2))
    beta_ext = estimate_beta_ext(household_population_to_fit,
                                 rhs_to_fit,
                                 growth_rate)
    with open('outputs/long_term_bubbles/beta_ext.pkl', 'wb') as f:
        dump(beta_ext, f)

prev=1.0e-5 # Starting prevalence
starting_immunity=0 # Starting antibody prev/immunity
gr_interval = [-SPEC['recovery_rate'], 1] # Interval used in growth rate estimation
gr_tol = 1e-3 # Absolute tolerance for growth rate estimation

class BubbleAnalysis:
    def __init__(self):
        basic_mixed_comp_list, basic_mixed_comp_dist = build_support_bubbles(
                composition_list,
                comp_dist,
                MAX_ADULTS,
                MAX_BUBBLE_SIZE,
                1)

        basic_mixed_comp_dist = basic_mixed_comp_dist/sum(basic_mixed_comp_dist)
        self.basic_bubbled_input = SEPIRInput(SPEC,
                                              basic_mixed_comp_list,
                                              basic_mixed_comp_dist)

        self.bubbled_household_population = HouseholdPopulation(
                                                        basic_mixed_comp_list,
                                                        basic_mixed_comp_dist,
                                                        self.basic_bubbled_input)

        rhs = SEPIRRateEquations(self.basic_bubbled_input,
                                 self.bubbled_household_population,
                                 NoImportModel(5,2))

    def __call__(self, p):
        print('now calling')
        try:
            result = self._implement_bubbles(p)
        except ValueError as err:
            print(
                'Exception raised for parameters={0}\n\tException: {1}'.format(
                p, err)
                )
            return 0.0
        return result

    def _implement_bubbles(self, p):
        print('p=',p)
        mixed_comp_list, mixed_comp_dist = build_support_bubbles(
                composition_list,
                comp_dist,
                MAX_ADULTS,
                MAX_BUBBLE_SIZE,
                p[0])

        mixed_comp_dist = mixed_comp_dist/sum(mixed_comp_dist)
        bubbled_model_input = deepcopy(self.basic_bubbled_input)
        bubbled_model_input.composition_distribution = mixed_comp_dist
        bubbled_model_input.k_ext = \
                        (1 - p[1]) * beta_ext * bubbled_model_input.k_ext

        household_population = deepcopy(self.bubbled_household_population)
        household_population.composition_distribution = mixed_comp_dist

        rhs = SEPIRRateEquations(bubbled_model_input,
                                 household_population,
                                 NoImportModel(5,2))

        print('calculating growth rate, p=',p)

        growth_rate = estimate_growth_rate(household_population,
                                           rhs,
                                           gr_interval,
                                           gr_tol,
                                           (1 - p[1]) * X0)
        if growth_rate is None:
            growth_rate = 0

        print('initialising, p=',p)

        H0 = make_initial_condition_by_eigenvector(growth_rate,
                                                   bubbled_model_input,
                                                   household_population,
                                                   rhs, prev,
                                                   starting_immunity)

        print('solving, p=',p)

        no_days = 100
        tspan = (0.0, no_days)
        solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)

        print('calculating output, p=',p)

        t = solution.t
        H = solution.y

        I = (H.T.dot(household_population.states[:, 3::5])).sum(axis=1)/ \
                                                household_population.ave_hh_size
        R = (H.T.dot(household_population.states[:, 4::5])).sum(axis=1)/ \
                                                household_population.ave_hh_size
        R_end_vec = H[:, -1] * \
                        household_population.states[:, 4::5].sum(axis=1)
        attack_ratio = \
                    (household_population.state_to_comp_matrix.T.dot(R_end_vec))
        attack_ratio = 100 * bubbled_model_input.composition_distribution.dot(
                                attack_ratio / bubbled_model_input.hh_size_list)

        recovered_states = where(
            ((rhs.states_sus_only + rhs.states_rec_only).sum(axis=1)
                    == household_population.states.sum(axis=1))
            & ((rhs.states_rec_only).sum(axis=1) > 0))[0]
        hh_outbreak_prop = 100 * H[recovered_states, -1].sum()

        peaks = 100 * max(I)
        R_end = 100 * R[-1]

        return [growth_rate, peaks, R_end, attack_ratio, hh_outbreak_prop]

def main(no_of_workers,
         bubble_prob_vals,
         external_mix_vals):
    bubble_system = BubbleAnalysis()
    print('built bubble system')
    results = []
    bubble_prob_range = arange(bubble_prob_vals[0],
                                bubble_prob_vals[1],
                                bubble_prob_vals[2])
    external_mix_range = arange(external_mix_vals[0],
                                external_mix_vals[1],
                                external_mix_vals[2])
    params = array([
        [b, e]
        for b in bubble_prob_range
        for e in external_mix_range])

    print('about to pool')
    prepool_time = get_time()

    with Pool(no_of_workers) as pool:
        results = pool.map(bubble_system, params)

    print('Calculations took', (get_time()-prepool_time)/60, 'minutes.')

    growth_rate_data = array([r[0] for r in results]).reshape(
        len(bubble_prob_range),
        len(external_mix_range))
    peak_data = array([r[1] for r in results]).reshape(
        len(bubble_prob_range),
        len(external_mix_range))
    end_data = array([r[2] for r in results]).reshape(
        len(bubble_prob_range),
        len(external_mix_range))
    hh_prop_data = array([r[3] for r in results]).reshape(
        len(bubble_prob_range),
        len(external_mix_range))
    attack_ratio_data = array([r[4] for r in results]).reshape(
        len(bubble_prob_range),
        len(external_mix_range))

    fname = 'outputs/long_term_bubbles/results.pkl'
    with open(fname, 'wb') as f:
        dump(
            (growth_rate_data,
             peak_data,
             end_data,
             hh_prop_data,
             attack_ratio_data,
             bubble_prob_range,
             external_mix_range),
            f)

    return -1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=8)
    parser.add_argument('--bubble_prob_vals',
                        type=int,
                        default=[0.0, 1.01, 0.05])
    parser.add_argument('--external_mix_vals',
                        type=int,
                        default=[0.0, 0.99, 0.05])
    args = parser.parse_args()

    main(args.no_of_workers,
         args.bubble_prob_vals,
         args.external_mix_vals)
