from argparse import ArgumentParser
from numpy import arange, array, atleast_2d, diag, hstack, ones, where
from os import mkdir
from os.path import isdir,isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from time import time
from multiprocessing import Pool
from model.preprocessing import ( estimate_beta_ext, merge_hh_inputs,
        SEIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import SINGLE_AGE_UK_SPEC, SINGLE_AGE_SEIR_SPEC_FOR_FITTING
from model.common import SEIRRateEquations
from model.imports import NoImportModel
from examples.temp_bubbles.common import (
        build_mixed_compositions_pairwise,
        build_mixed_compositions_threewise,
        demerged_initial_condition,
        initialise_merged_system,
        match_merged_states_to_unmerged)

'''If there is not already a results folder assigned to the outputs from this
script, create one now.'''
if isdir('outputs/temp_bubbles') is False:
    mkdir('outputs/temp_bubbles')

SPEC = {**SINGLE_AGE_UK_SPEC, **SINGLE_AGE_SEIR_SPEC_FOR_FITTING}
X0 = 2 * 1e-2 # Growth rate estimate for mid-December 2020 from ONS
ATOL = 1e-16 # IVP solver tolerance
NO_COMPARTMENTS = 4 # We use an SEIR model, hence 4 compartments
MAX_MERGED_SIZE = 12 # We only allow merges where total individuals is at most 12
MAX_UNMERGED_SIZE = 6 # As usual, we model the chunk of the population in households of size 6 or fewer
GUEST_TRANS_SCALING = 1 # This is strength of guest-host interactions relative to host-host interactions

growth_rate = X0
prev = 1e-2
starting_immunity = 1e-1

comp_dist = read_csv(
    'inputs/england_hh_size_dist.csv',
    header=0).to_numpy().squeeze()
comp_dist = comp_dist[:MAX_UNMERGED_SIZE]
comp_dist = comp_dist/sum(comp_dist)
max_hh_size = len(comp_dist)
composition_list = atleast_2d(arange(1, max_hh_size+1)).T

if isfile('outputs/temp_bubbles/beta_ext.pkl') is True:
    with open('outputs/temp_bubbles/beta_ext.pkl', 'rb') as f:
        beta_ext = load(f)
else:
    growth_rate = X0
    model_input_to_fit = SEIRInput(SPEC, composition_list, comp_dist)
    household_population_to_fit = HouseholdPopulation(
        composition_list, comp_dist, model_input_to_fit)
    rhs_to_fit = SEIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(4,1))
    beta_ext = estimate_beta_ext(household_population_to_fit, rhs_to_fit, growth_rate)
    with open('outputs/temp_bubbles/beta_ext.pkl', 'wb') as f:
        dump(beta_ext, f)


def create_unmerged_context(p):
    return UnmergedContext(*p)


class UnmergedContext:
    def __init__(self, i, density_exponent, t_start, t_end, merge_start):
        unmerged_input = SEIRInput(SPEC, composition_list, comp_dist)
        unmerged_input.density_expo = density_exponent
        unmerged_input.k_home = (
            unmerged_input.ave_hh_size ** density_exponent) * unmerged_input.k_home
        unmerged_input.k_ext = beta_ext * unmerged_input.k_ext
        self.input = unmerged_input

        self.population = HouseholdPopulation(
           composition_list,
           comp_dist,
           unmerged_input,
           True)

        self.rhs = SEIRRateEquations(
            unmerged_input,
            self.population,
            NoImportModel(NO_COMPARTMENTS, 1))

        self.recovered_states = where(
            ((self.rhs.states_sus_only + self.rhs.states_rec_only).sum(axis=1)
                    == self.population.states.sum(axis=1))
            & ((self.rhs.states_rec_only).sum(axis=1) > 0))[0]

        self.H0 = make_initial_condition_by_eigenvector(growth_rate,
                                                   unmerged_input,
                                                   self.population,
                                                   self.rhs,
                                                   prev,
                                                   starting_immunity)
        self.tspan = (t_start, merge_start)
        solution = solve_ivp(
            self.rhs,
            self.tspan,
            self.H0,
            first_step=0.001,
            atol=ATOL)
        baseline_time = solution.t
        self.baseline_H = solution.y

        self.baseline_H0 = self.baseline_H[:, -1]

        tspan = (merge_start, t_end)
        solution = solve_ivp(
            self.rhs,
            tspan,
            self.baseline_H[:, -1],
            first_step=0.001,
            atol=ATOL)
        baseline_time = hstack((baseline_time, solution.t))
        baseline_H = hstack((self.baseline_H, solution.y))

        baseline_S = baseline_H.T.dot(
            self.population.states[:, 0])/unmerged_input.ave_hh_size
        baseline_E = baseline_H.T.dot(
            self.population.states[:, 1])/unmerged_input.ave_hh_size
        baseline_I = baseline_H.T.dot(
            self.population.states[:, 2])/unmerged_input.ave_hh_size
        baseline_R = baseline_H.T.dot(
            self.population.states[:, 3])/unmerged_input.ave_hh_size

        self.peaks_baseline = max(baseline_I)
        self.R_end_baseline = baseline_R[-1]
        R_end_vec = baseline_H[:, -1] * \
                        self.population.states[:, 3]
        ar_baseline = (
                self.population.state_to_comp_matrix.T.dot(R_end_vec))
        self.ar_baseline = self.input.composition_distribution.dot(
                                        ar_baseline / self.input.hh_size_list)
        self.hh_prop_baseline = baseline_H[self.recovered_states, -1].sum()


def unpack_paramas_and_run_merge(p):
    '''This function enables multi-argument parallel run.'''
    return run_merge(*p)

def create_merged_systems(p):
    return MergedSystems(*p)

class MergedSystems:
    def __init__(
            self,
            merged_exp,
            merged_comp_list_2,
            merged_comp_dist_2,
            merged_comp_list_3,
            merged_comp_dist_3
            ):

        merged_input2 = SEIRInput(
                        SPEC, composition_list, comp_dist)
        merged_input2.composition_list = merged_comp_list_2
        merged_input2.composition_distribution = merged_comp_dist_2
        merged_input2 = merge_hh_inputs(merged_input2, 2, GUEST_TRANS_SCALING)
        merged_input2.density_expo = merged_exp
        self.merged_input2 = merged_input2
        self.merged_population2 = HouseholdPopulation(
            merged_comp_list_2,
            merged_comp_dist_2,
            merged_input2,
            True)
        self.rhs_merged2 = SEIRRateEquations(
            merged_input2,
            self.merged_population2,
            NoImportModel(NO_COMPARTMENTS, 2))

        merged_input3 = SEIRInput(
                        SPEC, composition_list, comp_dist)
        merged_input3.composition_list = merged_comp_list_3
        merged_input3.composition_distribution = merged_comp_dist_3
        merged_input3 = merge_hh_inputs(merged_input3, 3, GUEST_TRANS_SCALING)
        merged_input3.density_expo = merged_exp
        self.merged_input3 = merged_input3
        self.merged_population3 = HouseholdPopulation(
            merged_comp_list_3,
            merged_comp_dist_3,
            merged_input3,
            True)
        self.rhs_merged3 = SEIRRateEquations(
            merged_input3,
            self.merged_population3,
            NoImportModel(NO_COMPARTMENTS, 3))


def run_merge(
        i,
        j,
        pairings_2,
        pairings_3,
        hh_dimension,
        unmerged,
        merged,
        state_match_2,
        state_match_3,
        t_start,
        t_end,
        merge_start,
        merge_end
        ):
    '''This function runs a merge simulation for specified parameters.'''

    this_iteration_start = time()

    # STRATEGIES: 1 is form triangles for 2 days, 2 is form pair on
    # 25th and again on 26th, 3 is 1 plus pair on new year's, 4 is 2
    # plus pair on new year's
    # Strategy 0 is the actual policy of a 2-household bubble on Christmas day only

    merged_population2 = merged.merged_population2
    merged_population3 = merged.merged_population3
    rhs_merged2 = merged.rhs_merged2
    rhs_merged3 = merged.rhs_merged3

    t_merge_0, \
        H_merge_0, \
        t_postmerge_0, \
        H_postmerge_0 = \
        simulate_merge2(
            unmerged.population,
            merged_population2,
            unmerged.rhs,
            rhs_merged2,
            hh_dimension,
            pairings_2,
            state_match_2,
            [1],
            0,
            unmerged.baseline_H0,
            merge_start,
            merge_end)
    solution = solve_ivp(
        unmerged.rhs,
        (merge_start + 1, t_end),
        H_postmerge_0[:, -1],
        first_step=0.001,
        atol=ATOL)
    t_postmerge_0 = hstack(
        (t_postmerge_0, solution.t))
    H_postmerge_0 = hstack(
        (H_postmerge_0, solution.y))

    t_merge_1, \
        H_merge_1, \
        t_postmerge_1, \
        H_postmerge_1 = \
        simulate_merge3(
            unmerged.population,
            merged_population3,
            unmerged.rhs,
            rhs_merged3,
            hh_dimension,
            pairings_3,
            state_match_3,
            [2],
            0,
            unmerged.baseline_H0,
            merge_start,
            merge_end)
    t_merge_3, \
        H_merge_3, \
        t_postmerge_3, \
        H_postmerge_3 = \
        simulate_merge2(
            unmerged.population,
            merged_population2,
            unmerged.rhs,
            rhs_merged2,
            hh_dimension,
            pairings_2,
            state_match_2,
            [1],
            0,
            H_postmerge_1[:, -1],
            merge_end,
            t_end)

    solution = solve_ivp(
        unmerged.rhs,
        (merge_end, t_end),
        H_postmerge_1[:, -1],
        first_step=0.001,
        atol=ATOL)
    t_postmerge_1 = hstack(
        (t_postmerge_1, solution.t))
    H_postmerge_1 = hstack(
        (H_postmerge_1, solution.y))

    t_merge_2, \
        H_merge_2, \
        t_postmerge_2, \
        H_postmerge_2 = \
        simulate_merge2(
            unmerged.population,
            merged_population2,
            unmerged.rhs,
            rhs_merged2,
            hh_dimension,
            pairings_2,
            state_match_2,
            [1, 1],
            1,
            unmerged.baseline_H0,
            merge_start,
            merge_end)
    t_merge_4, \
        H_merge_4, \
        t_postmerge_4, \
        H_postmerge_4 = \
        simulate_merge2(
            unmerged.population,
            merged_population2,
            unmerged.rhs,
            rhs_merged2,
            hh_dimension,
            pairings_2,
            state_match_2,
            [1],
            0,
            H_postmerge_2[:, -1],
            merge_end,
            t_end)

    solution = solve_ivp(
        unmerged.rhs,
        (merge_end, t_end),
        H_postmerge_2[:, -1],
        first_step=0.001,
        atol=ATOL)
    t_postmerge_2 = hstack((
        t_postmerge_2, solution.t))
    H_postmerge_2 = hstack((
        H_postmerge_2, solution.y))

    merge_I_0 = (1/2) * H_merge_0.T.dot(
        merged_population2.states[:, 2] + merged_population2.states[:, 6])/unmerged.input.ave_hh_size
    postmerge_I_0 = H_postmerge_0.T.dot(unmerged.population.states[:, 2])/unmerged.input.ave_hh_size
    peaks_0 = max(hstack((merge_I_0, postmerge_I_0)))
    merge_R_0 = (1/2) * H_merge_0.T.dot(
        merged_population2.states[:, 3] + merged_population2.states[:, 7])/unmerged.input.ave_hh_size
    postmerge_R_0 = H_postmerge_0.T.dot(unmerged.population.states[:, 3])/unmerged.input.ave_hh_size
    R_end_0 = postmerge_R_0[-1]
    R_end_vec_0 = H_postmerge_0[:, -1] * \
                    unmerged.population.states[:, 3]
    ar_0 = (unmerged.population.state_to_comp_matrix.T.dot(R_end_vec_0))
    ar_0 = unmerged.input.composition_distribution.dot(
                                    ar_0 / unmerged.input.hh_size_list)
    hh_prop_0 = H_postmerge_0[unmerged.recovered_states, -1].sum()

    merge_I_1 = (1/3) * H_merge_1.T.dot(
        merged_population3.states[:, 2] + merged_population3.states[:, 6] +
        merged_population3.states[:, 10])/unmerged.input.ave_hh_size
    postmerge_I_1 = H_postmerge_1.T.dot(unmerged.population.states[:, 2])/unmerged.input.ave_hh_size
    peaks_1 = max(hstack((merge_I_1, postmerge_I_1)))
    merge_R_1 = (1/3) * H_merge_1.T.dot(
        merged_population3.states[:, 3] + merged_population3.states[:, 7] +
        merged_population3.states[:,11])/unmerged.input.ave_hh_size
    postmerge_R_1 = H_postmerge_1.T.dot(unmerged.population.states[:, 3])/unmerged.input.ave_hh_size
    R_end_1 = postmerge_R_1[-1]
    R_end_vec_1 = H_postmerge_1[:, -1] * \
                    unmerged.population.states[:, 3]
    ar_1 = (unmerged.population.state_to_comp_matrix.T.dot(R_end_vec_1))
    ar_1 = unmerged.input.composition_distribution.dot(
                                    ar_1 / unmerged.input.hh_size_list)
    hh_prop_1 = H_postmerge_1[unmerged.recovered_states, -1].sum()

    merge_I_2 = (1/2) * H_merge_2.T.dot(
        merged_population2.states[:, 2] + merged_population2.states[:, 6])/unmerged.input.ave_hh_size
    postmerge_I_2 = H_postmerge_2.T.dot(unmerged.population.states[:, 2])/unmerged.input.ave_hh_size
    peaks_2 = max(hstack((merge_I_2, postmerge_I_2)))
    merge_R_2 = (1/2) * H_merge_2.T.dot(
        merged_population2.states[:, 3] + merged_population2.states[:, 7])/unmerged.input.ave_hh_size
    postmerge_R_2 = H_postmerge_2.T.dot(unmerged.population.states[:, 3])/unmerged.input.ave_hh_size
    R_end_2 = postmerge_R_2[-1]
    R_end_vec_2 = H_postmerge_2[:, -1] * \
                    unmerged.population.states[:, 3]
    ar_2 = (unmerged.population.state_to_comp_matrix.T.dot(R_end_vec_2))
    ar_2 = unmerged.input.composition_distribution.dot(
                                    ar_2 / unmerged.input.hh_size_list)
    hh_prop_2 = H_postmerge_2[unmerged.recovered_states, -1].sum()

    merge_I_3 = (1/2) * H_merge_3.T.dot(
        merged_population2.states[:, 2] + merged_population2.states[:, 6])/unmerged.input.ave_hh_size
    postmerge_I_3 = H_postmerge_3.T.dot(unmerged.population.states[:, 2])/unmerged.input.ave_hh_size
    peaks_3 = max(hstack((merge_I_3, postmerge_I_3)))
    merge_R_3 = (1/2) * H_merge_3.T.dot(
        merged_population2.states[:, 3] + merged_population2.states[:, 7])/unmerged.input.ave_hh_size
    postmerge_R_3 = H_postmerge_3.T.dot(unmerged.population.states[:, 3])/unmerged.input.ave_hh_size
    R_end_3 = postmerge_R_3[-1]
    R_end_vec_3 = H_postmerge_3[:, -1] * \
                    unmerged.population.states[:, 3]
    ar_3 = (unmerged.population.state_to_comp_matrix.T.dot(R_end_vec_3))
    ar_3 = unmerged.input.composition_distribution.dot(
                                    ar_3 / unmerged.input.hh_size_list)
    hh_prop_3 = H_postmerge_3[unmerged.recovered_states, -1].sum()

    merge_I_4 = (1/2) * H_merge_4.T.dot(
        merged_population2.states[:, 2] + merged_population2.states[:, 6])/unmerged.input.ave_hh_size
    postmerge_I_4 = H_postmerge_4.T.dot(unmerged.population.states[:, 2])/unmerged.input.ave_hh_size
    peaks_4 = max(hstack((merge_I_4, postmerge_I_4)))
    merge_R_4 = (1/2) * H_merge_4.T.dot(
        merged_population2.states[:, 3] + merged_population2.states[:, 7])/unmerged.input.ave_hh_size
    postmerge_R_4 = H_postmerge_4.T.dot(unmerged.population.states[:, 3])/unmerged.input.ave_hh_size
    R_end_4 = postmerge_R_4[-1]
    R_end_vec_4 = H_postmerge_4[:, -1] * \
                    unmerged.population.states[:, 3]
    ar_4 = (unmerged.population.state_to_comp_matrix.T.dot(R_end_vec_4))
    ar_4 = unmerged.input.composition_distribution.dot(
                                    ar_4 / unmerged.input.hh_size_list)
    hh_prop_4 = H_postmerge_4[unmerged.recovered_states, -1].sum()

    return [peaks_0,
            R_end_0,
            ar_0,
            hh_prop_0,
            peaks_1,
            R_end_1,
            ar_1,
            hh_prop_1,
            peaks_2,
            R_end_2,
            ar_2,
            hh_prop_2,
            peaks_3,
            R_end_3,
            ar_3,
            hh_prop_3,
            peaks_4,
            R_end_4,
            ar_4,
            hh_prop_4,
            unmerged.input.density_expo,
            merged.merged_input2.density_expo]



def simulate_merge2(
        unmerged_population,
        merged_population,
        rhs_unmerged,
        rhs_merged,
        hh_dimension,
        pairings,
        state_match,
        duration_list,
        switches,
        premerge_H0,
        t0,
        t_end):

    hh_to_merge = 2

    for switch in range(switches+1):
        # print('Initialising merge number',switch,'...')
        H0 = initialise_merged_system(
            premerge_H0,
            unmerged_population,
            merged_population,
            state_match)
        tspan = (t0, t0 + duration_list[switch])
        # print('Integrating over merge period number',switch,'...')
        solution = solve_ivp(
            rhs_merged,
            tspan,
            H0/sum(H0),
            first_step=0.001,
            atol=ATOL)
        # print('Integration over merge period took',time()-t_mer,'seconds.')
        temp_time = solution.t
        temp_H = solution.y
        t0 = t0 + duration_list[switch]
        premerge_H0 = demerged_initial_condition(
            temp_H[:, -1],
            unmerged_population,
            merged_population,
            hh_dimension,
            pairings,
            hh_to_merge,
            4)
        if switch == 0:
            merge_time = temp_time
            merge_H = temp_H
        else:
            merge_time = hstack((merge_time, temp_time))
            merge_H = hstack((merge_H, temp_H))

    # print('Initialising post-merge period...')
    H0 = demerged_initial_condition(
        merge_H[:, -1],
        unmerged_population,
        merged_population,
        hh_dimension,
        pairings,
        hh_to_merge,
        4)
    tspan = (t0, t_end)
    # print('Integrating over post-merge period...')
    solution = solve_ivp(
        rhs_unmerged,
        tspan,
        H0,
        first_step=0.001,
        atol=ATOL)
    # print('Integration over post-merge period took',time()-t_post,'seconds.')
    postmerge_time = solution.t
    postmerge_H = solution.y

    return merge_time, merge_H, postmerge_time, postmerge_H


def simulate_merge3(
        unmerged_population,
        merged_population,
        rhs_unmerged,
        rhs_merged,
        hh_dimension,
        pairings,
        state_match,
        duration_list,
        switches,
        premerge_H0,
        t0,
        t_end):

    hh_to_merge = 3

    for switch in range(switches+1):
        # print('Initialising merge number',switch,'...')
        H0 = initialise_merged_system(
            premerge_H0,
            unmerged_population,
            merged_population,
            state_match)
        tspan = (t0, t0 + duration_list[switch])
        # print('Integrating over merge period number',switch,'...')
        # t_mer = time()
        solution = solve_ivp(
            rhs_merged,
            tspan,
            H0/sum(H0),
            first_step=0.001,
            atol=ATOL)
        # print('Integration over merge period took',time()-t_mer,'seconds.')
        temp_time = solution.t
        temp_H = solution.y
        t0 = t0 + duration_list[switch]
        premerge_H0 = demerged_initial_condition(
            temp_H[:, -1],
            unmerged_population,
            merged_population,
            hh_dimension,
            pairings,
            hh_to_merge,
            4)
        if switch == 0:
            merge_time = temp_time
            merge_H = temp_H
        else:
            merge_time = hstack((
                merge_time,
                temp_time))
            merge_H = hstack((merge_H, temp_H))

    # print('Initialising post-merge period...')
    H0 = demerged_initial_condition(
        merge_H[:, -1],
        unmerged_population,
        merged_population,
        hh_dimension,
        pairings,
        hh_to_merge,
        4)
    tspan = (t0, t_end)
    # print('Integrating over post-merge period...')
    # t_post = time()
    solution = solve_ivp(
        rhs_unmerged,
        tspan,
        H0,
        first_step=0.001,
        atol=ATOL)
    # print('Integration over post-merge period took',time()-t_post,'seconds.')
    postmerge_time = solution.t
    postmerge_H = solution.y

    return merge_time, merge_H, postmerge_time, postmerge_H


def main(no_of_workers,
         unmerged_exponent_vals,
         merged_exponent_vals):

    unmerged_exponents = arange(unmerged_exponent_vals[0],
                                unmerged_exponent_vals[1],
                                unmerged_exponent_vals[2])
    merged_exponents = arange(merged_exponent_vals[0],
                                merged_exponent_vals[1],
                                merged_exponent_vals[2])

    if isfile('outputs/temp_bubbles/threewise_merge_comps.pkl') is True:
        with open('outputs/temp_bubbles/threewise_merge_comps.pkl', 'rb') as f:
            print('Loading merged composition distribution...')
            merged_comp_list_2, \
                merged_comp_dist_2, \
                pairings_2, \
                merged_comp_list_3, \
                merged_comp_dist_3, \
                hh_dimension, \
                pairings_3 = load(f)
    else:
        print('Building merged composition distribution...')
        merged_comp_list_2, \
            merged_comp_dist_2, \
            hh_dimension, \
            pairings_2 = build_mixed_compositions_pairwise(
                composition_list, comp_dist)
        merged_comp_list_3, \
            merged_comp_dist_3, \
            hh_dimension, \
            pairings_3 = build_mixed_compositions_threewise(
                composition_list, comp_dist, MAX_MERGED_SIZE)

        with open('outputs/temp_bubbles/threewise_merge_comps.pkl', 'wb') as f:
            dump(
                (
                    merged_comp_list_2,
                    merged_comp_dist_2,
                    pairings_2,
                    merged_comp_list_3,
                    merged_comp_dist_3,
                    hh_dimension,
                    pairings_3
                ),
                f)
    # December 1st
    t0 = 335.0
    # December 25th
    merge_start = 359.0
    merge_end = 365.0
    # Continue running projections to end of Jan
    t_end = 396

    params = []
    for i, e in enumerate(unmerged_exponents):
        params.append((i, e, t0, t_end, merge_start))
    with Pool(no_of_workers) as pool:
        unmerged_results = pool.map(create_unmerged_context, params)
    baseline_peak_data = array([r.peaks_baseline for r in unmerged_results])
    baseline_end_data = array([r.R_end_baseline for r in unmerged_results])
    baseline_ar_data = array([r.ar_baseline for r in unmerged_results])
    baseline_hh_prop_data = array(
                            [r.hh_prop_baseline for r in unmerged_results])
    fname = 'outputs/temp_bubbles/baseline_results.pkl'
    with open(fname, 'wb') as f:
        dump(
            (baseline_peak_data,
             baseline_end_data,
             baseline_ar_data,
             baseline_hh_prop_data),
            f)

    params = []
    for e in merged_exponents:
        params.append((
            e,
            merged_comp_list_2,
            merged_comp_dist_2,
            merged_comp_list_3,
            merged_comp_dist_3
            ))
    with Pool(no_of_workers) as pool:
        merged_populations = pool.map(create_merged_systems, params)

    state_match_2 = match_merged_states_to_unmerged(
        unmerged_results[0].population,
        merged_populations[0].merged_population2,
        pairings_2,
        2,
        NO_COMPARTMENTS)
    state_match_3 = match_merged_states_to_unmerged(
        unmerged_results[0].population,
        merged_populations[0].merged_population3,
        pairings_3,
        3,
        NO_COMPARTMENTS)

    params = []
    for i, ei in enumerate(unmerged_exponents):
        for j, ej in enumerate(merged_exponents):
            params.append((
                i, j,
                pairings_2, pairings_3,
                hh_dimension,
                unmerged_results[i],
                merged_populations[j],
                state_match_2, state_match_3,
                t0, t_end, merge_start, merge_end))
    with Pool(no_of_workers) as pool:
        results = pool.map(unpack_paramas_and_run_merge, params)

    peak_data0 = array([r[0] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    end_data0 = array([r[1] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    ar_data0 = array([r[2] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    hh_prop_data0 = array([r[3] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    peak_data1= array([r[4] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    end_data1 = array([r[5] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    ar_data1 = array([r[6] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    hh_prop_data1 = array([r[7] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    peak_data2 = array([r[8] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    end_data2 = array([r[9] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    ar_data2 = array([r[10] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    hh_prop_data2 = array([r[11] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    peak_data3 = array([r[12] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    end_data3 = array([r[13] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    ar_data3 = array([r[14] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    hh_prop_data3 = array([r[15] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    peak_data4 = array([r[16] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    end_data4 = array([r[17] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    ar_data4 = array([r[18] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    hh_prop_data4 = array([r[19] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    par_data1 = array([r[20] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))
    par_data2 = array([r[20] for r in results]).reshape(
        len(unmerged_exponents),
        len(merged_exponents))

    fname = 'outputs/temp_bubbles/results.pkl'
    with open(fname, 'wb') as f:
        dump(
            (peak_data0,
             end_data0,
             ar_data0,
             hh_prop_data0,
             peak_data1,
             end_data1,
             ar_data1,
             hh_prop_data1,
             peak_data2,
             end_data2,
             ar_data2,
             hh_prop_data2,
             peak_data3,
             end_data3,
             ar_data3,
             hh_prop_data3,
             peak_data4,
             end_data4,
             ar_data4,
             hh_prop_data4,
             par_data1,
             par_data2),
            f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=8)
    parser.add_argument('--unmerged_exponent_vals', type=float, default=[0.0, 1.01, 0.05])
    parser.add_argument('--merged_exponent_vals', type=float, default=[0.0, 1.01, 0.05])
    args = parser.parse_args()
    start = time()
    main(args.no_of_workers,
         args.unmerged_exponent_vals,
         args.merged_exponent_vals)
    print('Execution took',time() - start,'seconds.')
