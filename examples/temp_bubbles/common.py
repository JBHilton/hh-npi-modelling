''' Common utilities for transient bubble calculation.
'''
from numpy import (
    append, arange, array, copy, hstack, ix_, ones, prod, shape, sum, unique,
    where, zeros)
from numpy import int64 as my_int
from scipy.sparse import csc_matrix as sparse
from scipy.special import factorial
from scipy.stats import multinomial

def build_mixed_compositions_pairwise(composition_list,
                                        composition_distribution,
                                        max_size=12):
    '''
    Creates household composition list and distribution for population of
    bubbles of two households.

    Parameters
    ----------
        composition_list : numpy array
            list of observed compositions in unbubbled population
        composition_distribution : numpy array
            distribution of observed compositions in unbubbled population
        max_size : int
            maximum permitted bubble size

    Returns
    -------
        mixed_comp_list : numpy array
            list of observed compositions in the bubbled population
        mixed_comp_dist : numpy array
            distribution of observed compositions in the bubbled population
        hh_dimension : int
            number of risk classes in unbubbled households
        pairings : numpy array
            array specifying position in unbubbled composition list of
            constituent compositions for each bubbbled composition
    '''

    no_comps = composition_list.shape[0]

    if composition_list.ndim==1:
        hh_dimension = 1
    else:
        hh_dimension = composition_list.shape[1]

    mixed_comp_list = []

    pairings = [[],[]]

    total_prob_1 = zeros((no_comps,))

    for hh1 in range(no_comps):
        hh2_max = min(no_comps, int((max_size - (hh1+1))))
        total_prob_1[hh1] = sum(composition_distribution[:hh2_max])
        for hh2 in range(no_comps):
            if (hh2>=hh1) and (hh2<hh2_max):

                pairings[0].append(hh1)
                pairings[1].append(hh2)

                this_merged_comp = zeros((2*hh_dimension,))
                this_merged_comp[:hh_dimension] = \
                 composition_list[hh1,]
                this_merged_comp[hh_dimension:2*hh_dimension] = \
                 composition_list[hh2,]

                mixed_comp_list.append(this_merged_comp)

    def mixed_comp_term(p0,p1):
        hh2_max = min(no_comps, max_size - (p0 + 1) - 1)
        return composition_distribution[p0] * \
                    (composition_distribution[p1] /
                    sum(composition_distribution[:hh2_max]))

    no_merged_comps = len(mixed_comp_list)
    mixed_comp_list = array(mixed_comp_list, dtype=my_int)
    pairings = array(pairings, dtype=my_int).T
    mixed_comp_dist = zeros((no_merged_comps,))
    for mc in range(no_merged_comps):
        p_unique = unique(pairings[mc, :])
        if len(p_unique)==1:
            mixed_comp_dist[mc] = mixed_comp_term(p_unique[0],p_unique[0])
        else:
            pair0 = p_unique[0]
            pair1 = p_unique[1]
            mixed_comp_dist[mc] = mixed_comp_term(pair0,pair1) + \
                                    mixed_comp_term(pair1, pair0)

    return mixed_comp_list, mixed_comp_dist, hh_dimension, pairings

def build_mixed_compositions_threewise(composition_list,
                                        composition_distribution,
                                        max_size):
    '''
    Creates household composition list and distribution for population of
    bubbles of three households.

    Parameters
    ----------
        composition_list : numpy array
            list of observed compositions in unbubbled population
        composition_distribution : numpy array
            distribution of observed compositions in unbubbled population
        max_size : int
            maximum permitted bubble size

    Returns
    -------
        mixed_comp_list : numpy array
            list of observed compositions in the bubbled population
        mixed_comp_dist : numpy array
            distribution of observed compositions in the bubbled population
        hh_dimension : int
            number of risk classes in unbubbled households
        pairings : numpy array
            array specifying position in unbubbled composition list of
            constituent compositions for each bubbbled composition
    '''

    no_comps = composition_list.shape[0]

    if composition_list.ndim==1:
        hh_dimension = 1
    else:
        hh_dimension = composition_list.shape[1]

    mixed_comp_list = []

    pairings = [[],[],[]]

    total_prob_1 = zeros((no_comps,))
    total_prob_2 = zeros((no_comps, no_comps)) # total_prob_2[i,j] stores summed probability of all possible third elements if first two are i,j

    for hh1 in range(no_comps):
        hh2_max = min(no_comps, int((max_size - (hh1+1))))
        total_prob_1[hh1] = sum(composition_distribution[:hh2_max])
        for hh2 in range(no_comps):
            hh3_max = min(no_comps, max_size - (hh1+1) - (hh2+1))
            total_prob_2[hh1, hh2] = sum(composition_distribution[:hh3_max])
            if (hh2>=hh1) and (hh2<hh2_max):
                for hh3 in range(hh2, hh3_max):

                    pairings[0].append(hh1)
                    pairings[1].append(hh2)
                    pairings[2].append(hh3)

                    this_merged_comp = zeros((3*hh_dimension,))
                    this_merged_comp[:hh_dimension] = \
                     composition_list[hh1,]
                    this_merged_comp[hh_dimension:2*hh_dimension] = \
                     composition_list[hh2,]
                    this_merged_comp[2*hh_dimension:] = \
                     composition_list[hh3,]

                    mixed_comp_list.append(this_merged_comp)

            #     total_prob_2[hh1, hh2] += composition_distribution[hh3]
            # total_prob_1[hh1] += composition_distribution[hh2]

    def mixed_comp_term(p0,p1,p2):
        hh2_max = min(no_comps, max_size - (p0 + 1) - 1)
        hh3_max = min(no_comps, max_size - (p0 + 1) - (p1 + 1))
        return composition_distribution[p0] * \
                    (composition_distribution[p1] /
                    sum(composition_distribution[:hh2_max])) * \
                    (composition_distribution[p2] /
                    sum(composition_distribution[:hh3_max]))

    no_merged_comps = len(mixed_comp_list)
    mixed_comp_list = array(mixed_comp_list, dtype=my_int)
    pairings = array(pairings, dtype=my_int).T
    mixed_comp_dist = zeros((no_merged_comps,))
    for mc in range(no_merged_comps):
        p_unique = unique(pairings[mc, :])
        if len(p_unique)==1:
            mixed_comp_dist[mc] = mixed_comp_term(p_unique[0],p_unique[0],p_unique[0])
        elif len(p_unique)==2:
            if len(where(pairings[mc,:]==p_unique[0])[0])==2:
                pair0 = p_unique[0]
                pair1 = p_unique[1]
            else:
                pair0 = p_unique[1]
                pair1 = p_unique[0]
            mixed_comp_dist[mc] = mixed_comp_term(pair0, pair0, pair1) + \
                                        mixed_comp_term(pair0, pair1, pair0) + \
                                        mixed_comp_term(pair1, pair0, pair0)
        else:
            pair0 = p_unique[0]
            pair1 = p_unique[1]
            pair2 = p_unique[2]
            mixed_comp_dist[mc] = mixed_comp_term(pair0,pair1,pair2) + \
                                    mixed_comp_term(pair0, pair2, pair1) + \
                                    mixed_comp_term(pair1, pair0, pair2) + \
                                    mixed_comp_term(pair1, pair2, pair0) + \
                                    mixed_comp_term(pair2, pair0, pair1) + \
                                    mixed_comp_term(pair2, pair1, pair0)
        # if len(unique(pairings[mc,:]))==2:
        #     mixed_comp_dist[mc] = 3 * mixed_comp_dist[mc]
        # elif len(unique(pairings[mc,:]))==3:
        #     mixed_comp_dist[mc] = 6 * mixed_comp_dist[mc]

    return mixed_comp_list, mixed_comp_dist, hh_dimension, pairings

def initialise_merged_system(
        H0_unmerged,
        unmerged_population,
        merged_population,
        state_match):
    '''
    Creates initial conditions for bubbled population system based on state of
    unbubbled population system by mapping between state spaces of the systems.

    Parameters
    ----------
        H0_unmerged : numpy array
            state of unbubbled system at start of merge
        unmerged_population : HouseholdPopulation
            HouseholdPopulation object for unbubbled system
        merged_population : HouseholdPopulation
            HouseholdPopulation object for bubbled system
        state_match : numpy array
            array mapping between bubbled and unbubbled state spaces

    Returns
    -------
        H0_merged : numpy array
            initial conditions for bubbled system at start of merge
    '''

    wc_um = unmerged_population.which_composition
    wc_m = merged_population.which_composition
    cd_um = unmerged_population.composition_distribution
    cd_m = merged_population.composition_distribution
    no_merged_states = len(wc_m)
    H0_merged = zeros((no_merged_states,))
    for state_no in range(no_merged_states):
        this_H0_merged = \
            cd_m[wc_m[state_no]] \
            * prod(H0_unmerged[state_match[state_no, :]]) \
            / prod(cd_um[wc_um[state_match[state_no, :]]])
        H0_merged[state_no] = this_H0_merged

    return H0_merged

def match_merged_states_to_unmerged(
        unmerged_population,
        merged_population,
        pairings,
        no_hh,
        no_compartments):
    '''
    Create a mapping between state spaces of the bubbled and unbubbled
    populations.

    Parameters
    ----------
        unmerged_population : HouseholdPopulation
            HouseholdPopulation object for unbubbled system
        merged_population : HouseholdPopulation
        pairings : numpy array
            array specifying position in unbubbled composition list of
            constituent compositions for each bubbbled composition
        no_hh : int
            number of households in each bubble
        no_compartments : int
            number of epidemiological compartments in the model

    Returns
    -------
        state_match : numpy array
            array mapping states of bubbled system to unbubbled system
    '''

    rp_um = unmerged_population.reverse_prod
    iv_um = unmerged_population.index_vector
    states_m = merged_population.states
    wc_m = merged_population.which_composition

    # pdb.set_trace()
    # iv_shifter = hstack((array(0),cumsum(unmerged_population.system_sizes))) # This shifts the index vectors so that they give you indices in the  full state list rather than in the individual matrix blocks

    state_match = zeros((len(wc_m), no_hh), dtype=my_int)

    for state_no in range(len(wc_m)):
        merged_comp = wc_m[state_no]
        for hh in range(no_hh):
            unmerged_comp = pairings[merged_comp, hh]
            this_iv = iv_um[unmerged_comp]
            this_state = states_m[
                state_no, hh * no_compartments:(hh+1) * no_compartments]
            index = this_iv[
                this_state.dot(rp_um[unmerged_comp]) + this_state[-1], 0]
            state_match[state_no, hh] = index

    return state_match

def demerged_initial_condition(H_merged,
                            unmerged_population,
                            merged_population,
                            hh_dimension,
                            pairings,
                            no_hh =2,
                            no_compartments = 5):
    '''
    Creates initial conditions for unbubbled population system based on state of
    bubbled population system by mapping between state spaces of the systems.

    Parameters
    ----------
        H_unmerged : numpy array
            state of bubbled system at start of merge
        unmerged_population : HouseholdPopulation
            HouseholdPopulation object for unbubbled system
        merged_population : HouseholdPopulation
            HouseholdPopulation object for bubbled system
        pairings : numpy array
            array specifying position in unbubbled composition list of
            constituent compositions for each bubbbled composition
        no_hh : int
            number of households in each bubble
        no_compartments : int
            number of epidemiological compartments in the model

    Returns
    -------
        H0 : numpy array
            initial conditions for unbubbled system
    '''

    H0_len = sum(unmerged_population.system_sizes)
    H0 = zeros((H0_len,))
    reverse_prod = unmerged_population.reverse_prod
    index_vector_list = unmerged_population.index_vector
    which_composition = merged_population.which_composition
    merged_states = merged_population.states
    unmerged_states = unmerged_population.states

    for i in range(len(H_merged)):
        for hh in range(no_hh):
            comp = pairings[which_composition[i], hh]
            state = merged_states[i,
                hh * hh_dimension * no_compartments :
                (hh+1) * hh_dimension * no_compartments]
            index_vector = index_vector_list[comp]
            index = index_vector[
                state.dot(reverse_prod[comp]) + state[-1], 0]
            H0[index] += (1 / no_hh) * H_merged[i]

    return H0
