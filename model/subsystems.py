'''This script defines all the subsystems for specific compartmental structures.
First, some functions for doing standard transition events are defined, then
these are used to create some functions for implementing common comprtmental
 structures. These are then stored in a dictionary, which always needs to go at
 the end of this script.'''

from copy import copy, deepcopy
from numpy import (
        arange, around, array, atleast_2d, concatenate, cumprod, diag, ix_,
        ones, prod, sum, where, zeros)
from numpy import int64 as my_int
from scipy.sparse import csc_matrix as sparse

def state_recursor(
        states,
        no_compartments,
        age_class,
        b_size,
        n_blocks,
        con_reps,
        c,
        x,
        depth,
        k):
    if depth < no_compartments-1:
        for x_i in arange(c + 1 - x.sum()):
            x[0, depth] = x_i
            x[0, depth+1:] = zeros(
                (1, no_compartments-depth-1),
                dtype=my_int)
            states, k = state_recursor(
                states,
                no_compartments,
                age_class,
                b_size,
                n_blocks,
                con_reps,
                c,
                x,
                depth+1,
                k)
    else:
        x[0, -1] = c - sum(x[0, :depth])
        for block in arange(n_blocks):
            repeat_range = arange(
                block * b_size
                + k * con_reps,
                block * b_size +
                (k + 1) * con_reps)
            states[repeat_range,
                no_compartments*age_class:no_compartments*(age_class+1)] = \
                ones(
                    (con_reps, 1),
                    dtype=my_int) \
                * array(
                    x,
                    ndmin=2, dtype=my_int)
        k += 1
        return states, k
    return states, k


def build_states_recursively(
        total_size,
        no_compartments,
        classes_present,
        block_size,
        num_blocks,
        consecutive_repeats,
        composition):
    states = zeros(
        (total_size, no_compartments*len(classes_present)),
        dtype=my_int)
    for age_class in range(len(classes_present)):
        k = 0
        states, k = state_recursor(
            states,
            no_compartments,
            age_class,
            block_size[age_class],
            num_blocks[age_class],
            consecutive_repeats[age_class],
            composition[classes_present[age_class]],
            zeros([1, no_compartments], dtype=my_int),
            0,
            k)
    return states, k


def build_state_matrix(household_spec):
    # Number of times you repeat states for each configuration
    consecutive_repeats = concatenate((
        ones(1, dtype=my_int), cumprod(household_spec.system_sizes[:-1])))
    block_size = consecutive_repeats * household_spec.system_sizes
    num_blocks = household_spec.total_size // block_size

    states, k = build_states_recursively(
        household_spec.total_size,
        household_spec.no_compartments,
        household_spec.class_indexes,
        block_size,
        num_blocks,
        consecutive_repeats,
        household_spec.composition)
    # Now construct a sparse vector which tells you which row a state appears
    # from in the state array

    # This loop tells us how many values each column of the state array can
    # take
    state_sizes = concatenate([
        (household_spec.composition[i] + 1)
        * ones(household_spec.no_compartments, dtype=my_int)
        for i in household_spec.class_indexes]).ravel()

    # This vector stores the number of combinations you can get of all
    # subsequent elements in the state array, i.e. reverse_prod(i) tells you
    # how many arrangements you can get in states(:,i+1:end)
    reverse_prod = array([0, *cumprod(state_sizes[:0:-1])])[::-1]

    # We can then define index_vector look up the location of a state by
    # weighting its elements using reverse_prod - this gives a unique mapping
    # from the set of states to the integers. Because lots of combinations
    # don't actually appear in the states array, we use a sparse array which
    # will be much bigger than we actually require
    rows = [
        states[k, :].dot(reverse_prod) + states[k, -1]
        for k in range(household_spec.total_size)]

    if min(rows) < 0:
        print(
            'Negative row indices found, proportional total',
            sum(array(rows) < 0),
            '/',
            len(rows),
            '=',
            sum(array(rows) < 0) / len(rows))
    index_vector = sparse((
        arange(household_spec.total_size),
        (rows, [0]*household_spec.total_size)))

    return states, reverse_prod, index_vector, rows

def inf_events(from_compartment,
                to_compartment,
                inf_compartment_list,
                inf_scales,
                r_home,
                density_expo,
                no_compartments,
                composition,
                states,
                index_vector,
                reverse_prod,
                class_idx,
                matrix_shape,
                Q_int,
                inf_event_row,
                inf_event_col,
                inf_event_class):

    # This function adds infection events to a within-household transition
    # matrix, allowing for multiple infectious classes

    # Total number of compartments contributing to within-household infection:
    no_inf_compartments = len(inf_compartment_list)

    for i in range(len(class_idx)):
        from_present = where(states[:, no_compartments*i+from_compartment] > 0)[0]

        inf_to = zeros(len(from_present), dtype=my_int)
        inf_rate = zeros(len(from_present))
        for k in range(len(from_present)):
            old_state = copy(states[from_present[k], :])
            old_infs = zeros(len(class_idx))
            for ic in range(no_inf_compartments):
                old_infs += \
                    (old_state[inf_compartment_list[ic] +
                                no_compartments * arange(len(class_idx))] /
                    (composition[class_idx]**density_expo)) * inf_scales[ic]
            inf_rate[k] = old_state[no_compartments*i] * (
                r_home[i, :].dot( old_infs ))
            new_state = old_state.copy()
            new_state[no_compartments*i + from_compartment] -= 1
            new_state[no_compartments*i + to_compartment] += 1
            inf_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]

        Q_int += sparse(
            (inf_rate, (from_present, inf_to)),
            shape=matrix_shape,)
        inf_event_row = concatenate((inf_event_row, from_present))
        inf_event_col = concatenate((inf_event_col, inf_to))
        inf_event_class = concatenate(
            (inf_event_class, class_idx[i]*ones((len(from_present)))))

    return Q_int, inf_event_row, inf_event_col, inf_event_class

''' The next function is used to set up the infection events in the SEPIRQ
model, where some members may be temporarily absent, changing the current
size of the household '''
def size_adj_inf_events(from_compartment,
                to_compartment,
                inf_compartment_list,
                inf_scales,
                r_home,
                iso_adjusted_comp,
                density_expo,
                no_compartments,
                composition,
                states,
                index_vector,
                reverse_prod,
                class_idx,
                matrix_shape,
                Q_int,
                inf_event_row,
                inf_event_col,
                inf_event_class):

     # Total number of compartments contributing to within-household infection
    no_inf_compartments = len(inf_compartment_list)

    for i in range(len(class_idx)):
        from_present = \
            where(states[:, no_compartments*i+from_compartment] > 0)[0]

        inf_to = zeros(len(from_present), dtype=my_int)
        inf_rate = zeros(len(from_present))
        for k in range(len(from_present)):
            old_state = copy(states[from_present[k], :])
            old_infs = zeros(len(class_idx))
            for ic in range(no_inf_compartments):
                old_infs += (old_state[inf_compartment_list[ic] + \
                            no_compartments * arange(len(class_idx))] /
                        (iso_adjusted_comp[k]**density_expo)) * inf_scales[ic]
            inf_rate[k] = old_state[no_compartments*i] * (
                r_home[i, :].dot( old_infs ))
            new_state = old_state.copy()
            new_state[no_compartments*i + from_compartment] -= 1
            new_state[no_compartments*i + to_compartment] += 1
            inf_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]

        Q_int += sparse(
            (inf_rate, (from_present, inf_to)),
            shape=matrix_shape,)
        inf_event_row = concatenate((inf_event_row, from_present))
        inf_event_col = concatenate((inf_event_col, inf_to))
        inf_event_class = concatenate(
            (inf_event_class, class_idx[i]*ones((len(from_present)))))

    return Q_int, inf_event_row, inf_event_col, inf_event_class

def progression_events(from_compartment,
                to_compartment,
                pc_rate,
                no_compartments,
                states,
                index_vector,
                reverse_prod,
                class_idx,
                matrix_shape,
                Q_int):

    # This function adds a single set of progression events to a
    # within-household transition matrix

    for i in range(len(class_idx)):
        from_present = \
            where(states[:, no_compartments*i+from_compartment] > 0)[0]

        prog_to = zeros(len(from_present), dtype=my_int)
        prog_rate = zeros(len(from_present))
        for k in range(len(from_present)):
            old_state = copy(states[from_present[k], :])
            prog_rate[k] = \
                pc_rate * old_state[no_compartments*i+from_compartment]
            new_state = copy(old_state)
            new_state[no_compartments*i+from_compartment] -= 1
            new_state[no_compartments*i+to_compartment] += 1
            prog_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]

        Q_int += sparse(
            (prog_rate, (from_present, prog_to)),
            shape=matrix_shape,)

    return Q_int

def stratified_progression_events(from_compartment,
                to_compartment,
                pc_rate_by_class,
                no_compartments,
                states,
                index_vector,
                reverse_prod,
                class_idx,
                matrix_shape,
                Q_int):

    ''' This function adds a single set of progression events to a
    within-household transition matrix, with progression rates stratified by
    class.'''

    for i in range(len(class_idx)):
        from_present = \
            where(states[:, no_compartments*i+from_compartment] > 0)[0]

        prog_to = zeros(len(from_present), dtype=my_int)
        prog_rate = zeros(len(from_present))
        for k in range(len(from_present)):
            old_state = copy(states[from_present[k], :])
            prog_rate[k] = pc_rate_by_class[i] * \
                old_state[no_compartments*i+from_compartment]
            new_state = copy(old_state)
            new_state[no_compartments*i+from_compartment] -= 1
            new_state[no_compartments*i+to_compartment] += 1
            prog_to[k] = index_vector[
                new_state.dot(reverse_prod) + new_state[-1], 0]
        Q_int += sparse(
            (prog_rate, (from_present, prog_to)),
            shape=matrix_shape)

    return Q_int

def isolation_events(from_compartment,
                to_compartment,
                iso_rate_by_class,
                class_is_isolating,
                iso_method,
                adult_bd,
                no_adults,
                children_present,
                adults_isolating,
                no_compartments,
                states,
                index_vector,
                reverse_prod,
                class_idx,
                matrix_shape,
                Q_int):

    ''' This function adds a single set of isolation events to a
    within-household transition matrix, with isolation rates stratified by
    class.'''

    for i in range(len(class_idx)):
        # The following if statement checks whether class i is meant to isolate
        # and whether any of the vulnerable classes are present
        if (class_is_isolating[class_idx[i],class_idx]).any():
            # If isolating internally, i is a child class, or there are no
            # children around, anyone can isolate
            if iso_method=='int' or (i<adult_bd) or not children_present:
                iso_permitted = \
                    where((states[:,no_compartments*i+from_compartment] > 0) * \
                    (states[:, no_compartments*i+to_compartment] == 0))[0]
             # If children are present adults_isolating must stay below
             # no_adults-1 so the children still have a guardian
            else:
                iso_permitted = \
                    where(
                        (states[:, no_compartments*i+from_compartment] > 0) * \
                        (adults_isolating<no_adults-1))[0]
            iso_present = \
                where(states[:, no_compartments*i+to_compartment] > 0)[0]

            iso_to = zeros(len(iso_permitted), dtype=my_int)
            iso_rate = zeros(len(iso_permitted))
            for k in range(len(iso_permitted)):
                old_state = copy(states[iso_permitted[k], :])
                iso_rate[k] = iso_rate_by_class[i] * \
                    old_state[no_compartments*i+from_compartment]
                new_state = copy(old_state)
                new_state[no_compartments*i+from_compartment] -= 1
                new_state[no_compartments*i+to_compartment] += 1
                iso_to[k] = index_vector[
                    new_state.dot(reverse_prod) + new_state[-1], 0]
            Q_int += sparse(
                (iso_rate, (iso_permitted, iso_to)),
                shape=matrix_shape)

    return Q_int

def _sir_subsystem(self, household_spec):
    '''This function processes a composition to create subsystems i.e.
    matrices and vectors describing all possible epdiemiological states
    for a given household composition
    Assuming frequency-dependent homogeneous within-household mixing
    composition[i] is the number of individuals in age-class i inside the
    household'''

    no_compartments = household_spec.no_compartments

    s_comp, i_comp, r_comp = range(no_compartments)

    composition = household_spec.composition
    matrix_shape = household_spec.matrix_shape
    sus = self.model_input.sus
    K_home = self.model_input.k_home
    gamma = self.model_input.gamma
    density_expo = self.model_input.density_expo

    # Set of individuals actually present here
    class_idx = household_spec.class_indexes

    K_home = K_home[ix_(class_idx, class_idx)]
    sus = sus[class_idx]
    r_home = atleast_2d(diag(sus).dot(K_home))

    states, \
        reverse_prod, \
        index_vector, \
        rows = build_state_matrix(household_spec)

    Q_int = sparse(household_spec.matrix_shape,)
    inf_event_row = array([], dtype=my_int)
    inf_event_col = array([], dtype=my_int)
    inf_event_class = array([], dtype=my_int)

    Q_int, inf_event_row, inf_event_col, inf_event_class = inf_events(s_comp,
                i_comp,
                [i_comp],
                [1],
                r_home,
                density_expo,
                no_compartments,
                composition,
                states,
                index_vector,
                reverse_prod,
                class_idx,
                matrix_shape,
                Q_int,
                inf_event_row,
                inf_event_col,
                inf_event_class)
    Q_int = progression_events(i_comp,
                    r_comp,
                    gamma,
                    no_compartments,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)

    S = Q_int.sum(axis=1).getA().squeeze()
    Q_int += sparse((
        -S, (
            arange(household_spec.total_size),
            arange(household_spec.total_size)
        )))
    return tuple((
        Q_int,
        states,
        array(inf_event_row, dtype=my_int, ndmin=1),
        array(inf_event_col, dtype=my_int, ndmin=1),
        array(inf_event_class, dtype=my_int, ndmin=1),
        reverse_prod,
        index_vector))

def _seir_subsystem(self, household_spec):
    '''This function processes a composition to create subsystems i.e.
    matrices and vectors describing all possible epdiemiological states
    for a given household composition
    Assuming frequency-dependent homogeneous within-household mixing
    composition[i] is the number of individuals in age-class i inside the
    household'''

    no_compartments = household_spec.no_compartments

    s_comp, e_comp, i_comp, r_comp = range(no_compartments)

    composition = household_spec.composition
    matrix_shape = household_spec.matrix_shape
    sus = self.model_input.sus
    K_home = self.model_input.k_home
    alpha = self.model_input.alpha
    gamma = self.model_input.gamma
    density_expo = self.model_input.density_expo

    # Set of individuals actually present here
    class_idx = household_spec.class_indexes

    K_home = K_home[ix_(class_idx, class_idx)]
    sus = sus[class_idx]
    r_home = atleast_2d(diag(sus).dot(K_home))

    states, \
        reverse_prod, \
        index_vector, \
        rows = build_state_matrix(household_spec)

    Q_int = sparse(household_spec.matrix_shape,)
    inf_event_row = array([], dtype=my_int)
    inf_event_col = array([], dtype=my_int)
    inf_event_class = array([], dtype=my_int)

    Q_int, inf_event_row, inf_event_col, inf_event_class = inf_events(s_comp,
                i_comp,
                [i_comp],
                [1],
                r_home,
                density_expo,
                no_compartments,
                composition,
                states,
                index_vector,
                reverse_prod,
                class_idx,
                matrix_shape,
                Q_int,
                inf_event_row,
                inf_event_col,
                inf_event_class)
    Q_int = progression_events(e_comp,
                    i_comp,
                    gamma,
                    no_compartments,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = progression_events(i_comp,
                    r_comp,
                    gamma,
                    no_compartments,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)

    S = Q_int.sum(axis=1).getA().squeeze()
    Q_int += sparse((
        -S, (
            arange(household_spec.total_size),
            arange(household_spec.total_size)
        )))
    return tuple((
        Q_int,
        states,
        array(inf_event_row, dtype=my_int, ndmin=1),
        array(inf_event_col, dtype=my_int, ndmin=1),
        array(inf_event_class, dtype=my_int, ndmin=1),
        reverse_prod,
        index_vector))

def _sepir_subsystem(self, household_spec):
    '''This function processes a composition to create subsystems i.e.
    matrices and vectors describing all possible epdiemiological states
    for a given household composition
    Assuming frequency-dependent homogeneous within-household mixing
    composition[i] is the number of individuals in age-class i inside the
    household'''

    no_compartments = household_spec.no_compartments

    s_comp, e_comp, p_comp, i_comp, r_comp = range(no_compartments)

    composition = household_spec.composition
    matrix_shape = household_spec.matrix_shape
    sus = self.model_input.sus
    K_home = self.model_input.k_home
    inf_scales = copy(self.model_input.inf_scales)
    alpha_1 = self.model_input.alpha_1
    alpha_2 = self.model_input.alpha_2
    gamma = self.model_input.gamma
    density_expo = self.model_input.density_expo

    # Set of individuals actually present here
    class_idx = household_spec.class_indexes

    K_home = K_home[ix_(class_idx, class_idx)]
    sus = sus[class_idx]
    r_home = atleast_2d(diag(sus).dot(K_home))
    for i in range(len(inf_scales)):
        inf_scales[i] = inf_scales[i][class_idx]

    states, \
        reverse_prod, \
        index_vector, \
        rows = build_state_matrix(household_spec)

    Q_int = sparse(household_spec.matrix_shape,)
    inf_event_row = array([], dtype=my_int)
    inf_event_col = array([], dtype=my_int)
    inf_event_class = array([], dtype=my_int)

    Q_int, inf_event_row, inf_event_col, inf_event_class = inf_events(s_comp,
                e_comp,
                [p_comp, i_comp],
                inf_scales,
                r_home,
                density_expo,
                no_compartments,
                composition,
                states,
                index_vector,
                reverse_prod,
                class_idx,
                matrix_shape,
                Q_int,
                inf_event_row,
                inf_event_col,
                inf_event_class)
    Q_int = progression_events(e_comp,
                    p_comp,
                    alpha_1,
                    no_compartments,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = progression_events(p_comp,
                    i_comp,
                    alpha_2,
                    no_compartments,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = progression_events(i_comp,
                    r_comp,
                    gamma,
                    no_compartments,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)

    S = Q_int.sum(axis=1).getA().squeeze()
    Q_int += sparse((
        -S, (
            arange(household_spec.total_size),
            arange(household_spec.total_size)
        )))
    return tuple((
        Q_int,
        states,
        array(inf_event_row, dtype=my_int, ndmin=1),
        array(inf_event_col, dtype=my_int, ndmin=1),
        array(inf_event_class, dtype=my_int, ndmin=1),
        reverse_prod,
        index_vector))

def _sepirq_subsystem(self, household_spec):
    '''This function processes a composition to create subsystems i.e.
    matrices and vectors describing all possible epdiemiological states
    for a given household composition
    Assuming frequency-dependent homogeneous within-household mixing
    composition[i] is the number of individuals in age-class i inside the
    household'''

    no_compartments = household_spec.no_compartments

    s_comp, e_comp, p_comp, i_comp, r_comp, q_comp = range(no_compartments)

    composition = household_spec.composition
    matrix_shape = household_spec.matrix_shape
    sus = self.model_input.sus
    K_home = self.model_input.k_home
    inf_scales = copy(self.model_input.inf_scales)
    alpha_1 = self.model_input.alpha_1
    alpha_2 = self.model_input.alpha_2
    gamma = self.model_input.gamma
    iso_rates = deepcopy(self.model_input.iso_rates)
    discharge_rate = self.model_input.discharge_rate
    density_expo = self.model_input.density_expo


    class_is_isolating = self.model_input.class_is_isolating
    iso_method = self.model_input.iso_method
    adult_bd = self.model_input.adult_bd
    no_adults = sum(composition[adult_bd:])
    children_present = sum(composition[:adult_bd])>0

    # Set of individuals actually present here
    class_idx = household_spec.class_indexes

    K_home = K_home[ix_(class_idx, class_idx)]
    sus = sus[class_idx]
    r_home = atleast_2d(diag(sus).dot(K_home))
    for i in range(len(inf_scales)):
        inf_scales[i] = inf_scales[i][class_idx]

    states, \
        reverse_prod, \
        index_vector, \
        rows = build_state_matrix(household_spec)

    iso_pos = q_comp + no_compartments * arange(len(class_idx))
    if iso_method == "ext":
        # This is number of people of each age class present in the household
        # given some may isolate
        iso_adjusted_comp = composition[class_idx] - states[:,iso_pos]
        # Replace zeros with ones - we only ever use this as a denominator
        # whose numerator will be zero anyway if it should be zero
        iso_adjusted_comp[iso_adjusted_comp==0] = 1
        if (iso_adjusted_comp<1).any():
            pdb.set_trace()
    else:
        iso_adjusted_comp = \
            composition[class_idx] - zeros(states[:,iso_pos].shape)
    # Number of adults isolating by state
    adults_isolating = \
        states[:,no_compartments*adult_bd+q_comp::no_compartments].sum(axis=1)
    Q_int = sparse(household_spec.matrix_shape,)
    inf_event_row = array([], dtype=my_int)
    inf_event_col = array([], dtype=my_int)
    inf_event_class = array([], dtype=my_int)

    if iso_method == 'int':
        inf_comps = [p_comp, i_comp, q_comp]
    else:
        inf_comps = [p_comp, i_comp]

    if iso_method == 'ext':
        for cmp in range(len(iso_rates)):
            iso_rates[cmp] = \
                self.model_input.ad_prob * iso_rates[cmp][class_idx]
    else:
        for cmp in range(len(iso_rates)):
            iso_rates[cmp] = iso_rates[cmp][class_idx]

    Q_int, inf_event_row, inf_event_col, inf_event_class = \
                size_adj_inf_events(s_comp,
                                    e_comp,
                                    inf_comps,
                                    inf_scales,
                                    r_home,
                                    iso_adjusted_comp,
                                    density_expo,
                                    no_compartments,
                                    composition,
                                    states,
                                    index_vector,
                                    reverse_prod,
                                    class_idx,
                                    matrix_shape,
                                    Q_int,
                                    inf_event_row,
                                    inf_event_col,
                                    inf_event_class)
    Q_int = progression_events(e_comp,
                    p_comp,
                    alpha_1,
                    no_compartments,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = progression_events(p_comp,
                    i_comp,
                    alpha_2,
                    no_compartments,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = progression_events(i_comp,
                    r_comp,
                    gamma,
                    no_compartments,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = isolation_events(e_comp,
                    q_comp,
                    iso_rates[e_comp],
                    class_is_isolating,
                    iso_method,
                    adult_bd,
                    no_adults,
                    children_present,
                    adults_isolating,
                    no_compartments,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = isolation_events(p_comp,
                    q_comp,
                    iso_rates[p_comp],
                    class_is_isolating,
                    iso_method,
                    adult_bd,
                    no_adults,
                    children_present,
                    adults_isolating,
                    no_compartments,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = isolation_events(i_comp,
                    q_comp,
                    iso_rates[i_comp],
                    class_is_isolating,
                    iso_method,
                    adult_bd,
                    no_adults,
                    children_present,
                    adults_isolating,
                    no_compartments,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = progression_events(q_comp,
                    r_comp,
                    discharge_rate,
                    no_compartments,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)

    S = Q_int.sum(axis=1).getA().squeeze()
    Q_int += sparse((
        -S, (
            arange(household_spec.total_size),
            arange(household_spec.total_size)
        )))
    return tuple((
        Q_int,
        states,
        array(inf_event_row, dtype=my_int, ndmin=1),
        array(inf_event_col, dtype=my_int, ndmin=1),
        array(inf_event_class, dtype=my_int, ndmin=1),
        reverse_prod,
        index_vector))


def _sedur_subsystem(self, household_spec):
    '''This function processes a composition to create subsystems i.e.
    matrices and vectors describing all possible epdiemiological states
    for a given household composition
    Assuming frequency-dependent homogeneous within-household mixing
    composition[i] is the number of individuals in age-class i inside the
    household'''

    no_compartments = household_spec.no_compartments

    s_comp, e_comp, d_comp, u_comp, r_comp = range(no_compartments)

    composition = household_spec.composition
    matrix_shape = household_spec.matrix_shape
    sus = self.model_input.sus
    det = self.model_input.det
    inf_scales = copy(self.model_input.inf_scales)
    K_home = self.model_input.k_home
    alpha = self.model_input.alpha
    gamma = self.model_input.gamma
    density_expo = self.model_input.density_expo

    # Set of individuals actually present here
    class_idx = household_spec.class_indexes

    K_home = K_home[ix_(class_idx, class_idx)]
    sus = sus[class_idx]
    det = det[class_idx]
    r_home = atleast_2d(diag(sus).dot(K_home))
    for i in range(len(inf_scales)):
        inf_scales[i] = inf_scales[i][class_idx]

    states, \
        reverse_prod, \
        index_vector, \
        rows = build_state_matrix(household_spec)

    Q_int = sparse(household_spec.matrix_shape,)
    inf_event_row = array([], dtype=my_int)
    inf_event_col = array([], dtype=my_int)
    inf_event_class = array([], dtype=my_int)


    Q_int, inf_event_row, inf_event_col, inf_event_class = inf_events(s_comp,
                e_comp,
                [d_comp,u_comp],
                inf_scales,
                r_home,
                density_expo,
                no_compartments,
                composition,
                states,
                index_vector,
                reverse_prod,
                class_idx,
                matrix_shape,
                Q_int,
                inf_event_row,
                inf_event_col,
                inf_event_class)
    Q_int = stratified_progression_events(e_comp,
                    d_comp,
                    alpha*det,
                    no_compartments,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = stratified_progression_events(e_comp,
                    u_comp,
                    alpha*(1-det),
                    no_compartments,
                    states,
                    index_vector,
                    reverse_prod,
                    class_idx,
                    matrix_shape,
                    Q_int)
    Q_int = progression_events(d_comp,
        r_comp,
        gamma,
        no_compartments,
        states,
        index_vector,
        reverse_prod,
        class_idx,
        matrix_shape,
        Q_int)
    Q_int = progression_events(u_comp,
        r_comp,
        gamma,
        no_compartments,
        states,
        index_vector,
        reverse_prod,
        class_idx,
        matrix_shape,
        Q_int)

    S = Q_int.sum(axis=1).getA().squeeze()
    Q_int += sparse((
        -S, (
            arange(household_spec.total_size),
            arange(household_spec.total_size)
        )))
    return tuple((
        Q_int,
        states,
        array(inf_event_row, dtype=my_int, ndmin=1),
        array(inf_event_col, dtype=my_int, ndmin=1),
        array(inf_event_class, dtype=my_int, ndmin=1),
        reverse_prod,
        index_vector))


''' Entries in the subsystem key are in the following order: [list of
compartments, number of compartments, list of compartments which
contribute to infection, compartment corresponding to new infections].
'''

subsystem_key = {
'SIR' : [_sir_subsystem, 3, [1], 1],
'SEIR' : [_seir_subsystem, 4, [2], 1],
'SEPIR' : [_sepir_subsystem,5, [2,3], 1],
'SEPIRQ' : [_sepirq_subsystem,6, [2,3,5], 1],
'SEDUR' : [_sedur_subsystem,5, [2,3], 1],
}
