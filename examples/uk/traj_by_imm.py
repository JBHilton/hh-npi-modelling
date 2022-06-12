'''This runs the UK-like model with a single set of parameters for 100 days for
a range of starting immunity levels.
'''
from numpy import array, log
from os import mkdir
from os.path import isdir, isfile
from pickle import load, dump
from pandas import read_csv
from scipy.integrate import solve_ivp
from model.preprocessing import ( estimate_beta_ext, estimate_growth_rate,
        SEPIRInput, HouseholdPopulation, make_initial_condition_by_eigenvector)
from model.specs import TWO_AGE_SEPIR_SPEC_FOR_FITTING, TWO_AGE_UK_SPEC
from model.common import SEPIRRateEquations
from model.imports import NoImportModel

if isdir('outputs/uk') is False:
    mkdir('outputs/uk')

DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

start_prev = 1e-2

# List of observed household compositions
composition_list = read_csv(
    'inputs/eng_and_wales_adult_child_composition_list.csv',
    header=0).to_numpy()
# Proportion of households which are in each composition
comp_dist = read_csv(
    'inputs/eng_and_wales_adult_child_composition_dist.csv',
    header=0).to_numpy().squeeze()


if isfile('outputs/uk/fitted_model_input.pkl') is True:
    with open('outputs/uk/fitted_model_input.pkl', 'rb') as f:
        model_input = load(f)
else:
    SPEC = {**TWO_AGE_SEPIR_SPEC_FOR_FITTING, **TWO_AGE_UK_SPEC}
    model_input_to_fit = SEPIRInput(SPEC, composition_list, comp_dist)
    household_population_to_fit = HouseholdPopulation(
        composition_list, comp_dist, model_input_to_fit)
    rhs_to_fit = SEPIRRateEquations(model_input_to_fit, household_population_to_fit, NoImportModel(5,2))
    beta_ext = estimate_beta_ext(household_population_to_fit, rhs_to_fit, growth_rate)
    model_input = model_input_to_fit
    model_input.k_ext *= beta_ext
    print('Estimated beta is',beta_ext)
    with open('outputs/uk/fitted_model_input.pkl', 'wb') as f:
        dump(model_input, f)

if isfile('outputs/uk/household_population.pkl') is True:
    with open('outputs/uk/household_population.pkl', 'rb') as f:
        household_population = load(f)
else:
    # With the parameters chosen, we calculate Q_int:
    household_population = HouseholdPopulation(
        composition_list, comp_dist, model_input)
    with open('outputs/uk/household_population.pkl', 'wb') as f:
        dump(household_population, f)

rhs = SEPIRRateEquations(model_input, household_population, NoImportModel(5,2))

H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, start_prev, 0.0)
tspan = (0.0, 365)
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)

time = solution.t
H = solution.y
S = H.T.dot(household_population.states[:, ::5])
E = H.T.dot(household_population.states[:, 1::5])
P = H.T.dot(household_population.states[:, 2::5])
I = H.T.dot(household_population.states[:, 3::5])
R = H.T.dot(household_population.states[:, 4::5])
time_series = {
'time':time,
'S':S,
'E':E,
'P':P,
'I':I,
'R':R
}

with open('outputs/uk/imm0_traj.pkl', 'wb') as f:
    dump((H, time_series), f)

H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, start_prev, 1e-6)
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)

time = solution.t
H = solution.y
S = H.T.dot(household_population.states[:, ::5])
E = H.T.dot(household_population.states[:, 1::5])
P = H.T.dot(household_population.states[:, 2::5])
I = H.T.dot(household_population.states[:, 3::5])
R = H.T.dot(household_population.states[:, 4::5])
time_series = {
'time':time,
'S':S,
'E':E,
'P':P,
'I':I,
'R':R
}

with open('outputs/uk/imm1e-6_traj.pkl', 'wb') as f:
    dump((H, time_series), f)

H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, start_prev, 1e-5)
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)

time = solution.t
H = solution.y
S = H.T.dot(household_population.states[:, ::5])
E = H.T.dot(household_population.states[:, 1::5])
P = H.T.dot(household_population.states[:, 2::5])
I = H.T.dot(household_population.states[:, 3::5])
R = H.T.dot(household_population.states[:, 4::5])
time_series = {
'time':time,
'S':S,
'E':E,
'P':P,
'I':I,
'R':R
}

with open('outputs/uk/imm1e-5_traj.pkl', 'wb') as f:
    dump((H, time_series), f)

H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, start_prev, 1e-4)
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)

time = solution.t
H = solution.y
S = H.T.dot(household_population.states[:, ::5])
E = H.T.dot(household_population.states[:, 1::5])
P = H.T.dot(household_population.states[:, 2::5])
I = H.T.dot(household_population.states[:, 3::5])
R = H.T.dot(household_population.states[:, 4::5])
time_series = {
'time':time,
'S':S,
'E':E,
'P':P,
'I':I,
'R':R
}

with open('outputs/uk/imm1e-4_traj.pkl', 'wb') as f:
    dump((H, time_series), f)

H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, start_prev, 1e-3)
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)

time = solution.t
H = solution.y
S = H.T.dot(household_population.states[:, ::5])
E = H.T.dot(household_population.states[:, 1::5])
P = H.T.dot(household_population.states[:, 2::5])
I = H.T.dot(household_population.states[:, 3::5])
R = H.T.dot(household_population.states[:, 4::5])
time_series = {
'time':time,
'S':S,
'E':E,
'P':P,
'I':I,
'R':R
}

with open('outputs/uk/imm1e-3_traj.pkl', 'wb') as f:
    dump((H, time_series), f)

H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, start_prev, 1e-2)
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)

time = solution.t
H = solution.y
S = H.T.dot(household_population.states[:, ::5])
E = H.T.dot(household_population.states[:, 1::5])
P = H.T.dot(household_population.states[:, 2::5])
I = H.T.dot(household_population.states[:, 3::5])
R = H.T.dot(household_population.states[:, 4::5])
time_series = {
'time':time,
'S':S,
'E':E,
'P':P,
'I':I,
'R':R
}

with open('outputs/uk/imm1e-2_traj.pkl', 'wb') as f:
    dump((H, time_series), f)

H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, start_prev, 1e-1)
solution = solve_ivp(rhs, tspan, H0, first_step=0.001, atol=1e-16)

time = solution.t
H = solution.y
S = H.T.dot(household_population.states[:, ::5])
E = H.T.dot(household_population.states[:, 1::5])
P = H.T.dot(household_population.states[:, 2::5])
I = H.T.dot(household_population.states[:, 3::5])
R = H.T.dot(household_population.states[:, 4::5])
time_series = {
'time':time,
'S':S,
'E':E,
'P':P,
'I':I,
'R':R
}

with open('outputs/uk/imm1e-1_traj.pkl', 'wb') as f:
    dump((H, time_series), f)
