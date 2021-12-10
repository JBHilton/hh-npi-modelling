'''This runs the UK-like model with a single set of parameters for 100 days
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
# pylint: disable=invalid-name

if isdir('outputs/uk') is False:
    mkdir('outputs/uk')

DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

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

r_est = estimate_growth_rate(household_population, rhs, [0.001, 5], 1e-9)

print('Estimated growth rate is',r_est,'.')
print('Estimated doubling time is',log(2) / r_est,'.')

H0 = make_initial_condition_by_eigenvector(growth_rate, model_input, household_population, rhs, 1e-5, 0.0)
S0 = H0.T.dot(household_population.states[:, ::5])
E0 = H0.T.dot(household_population.states[:, 1::5])
P0 = H0.T.dot(household_population.states[:, 2::5])
I0 = H0.T.dot(household_population.states[:, 3::5])
R0 = H0.T.dot(household_population.states[:, 4::5])
start_state = (1/model_input.ave_hh_size) * array([S0.sum(),
                                                   E0.sum(),
                                                   P0.sum(),
                                                   I0.sum(),
                                                   R0.sum()])
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

with open('outputs/uk/results.pkl', 'wb') as f:
    dump((H, time_series), f)
