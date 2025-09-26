import random
from xmlrpc.client import Binary
import pyomo.environ as pyo
import matplotlib.pyplot as plt


Ni = 1  # Number of instances (devices)
Mi = 5  # Number of models
C = 16  # Number of classes
Tmax = 200 # maximum time

model_sizes = {1: 5, 2: 10, 3: 20, 4: 25, 5: 30}
model_energy_device = model_sizes  # energy consumed by each model on each device (i.e., proportional to the model size)

M_max = max(model_sizes.values())  # maximum memory size of the devices

E_sel = 0  # energy needed to run the select model procedure  # NOTE: no energy for now !
Emax = max(model_energy_device.values()) + E_sel  # maximum energy should allow to run the largest model + select model procedure
E0_const = Emax  # initial energy is the maximum energy

model = pyo.ConcreteModel()

# Sets
model.I = pyo.Set(initialize=range(1, Ni+1), doc='Set of IoT devices')           # e.g., {1, 2, 3}
model.M = pyo.Set(initialize=range(1, Mi+1), doc='Set of tinyML models')         # e.g., {1, 2, ..., M}
model.C = pyo.Set(initialize=range(1, C+1), doc='Set of classes')                # e.g., {1, 2, ..., C}

model.T = pyo.Set(initialize=range(1, Tmax+1), ordered=True, doc='Set of timesteps')  # e.g., [1, ..., T]
model.T_all = pyo.Set(initialize=range(0, Tmax+1), ordered=True, doc='Set of timesteps')  # e.g., [0, 1, ..., T]


# ----------------------------------------------------------------
#
#
#                          Parameters
#
#
# ----------------------------------------------------------------

# size of the models
model.S_model = pyo.Param(model.M,
                          initialize=model_sizes,
                          within=pyo.NonNegativeReals,
                          doc='Size of model m')

# size of memory of the devices
memory_sizes = {i: M_max for i in range(1, Ni+1)}
model.Mi = pyo.Param(model.I,
                     initialize=memory_sizes,
                     within=pyo.NonNegativeReals,
                     doc='Initial memory of device i')

# energy consumed by model m on device i
energy_consumed = {(i, m): model_energy_device[m] for i in model.I for m in model.M}  # NOTE: same energy for all devices
model.Eim = pyo.Param(model.I, model.M,
                      initialize=energy_consumed,
                      within=pyo.NonNegativeReals,
                      doc='Energy consumed by model m on device i')

# energy consumed to load model m on device i
# NOTE: we assume that the energy consumed to load a model is the same as the energy consumed by the model
model.E_load = pyo.Param(model.I, model.M,
                         initialize=energy_consumed,
                         within=pyo.NonNegativeReals,
                         doc='Energy consumed to load model m on device i')


# energy needed to run the select model procedure on device i
model.Esel = pyo.Param(model.I,
                       initialize={i: E_sel for i in model.I},
                       within=pyo.NonNegativeReals,
                       doc='Energy needed to run the select model procedure on device i')

# maximum energy that can be stored on device i
Emax_device = {i: Emax for i in model.I}  # NOTE: same maximum energy for all devices
model.Emax = pyo.Param(model.I,
                       initialize=Emax_device,
                       within=pyo.NonNegativeReals,
                       doc='Maximum energy that can stored on device i')

# Confidence score of model m on device i at time t
# NOTE 1: in this simulation we assume that the confidence score is constant, i.e., it only depends on the model
# NOTE 2: the main idea is that we start by sampling Di from D at t=0, calculate the confidence score of each model on a sampled image from Di,
#         then we update Di, and repeat the process for t+1. However, this is too complex for now.
_min = min(model_sizes.values())
_max = max(model_sizes.values())
conf_model = {m : 0.5 + 0.5 * (model_sizes[m] - _min) / (_max - _min) for m in model.M}
print("Confidence by model: ", conf_model)
confidence = {(c, i, m, t): conf_model[m] for i in model.I for m in model.M for t in model.T for c in model.C}
model.gc = pyo.Param(model.C, model.I, model.M, model.T,
                     initialize=confidence,
                     within=pyo.UnitInterval,
                     doc='Confidence score at time t')

# Probability of appearance of class c in device i at time t
# NOTE 1: in this simulation we assume that the probability of appearance is constant and uniform
# NOTE 2: the main idea is that we start by sampling Di from D at t=0, calculate the probability of appearance of each class in Di,
#         then we update Di, and repeat the process for t+1. However, this is too complex for now.
#         This must follow the idea in model.gc.
prob_appearance = {(c, i, t): 1.0 / C for i in model.I for t in model.T for c in model.C}
model.P = pyo.Param(model.C, model.I, model.T,
                     initialize=prob_appearance,
                     within=pyo.UnitInterval,
                     doc='Probability of appearance of class c in device i at time t')

# Energy harvested at device i at time t
# NOTE: in this simulation we assume that the energy harvested is constant and is enough to run the largest model
energy_harvested = {(i, t): max(energy_consumed.values()) for i in model.I for t in model.T}
model.Eh = pyo.Param(model.I, model.T,
                     initialize=energy_harvested,
                     within=pyo.NonNegativeReals,
                     doc='Energy of device i at time t')

# rho[i, m]: monetary cost of energy on device i for model m
rho = {(i, m): energy_consumed[i, m] / Emax_device[i] for i in model.I for m in model.M}  # e.g., 0.01$ per MB
model.rho = pyo.Param(model.I, model.M,
                      initialize=rho,
                      within=pyo.NonNegativeReals,
                      doc='Monetary cost of energy on device i for model m')


# ----------------------------------------------------------------
#
#
#                          Variables
#
#
# ----------------------------------------------------------------

# Binary: model m is installed on device i at time t
model.im = pyo.Var(model.I, model.M, model.T_all, domain=pyo.Binary)

# NOTE: we install the largest model at the beginning of the simulation
# Fix install[i, m, 0] to initial values
print("Initial model selection for each device:")
for i in model.I:
    selection = []
    for m in model.M:
        val = 1 if m == Mi else 0
        model.im[i, m, 0].fix(val)
        selection.append(str(val))
    print(f"\tdevice {i}: {'-'.join(selection)}")

# model m is used for prediction at time t in device i
model.rm = pyo.Var(model.I, model.M, model.T, domain=pyo.Binary)

# model m is loaded in device i at time t
model.lm = pyo.Var(model.I, model.M, model.T, domain=pyo.Binary)

# whether the select model procedure is run on device i at time t
model.sp = pyo.Var(model.I, model.T, domain=pyo.Binary)

# whether the the final energy on device i at time t is above a certain threshold E_max
model.xi = pyo.Var(model.I, model.T, domain=pyo.Binary)

# energy available at time t in device i (continuous variable)
model.E = pyo.Var(model.I, model.T_all, domain=pyo.NonNegativeReals)
# Fix initial energy
print("Initial energy for each device:")
for i in model.I:
    model.E[i, 0].fix(E0_const)
    print(f"\tdevice {i}: {E0_const}")


# ----------------------------------------------------------------
#
#
#                          Constraints
#
#
# ----------------------------------------------------------------

# Equ (37)
def memory_constraint_rule(model, i, t):
    return sum(model.S_model[m] * model.im[i, m, t] for m in model.M) <= model.Mi[i]

model.MemoryConstraint = pyo.Constraint(model.I, model.T, rule=memory_constraint_rule)


# Equ (41)
def memory_availability_rule(model, i, t, m):
    return model.im[i, m, t] >= model.rm[i, m, t]

model.MemoryAvailability = pyo.Constraint(model.I, model.T, model.M, rule=memory_availability_rule)


# Equ (42)
def load_rule_1(model, i, t, m):
    return model.lm[i, m, t] >= model.im[i, m, t] - model.im[i, m, t-1]

model.LoadRule1 = pyo.Constraint(model.I, model.T, model.M, rule=load_rule_1)


# Equ (43)
def load_rule_2(model, i, t, m):
    return model.lm[i, m, t] <= model.im[i, m, t]

model.LoadRule2 = pyo.Constraint(model.I, model.T, model.M, rule=load_rule_2)

# Equ (44)
def load_rule_3(model, i, t, m):
    return model.lm[i, m, t] <= 1 - model.im[i, m, t-1]

model.LoadRule3 = pyo.Constraint(model.I, model.T, model.M, rule=load_rule_3)

# Equ (46)
def model_activation_gate(model, i, m, t):
    return model.rm[i, m, t] <= model.sp[i, t]

model.ModelGate = pyo.Constraint(model.I, model.M, model.T, rule=model_activation_gate)

# Equ (47)
def energy_to_run_rule(model, i, t):
    # NOTE: we assume that the energy needed to run the select model procedure is negligible for now
    #       because otherwise, we also need to check first if we have enough energy to run the select model procedure
    consumed_energy = model.sp[i, t] * model.Esel[i] + sum(model.Eim[i, m] * model.rm[i, m, t] for m in model.M)
    available_energy = model.E[i, t-1] + model.Eh[i, t]
    return consumed_energy <= available_energy

model.EnergyToRun = pyo.Constraint(model.I, model.T, rule=energy_to_run_rule)


# Energy update rule is more complex because there is a upper limit on the stored energy given by Emax
# Equ (52)
def energy_limits_rule1(model, i, t):
    return model.E[i, t] <= model.Emax[i]

model.EnergyLimits1 = pyo.Constraint(model.I, model.T, rule=energy_limits_rule1)

# Equ (53)
def energy_limits_rule2(model, i, t):
    consumed_energy = model.sp[i, t] * model.Esel[i] + sum(model.Eim[i, m] * model.rm[i, m, t] for m in model.M)
    available_energy = model.E[i, t-1] + model.Eh[i, t]
    return model.E[i, t] <= available_energy - consumed_energy

model.EnergyLimits2 = pyo.Constraint(model.I, model.T, rule=energy_limits_rule2)

# Equ (54)
def energy_limits_rule3(model, i, t):
    consumed_energy = model.sp[i, t] * model.Esel[i] + sum(model.Eim[i, m] * model.rm[i, m, t] for m in model.M)
    available_energy = model.E[i, t-1] + model.Eh[i, t]
    return model.E[i, t] >= available_energy - consumed_energy - model.xi[i, t] * model.Emax[i]

model.EnergyLimits3 = pyo.Constraint(model.I, model.T, rule=energy_limits_rule3)

# Equ (55)
def energy_limits_rule4(model, i, t):
    return model.E[i, t] >= model.Emax[i] - (1 - model.xi[i, t]) * model.Emax[i]

model.EnergyLimits4 = pyo.Constraint(model.I, model.T, rule=energy_limits_rule4)


def objective_rule(model):
    term1 = sum(
        model.P[c, i, t] * model.gc[c, i, m, t] * model.rm[i, m, t]
        for i in model.I for t in model.T for c in model.C for m in model.M for c in model.C
    )

    term2 = sum(
        model.gc[c, i, m, t] * model.im[i, m, t]
        for i in model.I for m in model.M for t in model.T for c in model.C
    )

    term3 = sum(
        model.rho[i, m] * model.lm[i, m, t]
        for i in model.I for m in model.M for t in model.T
    )

    return term1 + term2 - term3

model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


exit(0)  # still need to figure out some constraints to run the model


# Solver (make sure you have glpk or cbc installed)
solver = pyo.SolverFactory("glpk")

# Solve
results = solver.solve(model, tee=True)
# Check solver status
print(results.solver.status)
print(results.solver.termination_condition)

# Display results
model.display()

# Plot results
for i in model.I:
    energy = [pyo.value(model.E[i, t]) for t in model.T_all]
    plt.plot(model.T_all, energy, label=f'Device {i}')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy over Time')
plt.legend()
plt.grid()
plt.savefig('energy_over_time.png')
plt.close()
