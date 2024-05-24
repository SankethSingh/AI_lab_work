pip install pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianNetwork([('Burglary', 'Alarm'),
                         ('Earthquake', 'Alarm'),
                         ('Alarm', 'Davidcalls'),
                         ('Alarm', 'Sophiacalls')])

cpd_burglary = TabularCPD(variable='Burglary', variable_card=2, values=[[0.998], [0.002]])
cpd_earthquake = TabularCPD(variable='Earthquake', variable_card=2, values=[[0.999], [0.001]])

cpd_alarm = TabularCPD(variable='Alarm', variable_card=2,
                       values=[[0.999, 0.71, 0.06, 0.05],
                               [0.001, 0.29, 0.94, 0.95]],
                       evidence=['Burglary', 'Earthquake'],
                       evidence_card=[2, 2])

cpd_Davidcalls = TabularCPD(variable='Davidcalls', variable_card=2,
                           values=[[0.95, 0.1],
                                   [0.05, 0.9]],
                           evidence=['Alarm'],
                           evidence_card=[2])

cpd_Sophiacalls = TabularCPD(variable='Sophiacalls', variable_card=2,
                           values=[[0.99, 0.3],
                                   [0.01, 0.7]],
                           evidence=['Alarm'],
                           evidence_card=[2])
model.add_cpds(cpd_burglary, cpd_earthquake, cpd_alarm, cpd_Davidcalls, cpd_Sophiacalls)
assert model.check_model()
infer = VariableElimination(model)

prob_alarm_given_calls = infer.query(variables=['Alarm'], evidence={'Davidcalls': 1, 'Sophiacalls': 1, 'Earthquake': 0, 'Burglary': 0})
print(prob_alarm_given_calls)
