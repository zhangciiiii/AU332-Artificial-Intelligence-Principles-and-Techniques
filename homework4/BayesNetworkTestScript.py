
from BayesianNetworks import *
import numpy as np
import pandas as pd

#############################
## Example Tests from Bishop Pattern recognition textbook on page 377
#############################
BatteryState = readFactorTable(['battery'], [0.9, 0.1], [[1, 0]])
FuelState = readFactorTable(['fuel'], [0.9, 0.1], [[1, 0]])
GaugeBF = readFactorTable(['gauge', 'battery', 'fuel'], [0.8, 0.2, 0.2, 0.1, 0.2, 0.8, 0.8, 0.9], [[1, 0], [1, 0], [1, 0]])

carNet = [BatteryState, FuelState, GaugeBF] # carNet is a list of factors 
## Notice that different order of operations give the same answer
## (rows/columns may be permuted)
joinFactors(joinFactors(BatteryState, FuelState), GaugeBF)
joinFactors(joinFactors(GaugeBF, FuelState), BatteryState)

marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'gauge')
joinFactors(marginalizeFactor(GaugeBF, 'gauge'), BatteryState)

joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState)
marginalizeFactor(joinFactors(joinFactors(GaugeBF, FuelState), BatteryState), 'battery')

marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'gauge')
marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'fuel')

evidenceUpdateNet(carNet, ['fuel'], [1])
evidenceUpdateNet(carNet, ['fuel', 'battery'], [1, 0])

## Marginalize must first combine all factors involving the variable to
## marginalize. Again, this operation may lead to factors that aren't
## probabilities.
marginalizeNetworkVariables(carNet, ['battery']) ## this returns back a list
marginalizeNetworkVariables(carNet, ['fuel']) ## this returns back a list
marginalizeNetworkVariables(carNet, ['battery', 'fuel'])

# inference
print("inference starts")
print(inference(carNet, ['battery', 'fuel'], [], []) )        ## chapter 8 equation (8.30)
print(inference(carNet, ['battery'], ['fuel'], [0]))           ## chapter 8 equation (8.31)
print(inference(carNet, ['battery'], ['gauge'], [0]))          ##chapter 8 equation  (8.32)
print(inference(carNet, [], ['gauge', 'battery'], [0, 0]))    ## chapter 8 equation (8.33)
print("inference ends")
###########################################################################
#RiskFactor Data Tests
###########################################################################
riskFactorNet = pd.read_csv('C:\important data\study\Artificial Intelligence\homework4\RiskFactorsData.csv')

# Create factors

income      = readFactorTablefromData(riskFactorNet, ['income'])
smoke       = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise    = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
bmi         = readFactorTablefromData(riskFactorNet, ['bmi', 'income'])
diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi']) # in the right are the conditions.

## you need to create more factor tables

risk_net = [income, smoke, exercise, bmi, diabetes]
print("income dataframe is ")
print(income)
factors = riskFactorNet.columns

# example test p(diabetes|smoke=1,exercise=2)

margVars = list(set(factors) - {'diabetes', 'smoke', 'exercise'})
obsVars  = ['smoke', 'exercise']
obsVals  = [1, 2]

p = inference(risk_net, margVars, obsVars, obsVals)
print(p)


### Please write your own test scrip similar to  the previous example 
###########################################################################
#HW4 test scrripts start from here
###########################################################################

riskFactorNet = pd.read_csv('C:\important data\study\Artificial Intelligence\homework4\RiskFactorsData.csv')
income      = readFactorTablefromData(riskFactorNet, ['income'])
smoke       = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise    = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
bmi         = readFactorTablefromData(riskFactorNet, ['bmi', 'income','exercise'])
bp   = readFactorTablefromData(riskFactorNet, ['bp','exercise', 'income', 'smoke'])
cholesterol = readFactorTablefromData(riskFactorNet, ['cholesterol','exercise', 'income', 'smoke'])
diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])
stroke = readFactorTablefromData(riskFactorNet, ['stroke','bmi', 'bp', 'cholesterol'])
attack = readFactorTablefromData(riskFactorNet, ['attack','bmi', 'bp', 'cholesterol'])
angina = readFactorTablefromData(riskFactorNet, ['angina','bmi', 'bp', 'cholesterol'])

factors = riskFactorNet.columns
print('===========================================================')
# T2 -----------------------------------------------------
def T2_bad_good_habits(if_good = 'bad'):
    # for bad habits:
    obsVars  = ['smoke', 'exercise']
    if if_good is 'good':
        obsVals  = [2, 1]
    else:
        obsVals  = [1, 2]

    # diabetes: 
    margVars = list(set(factors) - {'diabetes', 'smoke', 'exercise'})
    diabetes_net = [income, smoke, exercise, bmi, diabetes]
    diabetes_result = inference(diabetes_net, margVars, obsVars, obsVals)

    # stroke:
    margVars = list(set(factors) - {'stroke', 'smoke', 'exercise'})
    stroke_net = [income, smoke, exercise, bmi, bp, cholesterol, stroke]
    stroke_result = inference(stroke_net, margVars, obsVars, obsVals)

    # heart attack:
    margVars = list(set(factors) - {'attack', 'smoke', 'exercise'})
    attack_net = [income, smoke, exercise, bmi, bp, cholesterol, attack]
    attack_result = inference(attack_net, margVars, obsVars, obsVals)

    # angina:
    margVars = list(set(factors) - {'angina', 'smoke', 'exercise'})
    angina_net = [income, smoke, exercise, bmi, bp, cholesterol, angina]
    angina_result = inference(angina_net, margVars, obsVars, obsVals)

    print('with {} habits:'.format(if_good))
    print('diabetes:')
    print(diabetes_result)
    print('stroke:')
    print(stroke_result)
    print('attack:')
    print(attack_result)
    print('angina:')
    print(angina_result)
    print('---------------------------------')

def T2_poor_good_health(if_good = 'pool'):
    # for bad habits:
    obsVars  = ['bp', 'cholesterol','bmi']

    if if_good is 'good':
        obsVals = [3,2,2]
    else:
        obsVals = [1,1,3]

    # diabetes: 
    margVars = list(set(factors) - {'diabetes', 'bp', 'cholesterol','bmi'})
    diabetes_net = [income, smoke, exercise, bmi, diabetes]
    diabetes_result = inference(diabetes_net, margVars, obsVars, obsVals)


    # stroke:
    margVars = list(set(factors) - {'stroke', 'bp', 'cholesterol','bmi'})
    stroke_net = [income, smoke, exercise, bmi, bp, cholesterol, stroke]
    stroke_result = inference(stroke_net, margVars, obsVars, obsVals)

    # heart attack:
    margVars = list(set(factors) - {'attack', 'bp', 'cholesterol','bmi'})
    attack_net = [income, smoke, exercise, bmi, bp, cholesterol, attack]
    attack_result = inference(attack_net, margVars, obsVars, obsVals)

    # angina:
    margVars = list(set(factors) - {'angina', 'bp', 'cholesterol','bmi'})
    angina_net = [income, smoke, exercise, bmi, bp, cholesterol, angina]
    angina_result = inference(angina_net, margVars, obsVars, obsVals)

    print('with {} health:'.format(if_good))
    print('diabetes:')
    print(diabetes_result)
    print('stroke:')
    print(stroke_result)
    print('attack:')
    print(attack_result)
    print('angina:')
    print(angina_result)
    print('---------------------------------')

print("T2:")
T2_bad_good_habits('bad')
T2_bad_good_habits('good')

T2_poor_good_health('pool')
T2_poor_good_health('good')

# T3 -----------------------------------------------------
def T3_income_and_health():
    diabetes_list = []
    stroke_list = []
    attack_list = []
    angina_list = []
    for income_status in range(1,9):
        obsVars = ['income']
        obsVals = [income_status]

        # diabetes
        margVars = list(set(factors) - {'diabetes', 'income'})
        diabetes_net = [income, smoke, exercise, bmi, diabetes]
        diabetes_result = inference(diabetes_net, margVars, obsVars, obsVals)
        diabetes_list.append(float(diabetes_result[diabetes_result['diabetes']==1]['probs']))
        
        # stroke:
        margVars = list(set(factors) - {'stroke', 'income'})
        stroke_net = [income, smoke, exercise, bmi, bp, cholesterol, stroke]
        stroke_result = inference(stroke_net, margVars, obsVars, obsVals)
        stroke_list.append(float(stroke_result[stroke_result['stroke']==1]['probs']))

        # heart attack:
        margVars = list(set(factors) - {'attack', 'income'})
        attack_net = [income, smoke, exercise, bmi, bp, cholesterol, attack]
        attack_result = inference(attack_net, margVars, obsVars, obsVals)
        attack_list.append(float(attack_result[attack_result['attack']==1]['probs']))

        # angina:
        margVars = list(set(factors) - {'angina', 'income'})
        angina_net = [income, smoke, exercise, bmi, bp, cholesterol, angina]
        angina_result = inference(angina_net, margVars, obsVars, obsVals)
        angina_list.append(float(angina_result[angina_result['angina']==1]['probs']))

    import matplotlib.pyplot as plt
    index = list(range(1,len(diabetes_list)+1))
    print('income:',index)
    print('diabetes:',diabetes_list)
    print('stroke:',stroke_list)
    print('attack:',attack_list)
    print('angina:',angina_list)
    l1,=plt.plot(index,diabetes_list)
    l2,=plt.plot(index,stroke_list)
    l3,=plt.plot(index,attack_list)
    l4,=plt.plot(index,angina_list)
    plt.legend(handles = [l1, l2, l3, l4], labels = ['diabetes', 'stroke', 'attack', 'angina'], loc = 'best')
    plt.ylabel('probability') 
    plt.xlabel('income status')
    plt.show()

print('T3:')
T3_income_and_health()

# T4 -----------------------------------------------------
def T4_habits_edge_habit_outcome(if_good = 'bad'):
    diabetes = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi', 'exercise', 'smoke'])
    stroke = readFactorTablefromData(riskFactorNet, ['stroke','bmi', 'bp', 'cholesterol', 'exercise', 'smoke'])
    attack = readFactorTablefromData(riskFactorNet, ['attack','bmi', 'bp', 'cholesterol', 'exercise', 'smoke'])
    angina = readFactorTablefromData(riskFactorNet, ['angina','bmi', 'bp', 'cholesterol', 'exercise', 'smoke'])
    obsVars  = ['smoke', 'exercise']
    if if_good is 'good':
        obsVals  = [2, 1]
    else:
        obsVals  = [1, 2]

    # diabetes: 
    margVars = list(set(factors) - {'diabetes', 'smoke', 'exercise'})
    diabetes_net = [income, smoke, exercise, bmi, bp, cholesterol, diabetes]
    diabetes_result = inference(diabetes_net, margVars, obsVars, obsVals)

    # stroke:
    margVars = list(set(factors) - {'stroke', 'smoke', 'exercise'})
    stroke_net = [income, smoke, exercise, bmi, bp, cholesterol, stroke]
    stroke_result = inference(stroke_net, margVars, obsVars, obsVals)

    # heart attack:
    margVars = list(set(factors) - {'attack', 'smoke', 'exercise'})
    attack_net = [income, smoke, exercise, bmi, bp, cholesterol, attack]
    attack_result = inference(attack_net, margVars, obsVars, obsVals)

    # angina:
    margVars = list(set(factors) - {'angina', 'smoke', 'exercise'})
    angina_net = [income, smoke, exercise, bmi, bp, cholesterol, angina]
    angina_result = inference(angina_net, margVars, obsVars, obsVals)

    print('with {} habits:'.format(if_good))
    print('diabetes:')
    print(diabetes_result)
    print('stroke:')
    print(stroke_result)
    print('attack:')
    print(attack_result)
    print('angina:')
    print(angina_result)
    print('---------------------------------')

def T4_health_edge_habit_outcome(if_good = 'pool'):
    diabetes = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi', 'exercise', 'smoke'])
    stroke = readFactorTablefromData(riskFactorNet, ['stroke','bmi', 'bp', 'cholesterol', 'exercise', 'smoke'])
    attack = readFactorTablefromData(riskFactorNet, ['attack','bmi', 'bp', 'cholesterol', 'exercise', 'smoke'])
    angina = readFactorTablefromData(riskFactorNet, ['angina','bmi', 'bp', 'cholesterol', 'exercise', 'smoke'])
    # for bad habits:
    obsVars  = ['bp', 'cholesterol','bmi']

    if if_good is 'good':
        obsVals = [3,2,2]
    else:
        obsVals = [1,1,3]

    # diabetes: 
    margVars = list(set(factors) - {'diabetes', 'bp', 'cholesterol','bmi'})
    diabetes_net = [income, smoke, exercise, bmi, bp, cholesterol, diabetes]
    diabetes_result = inference(diabetes_net, margVars, obsVars, obsVals)

    # stroke:
    margVars = list(set(factors) - {'stroke', 'bp', 'cholesterol','bmi'})
    stroke_net = [income, smoke, exercise, bmi, bp, cholesterol, stroke]
    stroke_result = inference(stroke_net, margVars, obsVars, obsVals)

    # heart attack:
    margVars = list(set(factors) - {'attack', 'bp', 'cholesterol','bmi'})
    attack_net = [income, smoke, exercise, bmi, bp, cholesterol, attack]
    attack_result = inference(attack_net, margVars, obsVars, obsVals)

    # angina:
    margVars = list(set(factors) - {'angina', 'bp', 'cholesterol','bmi'})
    angina_net = [income, smoke, exercise, bmi, bp, cholesterol, angina]
    angina_result = inference(angina_net, margVars, obsVars, obsVals)

    print('with {} health:'.format(if_good))
    print('diabetes:')
    print(diabetes_result)
    print('stroke:')
    print(stroke_result)
    print('attack:')
    print(attack_result)
    print('angina:')
    print(angina_result)
    print('---------------------------------')
print('T4:')
T4_habits_edge_habit_outcome('bad')
T4_habits_edge_habit_outcome('good')

T4_health_edge_habit_outcome('pool')
T4_health_edge_habit_outcome('good')


def T5_edge_between_outcome():
    stroke = readFactorTablefromData(riskFactorNet, ['stroke','bmi', 'bp', 'cholesterol', 'exercise', 'smoke'])
    diabetes = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi', 'exercise', 'smoke'])
    # stroke:
    margVars = list(set(factors) - {'stroke', 'diabetes'})
    stroke_net = [income, smoke, exercise, bmi, bp, cholesterol, stroke, diabetes]
    obsVars  = ['diabetes']
    obsVals = [1]
    stroke_result = inference(stroke_net, margVars, obsVars, obsVals)
    print('P(stroke = 1|diabetes = 1)')
    print(stroke_result)
    obsVals = [3]
    stroke_result = inference(stroke_net, margVars, obsVars, obsVals)
    print('P(stroke = 1|diabetes = 3)')
    print(stroke_result)

    print('\nAdd an edge from diabetes to stroke\n')

    stroke = readFactorTablefromData(riskFactorNet, ['stroke','bmi', 'bp', 'cholesterol', 'exercise', 'smoke','diabetes'])
    diabetes = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi', 'exercise', 'smoke'])
    # stroke:
    margVars = list(set(factors) - {'stroke', 'diabetes'})
    stroke_net = [income, smoke, exercise, bmi, bp, cholesterol, stroke, diabetes]
    obsVars  = ['diabetes']
    obsVals = [1]
    stroke_result = inference(stroke_net, margVars, obsVars, obsVals)
    print('P(stroke = 1|diabetes = 1)')
    print(stroke_result)
    obsVals = [3]
    stroke_result = inference(stroke_net, margVars, obsVars, obsVals)
    print('P(stroke = 1|diabetes = 3)')
    print(stroke_result)
    print('---------------------------------')
    
print('T5')
T5_edge_between_outcome()

