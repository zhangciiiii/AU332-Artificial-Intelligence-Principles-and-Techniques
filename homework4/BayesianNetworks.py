import numpy as np
import pandas as pd
from functools import reduce

## Function to create a conditional probability table
## Conditional probability is of the form p(x1 | x2, ..., xk)
## varnames: vector of variable names (strings) first variable listed 
##           will be x_i, remainder will be parents of x_i, p1, ..., pk
## probs: vector of probabilities for the flattened probability table
## outcomesList: a list containing a vector of outcomes for each variable
## factorTable is in the type of pandas dataframe
## See the test file for examples of how this function works
def readFactorTable(varnames, probs, outcomesList):
    factorTable = pd.DataFrame({'probs': probs})

    totalfactorTableLength = len(probs)
    numVars = len(varnames)

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(totalfactorTableLength / (k * numLevs))
        k = k * numLevs

    return factorTable

## Build a factorTable from a data frame using frequencies
## from a data frame of data to generate the probabilities.
## data: data frame read using pandas read_csv
## varnames: specify what variables you want to read from the table
## factorTable is in the type of pandas dataframe
def readFactorTablefromData(data, varnames):
    numVars = len(varnames)
    outcomesList = []

    for i in range(0, numVars):
        name = varnames[i]
        outcomesList = outcomesList + [list(set(data[name]))]

    lengths = list(map(lambda x: len(x), outcomesList))
    m = reduce(lambda x, y: x * y, lengths)
   
    factorTable = pd.DataFrame({'probs': np.zeros(m)})

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    numLevels = len(outcomesList[0])

    # creates the vector called fact to index probabilities 
    # using matrix multiplication with the data frame
    fact = np.zeros(data.shape[1])
    lastfact = 1
    for i in range(len(varnames) - 1, -1, -1):
        fact = np.where(np.isin(list(data), varnames[i]), lastfact, fact)
        lastfact = lastfact * len(outcomesList[i])

    # Compute unnormalized counts of subjects that satisfy all conditions
    a = (data - 1).dot(fact) + 1
    for i in range(0, m):
        factorTable.at[i,'probs'] = sum(a == (i+1))

    # normalize the conditional probabilities
    skip = int(m / numLevels)
    for i in range(0, skip):
        normalizeZ = 0
        for j in range(i, m, skip):
            normalizeZ = normalizeZ + factorTable['probs'][j]
        for j in range(i, m, skip):
            if normalizeZ != 0:
                factorTable.at[j,'probs'] = factorTable['probs'][j] / normalizeZ

    return factorTable


## Join of two factors
## factor1, factor2: two factor tables
##
## Should return a factor table that is the join of factor 1 and 2.
## You can assume that the join of two factors is a valid operation.
## Hint: You can look up pd.merge for mergin two factors
def joinFactors(factor1, factor2):
    # your code
    if ( factor1.empty) or ( factor2.empty):
        return ( factor1 if  factor2.empty else factor2)
    intersection = list((factor1.columns).intersection((factor2.columns)))
    intersection.remove('probs')
    # print(intersection)
    copy_factor1 = pd.DataFrame.copy(factor1)
    copy_factor2 = pd.DataFrame.copy(factor2)
    copy_factor1['bridge']=1
    copy_factor2['bridge']=1
    intersection.append('bridge')
    Factor = pd.merge(copy_factor1, copy_factor2, how='outer', on=intersection)
    # print(Factor)
    Factor['probs_x'] *= Factor['probs_y']
    # print(Factor)
    Factor = Factor.rename(columns={'probs_x':'probs'}).drop(columns=['probs_y','bridge'])
    return Factor

## Marginalize a variable from a factor
## table: a factor table in dataframe
## hiddenVar: a string of the hidden variable name to be marginalized
##
## Should return a factor table that marginalizes margVar out of it.
## Assume that hiddenVar is on the lecopy_factorTable side of the conditional.
## Hint: you can look can pd.groupby
def marginalizeFactor(factorTable, hiddenVar):
    # your code 
    copy_factorTable = pd.DataFrame.copy(factorTable)
    if  hiddenVar not in list(copy_factorTable.columns):
        # print('return raw')
        # print(list(copy_factorTable.columns),hiddenVar)
        return factorTable
    if hiddenVar in list(copy_factorTable.columns):
        # print(list(copy_factorTable.columns),hiddenVar)
        copy_factorTable = copy_factorTable.drop(columns=hiddenVar) # delete
        val_list = list(copy_factorTable.columns)
        val_list.remove('probs')
        if not val_list:
            return factorTable

        copy_factorTable = copy_factorTable[copy_factorTable.columns].groupby(val_list, as_index=False).mean()

        return copy_factorTable

## Marginalize a list of variables 
## bayesnet: a list of factor tables and each table in dataframe type
## hiddenVar: a string of the variable name to be marginalized
##
## Should return a Bayesian network containing a list of factor tables that results
## when the list of variables in hiddenVar is marginalized out of bayesnet.
def marginalizeNetworkVariables(bayesNet, hiddenVar):
    # your code 
    marginalized_bayesNet = []
    for factorTable in bayesNet:
        # print(factorTable)
        copy_factorTable = pd.DataFrame.copy(factorTable)
        for hiddenVar_x in hiddenVar:
            result = marginalizeFactor(copy_factorTable, hiddenVar_x)
        # print(factorTable)
            copy_factorTable = pd.DataFrame.copy(result)

        marginalized_bayesNet.append(result)
    return marginalized_bayesNet

## Update BayesNet for a set of evidence variables
## bayesNet: a list of factor and factor tables in dataframe format
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## Set the values of the evidence variables. Other values for the variables
## should be removed from the tables. You do not need to normalize the factors
def evidenceUpdateNet(bayesNet, evidenceVars, evidenceVals):
    # your code 
    '''
    no need to normalize the factors
    '''
    current_net = bayesNet.copy()
    for variable,value in zip(evidenceVars, evidenceVals):
        net_for_loop = current_net.copy()
        current_net = []       
        for factorTable in net_for_loop:
            if variable in factorTable.columns:
                factorTable = factorTable[factorTable[variable]==int(value)] # leave the corresponding variable with required value
                current_net.append(factorTable)
            else:
                current_net.append(factorTable)
    return current_net


## Run inference on a Bayesian network
## bayesNet: a list of factor tables and each table in dataframe type
## hiddenVar: a string of the variable name to be marginalized
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## This function should run variable elimination algorithm by using 
## join and marginalization of the sets of variables. 
## The order of the elimiation can follow hiddenVar ordering
## It should return a single joint probability table. The
## variables that are hidden should not appear in the table. The variables
## that are evidence variable should appear in the table, but only with the single
## evidence value. The variables that are not marginalized or evidence should
## appear in the table with all of their possible values. The probabilities
## should be normalized to sum to one.
def inference(bayesNet, hiddenVar, evidenceVars, evidenceVals):
    # your code 

    # update net with evidenceUpdateNet
    updated_net = evidenceUpdateNet(bayesNet,evidenceVars,evidenceVals)

    # filter all the variables
    all_variables = set()
    for factorTable in bayesNet:
        all_variables.update(factorTable.columns)
    all_variables.remove('probs')

    # for each variable in the net
    for variable in all_variables:
        copy_net = updated_net.copy()
        updated_net = []
        factor = pd.DataFrame(columns=['probs'])
        
        # for each table in the net
        for factorTable in copy_net:
            if variable in factorTable.columns:
                factor = joinFactors(factor, factorTable)
            else:
                updated_net.append(factorTable)
        if variable in hiddenVar:
            # if variable is in hiddenVar, whitch means it should be marginalized
            factor = marginalizeFactor(factor, variable)
        updated_net.append(factor)
    
    # normalization
    norm_scale = sum(list(factor['probs']))
    factor['probs'] /= norm_scale
    return factor
