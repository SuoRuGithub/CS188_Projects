# factorOperations.py
# -------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from typing import List
from bayesNet import Factor
import functools
from util import raiseNotDefined

from copy import deepcopy   # 这个在实现问题3的时候是有用的

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factor)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()

########### ########### ###########
########### QUESTION 2  ###########
########### ########### ###########

def joinFactors(factors: List[Factor]):
    """
    Input factors is a list of factors.  
    
    # Factor是一组条件（包括它们的概率），我们需要将这些Factor的条件绑在一起
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    我们有若干变量，它们可以取一些值，assignment指的是一组确定的取值
    Factor.getAllPossibleAssignmentDicts    # 返回[assignmentDict]，其中assignmentDict形如{variable: aggignment}
    Factor.getProbability                   # 输入一个assignmentDict，返回这个组合下的概率
    Factor.setProbability                   # 设置一个assignmentDict对应的概率
    Factor.unconditionedVariables           # 返回当前factor下非条件变量
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))


    "*** YOUR CODE HERE ***"
    # 给的输入是一堆factor，每个Factor包含着一些条件，我们要做的是使用链式法则将它们联合起来，
    # 做法就是递归嘛，第一个先和第二个联合，在放进去，直到只剩下一个就可以了
    # 明明说好的factors是一个列表，怎么报错告诉我这是一个字典?
    factors = list(factors)
    factor0 = factors[0]

    for factor1 in factors[1:]:
        conditionedVar0 = factor0.conditionedVariables()    # 需要注意，这里返回的是一个set
        conditionedVar1 = factor1.conditionedVariables()
        UnconditionedVar0 = factor0.unconditionedVariables() 
        UnconditionedVar1 = factor1.unconditionedVariables() 
        # 新的conditioned variables应该是之前两者的并，新的unconditioned variables应该是之间两者的交
        # ConditionedVar = conditionedVar0
        # for var in conditionedVar1:
            # if var not in conditionedVar0:
                # ConditionedVar.append(var) 
        ConditionedVar = set([var for var in conditionedVar0 if var not in UnconditionedVar1] + 
                             [var for var in conditionedVar1 if var not in UnconditionedVar0])
        UnconditionedVar = UnconditionedVar0.union(UnconditionedVar1)
        # UnconditionedVar = set([var for var in UnconditionedVar0 if var in UnconditionedVar1])
        # 新的VariableDomainsDict应该是这样的：
        # 如果variable是0的条件，而不是1的，那么按0原来的走（反之同理）
        # 如果variable是0和1共同的条件，那么取它们的交集
        # 如果谁的条件都不是，看都不看
        VariableDict0 = factor0.variableDomainsDict()
        VariableDict1 = factor1.variableDomainsDict()
        VariableDomainsDict = {}
        # VariableDomainsDict应该包含所有变量可能的取值
        for var in ConditionedVar.union(UnconditionedVar):
            if var in conditionedVar0 or var in UnconditionedVar0:
                VariableDomainsDict[var]=VariableDict0[var]
            else:
                VariableDomainsDict[var]=VariableDict1[var]
        new_factor = Factor(UnconditionedVar, ConditionedVar, VariableDomainsDict)
        # 接下来就该设置新的factor的概率了
        for i in new_factor.getAllPossibleAssignmentDicts():
            prob=factor0.getProbability(i)*factor1.getProbability(i)
            new_factor.setProbability(i, prob)
        factor0 = new_factor
    return factor0
    # raiseNotDefined()
    "*** END YOUR CODE HERE ***"

########### ########### ###########
########### QUESTION 3  ###########
########### ########### ###########

def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor: Factor, eliminationVariable: str):
        """
        # 计算边缘概率。
        # 已经有给定的factor，也就是已知assignment的条件概率，现在就是对输入的eliminationVariable边缘化（放屁，不是边缘化，反而是边缘化的逆过程）
        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor. # ？学生朽木 ？ 
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))

        "*** YOUR CODE HERE ***"
        # 我们的主要任务就是：重新给出conditionedVar和UnconditionedVar,然后重新设置概率
        # 搅吧搅吧！我的脑子都要搅晕了。还是举个例子
        # 比如原来的Factor告诉我们，豌豆黄子叶的概率和绿子叶的概率，现在我们需要eliminate一下皱粒和圆粒
        # 首先要把圆粒/皱粒从非条件变量变成条件变量
        # 然后对于每一种现有的情况（比如圆粒），我们需要讨论它在eliminationVar变量上不同的取值，赋给新的factor的各个assignment
        # 整个过程就是：{0.75, 0.25} -> {0.5625, 0.1875, 0.1875, 0.0625}
        # P.S. 我真的都快被蠢笑了，这么点东西你是怎么琢磨一晚上还想不清楚的，只能说菜就多练吧
        VarDomains = factor.variableDomainsDict()   # 这个东西给出了是所有变量的值域
        eliminateDomains = VarDomains[eliminationVariable]    # 这个家伙是eliminationVariable所有可能的取值
        
        UnconditionedVar = deepcopy(factor.unconditionedVariables())
        ConditionedVar = deepcopy(factor.conditionedVariables())
        print(UnconditionedVar)
        print(ConditionedVar)
        UnconditionedVar.remove(eliminationVariable)
        # ConditionedVar.add(eliminationVariable)   # 什么情况？调了一晚上死活不对，注释掉这一行就可以了
        # print(UnconditionedVar)
        # print(ConditionedVar)
        # print(VarDomains)
        new_factor = Factor(UnconditionedVar, ConditionedVar, VarDomains)
        print(factor.getAllPossibleAssignmentDicts())
        print(new_factor.getAllPossibleAssignmentDicts())
        for assignments in deepcopy(new_factor.getAllPossibleAssignmentDicts()):
            # 我们现在在遍历所有可能的情况
            # 遍历它们的时候，我们再遍历一遍eliminationVar所有可能的取值，也就是对eliminationVar取边缘了
            prob = 0.
            for elimination in eliminateDomains:
                assignments[eliminationVariable] = elimination
                prob += factor.getProbability(assignments)
            del assignments[eliminationVariable]
            # print(assignments)
            new_factor.setProbability(assignments, prob)
        return new_factor
        # raiseNotDefined()
        "*** END YOUR CODE HERE ***"

    return eliminate

eliminate = eliminateWithCallTracking()

