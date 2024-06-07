# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp  # GridWorld
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # V <- max(sum of T(R + rV))
        # Counter类是字典的子类，不同的是它会自动初始化内容为0.详细定义在util里面
        # 我还不是很懂，下面的代码其实是抄的
        # 我们就是要迭代：V = max_a U而已
        for i in range(self.iterations):
            valueForState = util.Counter()
            for state in self.mdp.getStates():  # 我们要更新目前地图上每一个位置的V
                valuesForActions = util.Counter()
                for action in self.mdp.getPossibleActions(state):   # 我们计算出每一个位置的Q值，Q值描述的是采取某个action之后能获得的收益
                    valuesForActions[action] = self.computeQValueFromValues(state, action)
                valueForState[state] = valuesForActions[valuesForActions.argMax()]  # 更新每个state的V值，都是max Q
            for state in self.mdp.getStates():
                self.values[state] = valueForState[state]   # 更新


    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Q value = T(R + r V)对s求和
        
        # 需要使用mdp.getTransitionStateAndProb()函数。
        # 这个函数Returns list of (nextState, prob) pairs
        QValue = 0
        transition = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in transition:
            QValue += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState))  # 刚开始我写成state了，这是不正确的
        return QValue

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        valuesForActions = util.Counter() 
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None # 我刚开始写成了return True，最要命的问题是debug的时候trace back不到这里，得亏是这个代码不是很复杂
        # 要根据V的值返回最优的策略，我们希望返回当前状态下的最优选择
        # 我们已经有一个self.value的Counter实例，我们可以根据每个位置的V值得到最佳的策略
        # 具体来说，我们遍历当前状态的所有的动作，计算它们的每个动作对应的Q值，然后我们就选择这些里面最大的动作
        for action in actions:
            valuesForActions[action] = self.computeQValueFromValues(state, action)

        return valuesForActions.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
            """
                * Please read learningAgents.py before reading this.*

                A PrioritizedSweepingValueIterationAgent takes a Markov decision process
                (see mdp.py) on initialization and runs prioritized sweeping value iteration
                for a given number of iterations using the supplied parameters.
            """
            def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
                """
                Your prioritized sweeping value iteration agent should take an mdp on
                construction, run the indicated number of iterations,
                and then act according to the resulting policy.
                """
                self.theta = theta
                ValueIterationAgent.__init__(self, mdp, discount, iterations)

            def runValueIteration(self):
                "*** YOUR CODE HERE ***"
    
