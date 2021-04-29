from pacai.agents.learning.value import ValueEstimationAgent
from pacai.util import counter
import copy
class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = counter.Counter()  # A Counter is a dict with default 0

        # Compute the values here.

        # Number of iterations
        for i in range(self.iters):
            values_temp = copy.deepcopy(self.values)

            for state in self.mdp.getStates():
                Qvalue = counter.Counter()

                if self.mdp.isTerminal(state):
                    continue
                    # Qvalue = counter.Counter()

                for action in self.mdp.getPossibleActions(state):

                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):

                        R = self.mdp.getReward(state, action, nextState)
                        discount = self.discountRate
                        Qvalue[action] += prob * (R + (discount * (values_temp[nextState])))

                    self.values[state] = Qvalue[Qvalue.argMax()]
        # raise NotImplementedError()

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values[state]

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)

    def getQValue(self, state, action):
        Qvalue = 0
        T = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in T:
            R = self.mdp.getReward(state, action, nextState)
            Qvalue += prob * (R + (self.discountRate * (self.getValue(nextState))))

        return Qvalue

    def getPolicy(self, state):

        if self.mdp.isTerminal(state):
            return None

        possibleActions = self.mdp.getPossibleActions(state)

        Qvalue = counter.Counter()

        for action in possibleActions:
            Qvalue[action] = self.getQValue(state, action)

        return Qvalue.argMax()
