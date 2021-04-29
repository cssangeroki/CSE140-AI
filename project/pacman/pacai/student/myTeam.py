from pacai.util import reflection, counter, util
from pacai.agents.capture.capture import CaptureAgent
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions
import math
import random

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.student.myTeam.OffenseAgent',
        second = 'pacai.student.myTeam.DefenseAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = reflection.qualifiedImport(first)
    secondAgent = reflection.qualifiedImport(second)

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]

class DefenseAgent(ReflexCaptureAgent):

    def getFeatures(self, gameState, action):
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        borders = successor.getWalls()
        midWidth = math.floor(borders.getWidth() / 2)

        yValue = random.randint(0, borders.getHeight() - 1)
        midPoint = (midWidth, yValue)

        while successor.hasWall(midWidth, yValue):

            yValue = random.randint(0, borders.getHeight() - 1)
            midPoint = (midWidth, yValue)

        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if len(invaders) == 0:
            distanceToMid = self.getMazeDistance(myPos, midPoint)
            features['distanceToMiddle'] = distanceToMid
        else:
            features['distanceToMiddle'] = 0
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -100,
            'stop': -100,
            'reverse': -2,
            'distanceToMiddle': -10
        }

class OffenseAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)

        if self.index == min(self.getTeam(gameState)):
            self.offense = True
        else:
            self.offense = False

    def chooseAction(self, gameState):
        numAgents = self.getOpponents(gameState)
        numAgents = [self.index] + numAgents

        def alphaBeta(gameState, alpha, beta, depth, agent, action):
            if gameState.isOver() or depth < 0:
                return self.evaluationFunction(agent, gameState, action)
            if agent == self.index:
                return maxValue(gameState, alpha, beta, depth, agent)
            else:
                return minValue(gameState, alpha, beta, depth, agent)

        def maxValue(gameState, alpha, beta, depth, agent):
            v = float('-inf')
            legalMoves = gameState.getLegalActions(agent)
            for action in legalMoves:
                if action != Directions.STOP:
                    s = self.getSuccessor(agent, gameState, action)
                    a = numAgents[numAgents.index(agent) + 1]
                    v = max(v, alphaBeta(s, alpha, beta, depth, a, action))
                    if v >= beta:
                        return v
                    alpha = max(alpha, v)
            return v

        def minValue(gameState, alpha, beta, depth, agent):
            v = float('inf')
            a = numAgents[(numAgents.index(agent) + 1) % len(numAgents)]
            d = depth - 1 if a == self.index else depth
            legalMoves = gameState.getLegalActions(agent)
            for action in legalMoves:
                if action != Directions.STOP:
                    s = self.getSuccessor(agent, gameState, action)
                    v = min(v, alphaBeta(s, alpha, beta, d, a, action))
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
            return v

        alpha = float('-inf')
        beta = float('inf')
        v = float('-inf')
        act = None
        for action in gameState.getLegalActions(self.index):
            if action != Directions.STOP:
                oldV = v
                s = self.getSuccessor(self.index, gameState, action)
                ab = alphaBeta(s, alpha, beta, 1, self.index, action)
                v = max(v, ab)
                if v != oldV:
                    act = action
                if v >= beta:
                    return action
                alpha = max(alpha, v)

        return act

    def getSuccessor(self, agent, gameState, action):
        successor = gameState.generateSuccessor(agent, action)
        pos = successor.getAgentState(agent).getPosition()

        if (pos != util.nearestPoint(pos)):
            return successor.generateSuccessor(agent, action)
        else:
            return successor

    def getFeatures(self, agent, gameState, action):
        features = counter.Counter()

        successor = gameState
        features['successorScore'] = self.getScore(successor)

        food = self.getFood(successor)
        foodList = food.asList()

        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        opponents = self.getOpponents(gameState)
        agentState = gameState.getAgentState(self.index)
        myPos = agentState.getPosition()
        closeGhosts = 0
        for opponent in opponents:
            opponentPos = gameState.getAgentState(opponent).getPosition()
            if self.getMazeDistance(myPos, opponentPos) < 2:
                closeGhosts += 1
        features['closeGhosts'] = closeGhosts

        foodAtMyPos = food[int(myPos[0])][int(myPos[1])]
        if not closeGhosts and foodAtMyPos:
            features['eatFood'] = 1.0

        # ATTACK FEATURE
        capsules = self.getCapsules(gameState)
        powerPills = [self.getMazeDistance(myPos, pill) for pill in capsules]
        if len(powerPills) > 0:
            eatPill = min(powerPills)
        else:
            eatPill = 0

        if closeGhosts > 0:
            features['eatPill'] = eatPill
            features['eatFood'] = 0
        else:
            features['eatPill'] = 0

        return features

    def getWeights(self, gameState):
        return {
            'successorScore': 100,
            'distanceToFood': -1,
            'closeGhosts': 15,
            'eatFood': 50,
            'eatPill': -10
        }

    def evaluationFunction(self, agent, gameState, action):
        weights = counter.Counter(self.getWeights(gameState))
        features = self.getFeatures(agent, gameState, action)

        return weights * features
