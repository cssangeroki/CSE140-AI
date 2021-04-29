from pacai.util import reflection, counter, util
from pacai.agents.capture.capture import CaptureAgent
import random
from pacai.core.directions import Directions
import math

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.student.myTeam.OffenseAgent',
        second = 'pacai.agents.capture.defense.DefensiveReflexAgent'):
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

class OffenseAgent(CaptureAgent):

    def chooseAction(self, gameState):
        numAgents = self.getOpponents(gameState)
        numAgents = [self.index] + numAgents

        def alphaBeta(gameState, alpha, beta, depth, agent):
            if gameState.isOver() or depth < 0:
                return self.evaluationFunction(gameState)
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
                    v = max(v, alphaBeta(s, alpha, beta, depth, a))
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
                    v = min(v, alphaBeta(s, alpha, beta, d, a))
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
                ab = alphaBeta(s, alpha, beta, 2, self.index)
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

    def getFeatures(self, state):
        food = self.getFood(state)
        capsules = self.getCapsules(state)
        features = counter.Counter()

        features['score'] = self.getScore(state)

        opponents = self.getOpponents(state)
        myState = state.getAgentState(self.index)
        myPos = myState.getPosition()
        closeGhosts = 0
        for opponent in opponents:
            opponentPos = state.getAgentState(opponent).getPosition()
            if self.getMazeDistance(myPos, opponentPos) < 2:
                closeGhosts += 1
        features['#-of-ghosts-1-step-away'] = closeGhosts

        foodAtMyPos = food[int(myPos[0])][int(myPos[1])]
        if not closeGhosts and foodAtMyPos:
            features['eats-food'] = 10.0

        foodDist = []
        for x in range(food.getWidth()):
            for y in range(food.getHeight()):
                if food[x][y]:
                    foodDist.append(self.getMazeDistance(myPos, (x, y)))
        features['closest-food'] = min(foodDist)

        # ATTACK FEATURE
        powerPills = [self.getMazeDistance(myPos, pill) for pill in capsules]
        if len(powerPills) > 0:
            eatPill = min(powerPills)
        else:
            eatPill = 0

        if closeGhosts > 0:
            features['eat-Pill'] = eatPill
            features['eats-food'] = 0
        else:
            features['eat-Pill'] = 0

        return features

    def getWeights(self, state):
        return {
            'score': 100,
            'closest-food': -1,
            '#-of-ghosts-1-step-away': 15,
            'eats-food': 50,
            'eat-Pill': -10
        }

    def evaluationFunction(self, gameState):
        weights = counter.Counter(self.getWeights(gameState))
        features = self.getFeatures(gameState)
        return weights * features

class DefenseAgent(CaptureAgent):

    def chooseAction(self, gameState):
        numAgents = self.getOpponents(gameState)
        numAgents = [self.index] + numAgents

        def alphaBeta(gameState, alpha, beta, depth, agent):
            if gameState.isOver() or depth < 0:
                return self.evaluationFunction(gameState)
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
                    v = max(v, alphaBeta(s, alpha, beta, depth, a))
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
                    v = min(v, alphaBeta(s, alpha, beta, d, a))
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
                ab = alphaBeta(s, alpha, beta, 2, self.index)
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

    def getFeatures(self, gameState, action):
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        borders = successor.getWalls()
        midWidth = math.floor(borders.width / 2)

        yValue = random.randint(0, successor.getWalls().height)

        while successor.hasWall(midWidth, yValue):

            yValue = random.randint(0, successor.getWalls().height)
            midPoint = (midWidth, yValue)

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
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

    def evaluationFunction(self, gameState):
        weights = counter.Counter(self.getWeights(gameState))
        features = self.getFeatures(gameState)
        return weights * features
