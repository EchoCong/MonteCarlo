from captureAgents import CaptureAgent
from game import Directions
import random, time, util
from util import nearestPoint

# AGENTS   #
###############################################
# Implementacoes dos agentes.                 #
# Devem estender a classe base CaptureAgent.  #
###############################################

class EvaluationBasedAgent(CaptureAgent):
    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 1.0}


# #   AGENTS   #
# #####################
# # Attacker Agent    #
# #####################

class Attacker(EvaluationBasedAgent):
    "Gera Carlo, o agente ofensivo."

    def getFeatures(self, gameState, action):
        """
        Get features used for state evaluation.
        1. Distance to Capsule.
        2. Successor Score.
        3. Distance to Ghost.
        4. Distance to Food.
        5*. Food Count in View.
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()

        # Compute score from successor state
        features['successorScore'] = self.getScore(successor) - self.getScore(gameState)

        # Compute distance to capsule
        capsules = self.getCapsules(successor)
        if len(capsules) > 0:
            nearestCapsuleDist = min(self.getMazeDistance(myPos, capsule) for capsule in capsules)
            features["distanceToCapsule"] = nearestCapsuleDist

        # Compute distance to the nearest food
        nearestFood, nearestFoodDist = self.getNearestFood(successor)
        if len(nearestFood) > 0:
            features['distanceToFood'] = nearestFoodDist

        # Compute distance to closest ghost
        nearestGhost, nearestGhostDist = self.getNearestGhost(successor)
        if nearestGhostDist is not None and nearestGhostDist <= 5:
            features['distanceToGhost'] = nearestGhostDist

        return features

    def getWeights(self, gameState, action):
        """
        Get weights for the features used in the evaluation.
        if self.getCapsules(gameState)!= None:
            capsuleWeight:
                a. avoiding ghost (Distance to Ghost)
                b. eating capsule (Distance to Capsule)
                c. do not eat food (except in the path) (Distance to food = 0).
            As long as eating capsule, greedy eat foods (both offensiveAgent and defensiveAgent).
        else:
            defaultWeight:
                a. avoiding ghost (Distance to Ghost)
                b. adding score (Successor Score)
                c. eat foods (Distance to food)
                d. capsule (Distance to Capsule = 0)
        """
        successor = self.getSuccessor(gameState, action)

        # If capsule in enemy field exists, chasing capsule regardless foods.
        if self.getCapsules(gameState) is not None:
            return {'distanceToCapsule': -5, 'distanceToGhost': 6}

        # If opponent is scared, the agent should not care about distanceToGhost

        # #########################
        # Now our pacman will eat all capsules first, waste of capsules.
        # #########################

        nearestGhost, nearestGhostDist = self.getNearestGhost(successor)
        for agent in nearestGhost:
            if agent[1].scaredTimer > 5:
                return {'successorScore': 7, 'distanceToFood': -6}

        # Weights normally used
        return {'successorScore': 10, 'distanceToFood': -5, 'distanceToGhost': 2}

    def getNearestGhost(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        ghostsAll = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghostsInRange = filter(lambda x: not x.isPacman and x.getPosition() is not None, ghostsAll)
        ghostsPos = [ghost.getPosition() for ghost in ghostsInRange]
        if len(ghostsInRange) > 0:
            nearestGhostDist = min(self.getMazeDistance(myPos, ghost) for ghost in ghostsPos)
            nearestGhostPos, nearestGhost = filter(lambda x: x[0] == nearestGhostDist, zip(ghostsPos, ghostsInRange))
            return nearestGhost, nearestGhostDist
        return None, None

    def getNearestFood(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        foods = self.getFood(gameState).asList()
        if len(foods) > 0:
            return min([(self.getMazeDistance(myPos, food), food) for food in foods])

    def uctSimulation(self, depth, gameState):
        """
        SIMULATE and BACK UP STEP in Monte Carlo Search Tree:

        Random simulate some actions for the agent. The actions other agents can take
        are ignored, or, in other words, we consider their actions is always STOP.
        The final state from the simulation is evaluated.

        Should think how to implement algorithm.
        return evaluateValue
        """
        new_state = gameState.deepCopy()
        while depth > 0:
            # Get valid actions
            actions = new_state.getLegalActions(self.index)
            # The agent should not stay put in the simulation
            actions.remove(Directions.STOP)
            current_direction = new_state.getAgentState(self.index).configuration.direction
            # The agent should not use the reverse direction during simulation
            reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            # Randomly chooses a valid action
            a = random.choice(actions)
            # Compute new state and update depth
            new_state = new_state.generateSuccessor(self.index, a)
            depth -= 1
        # Evaluate the final simulation state
        return self.evaluate(new_state, Directions.STOP)

    def takeToEmptyAlley(self, gameState, action, depth):
        """
        Verify if an action takes the agent to an alley with
        no pacdots.
        """
        if depth == 0:
            return False
        old_score = self.getScore(gameState)
        new_state = gameState.generateSuccessor(self.index, action)
        new_score = self.getScore(new_state)
        if old_score < new_score:
            return False
        actions = new_state.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
        if reversed_direction in actions:
            actions.remove(reversed_direction)
        if len(actions) == 0:
            return True
        for a in actions:
            if not self.takeToEmptyAlley(new_state, a, depth - 1):
                return False
        return True

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        # Variables used to verify if the agent os locked
        self.numEnemyFood = "+inf"
        self.inactiveTime = 0

    # Implemente este metodo para pre-processamento (15s max).
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()

    def pureEnvBFS(self, gameState, goalPos):
        """
        Called when otherAgentState.scaredTimer > 5 (goal is nearest food)
         or otherAgent.isPacman in view && no mate around (goal is pacman):
        Use A Star Algorithm to get the best next action to eat the nearest dot.
        Using MazeDistance as heuristic value.
        """
        myPos = gameState.getAgentState(self.index).getPosition()
        # ######################################
        # try to solve with BFS
        # ######################################
        queue, expended, path = util.Queue(), [], []
        queue.push((myPos, path))

        while not queue.isEmpty():
            currPos, path = queue.pop()
            expended.append(currPos)
            if currPos is goalPos:
                return path[0]
            actions = gameState.getLegalActions(self.index)
            for action in actions:
                succPos = gameState.generateSuccessor(self.index, action).getAgentState(self.index).getPosition()
                if not succPos in expended:
                    expended.append(succPos)
                    queue.push((succPos, path + [action]))
        return []

        # #######################################
        # try to solve with A star, but no function queue.update in this version util file
        # #######################################
        # queue, expended, path = util.PriorityQueue(), [], []
        # queue.push((myPos, []), 0.0)
        #
        # while not queue.isEmpty():
        #     currPos, path = queue.pop()
        #     if currPos is goalPos:
        #         return path[0]
        #     expended.append(currPos)
        #     actions = gameState.getLegalActions(self.index)
        #     for action in actions:
        #         succPos = gameState.generateSuccessor(self.index, action).getAgentState(self.index).getPosition()
        #         if succPos is goalPos or not succPos in expended:
        #             expended.append(succPos)
        #             queue.update((succPos, path + [action]),
        #                          len(path + [action]) + self.getMazeDistance(succPos, goalPos))
        # return []


    def getRationalActions(self, gameState):
        """
        EXPAND Step in Monte Carlo Search Tree:

        actions = gameState.getLegalActions(self.index)
        aBooleanValue = takeToEmptyAlley(self, gameState, action, depth)
        actions.remove(Directions.STOP and action lead pacman into a empty alley)
        """

        # Get valid actions. Staying put is almost never a good choice, so
        # the agent will ignore this action.
        all_actions = gameState.getLegalActions(self.index)
        all_actions.remove(Directions.STOP)
        actions = []
        for a in all_actions:
            # if not self.takeToEmptyAlley(gameState, a, 5):
            actions.append(a)
        if len(actions) == 0:
            actions = all_actions

        fvalues = []
        for a in actions:
            new_state = gameState.generateSuccessor(self.index, a)
            value = 0
            for i in range(1, 31):
                value += self.uctSimulation(10, new_state)
            fvalues.append(value)

        best = max(fvalues)
        ties = filter(lambda x: x[0] == best, zip(fvalues, actions))
        actions = random.choice(ties)[1]
        return actions
    def chooseAction(self, gameState):
        """
        SELECT STEP in Monte Carlo Search Tree:

        if(otherAgent.isPacman in view && no mate around):
            chasing enemy: distance to otherAgent.isPacman.
            return nextAction = classical planning action to enemy?

        if(otherAgentState.scaredTimer > 5):
            Classical planing eating food
            retur nextAction = classical planning action to foods. // but avoid eating other capsules!

        actions = getRationalActions(): no STOP, no action to EMPTY ALLEY.
        for action in actions:
            values.append(UCTSimulation(action))
        bestAction = max(values)
        ties = filter(lambda x: x[0] == best, zip(fvalues, actions))
        nextAction = random.choice(ties)[1]
        """
        myPos = gameState.getAgentState(self.index).getPosition()
        nearestGhost, nearestGhostDist = self.getNearestGhost(gameState)
        mates = self.getTeam(gameState) - [self.index]
        nearestMateDist = min(self.getMazeDistance(myPos, mate) for mate in mates)

        if nearestGhost.isPacman and nearestMateDist > nearestGhostDist:
            return self.pureEnvBFS(gameState, nearestGhost)

        if nearestGhost.scaredTime > 5:
            nearestFood, nearestFoodDist = self.getNearestFood(gameState)
            return self.pureEnvBFS(gameState, nearestFood)

        return self.getRationalActions(gameState)