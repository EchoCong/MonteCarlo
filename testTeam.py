from captureAgents import CaptureAgent
from captureAgents import AgentFactory
from game import Directions
import random, time, util
from util import nearestPoint

def createTeam(firstIndex, secondIndex, isRed,
               first='Attacker', second='Defender'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]
#############
# FACTORIES #
###############################################
# Instanciam os agentes no inicio da partida. #
# Devem estender a classe base AgentFactory.  #
###############################################

class MonteCarloFactory(AgentFactory):
    "Gera um time MonteCarloTeam"

    def __init__(self, isRed):
        AgentFactory.__init__(self, isRed)
        self.agentList = ['attacker', 'defender']

    def getAgent(self, index):
        if len(self.agentList) > 0:
            agent = self.agentList.pop(0)
            if agent == 'attacker':
                return Attacker(index)
        return Defender(index)

        #############


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

    def getRationalActions(self, gameState):
        """
        EXPAND Step in Monte Carlo Search Tree:

        actions = gameState.getLegalActions(self.index)
        aBooleanValue = takeToEmptyAlley(self, gameState, action, depth)
        actions.remove(Directions.STOP and action lead pacman into a empty alley)
        """

        currentEnemyFood = len(self.getFood(gameState).asList())
        if self.numEnemyFood != currentEnemyFood:
            self.numEnemyFood = currentEnemyFood
            self.inactiveTime = 0
        else:
            self.inactiveTime += 1
        # If the agent dies, inactiveTime is reseted.
        if gameState.getInitialAgentPosition(self.index) == gameState.getAgentState(self.index).getPosition():
            self.inactiveTime = 0

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
        toPlay = random.choice(ties)[1]

        # print 'eval time for offensive agent %d: %.4f' % (self.index, time.time() - start)
        return toPlay

    def getNearestGhost(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        ghostsAll = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghostsInRange = filter(lambda x: not x.isPacman and x.getPosition() is not None, ghostsAll)
        if len(ghostsInRange) > 0:
            return min([(self.getMazeDistance(myPos, ghost.getPosition()), ghost) for ghost in ghostsInRange])
        return None, None

    def getNearestFood(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        foods = self.getFood(gameState).asList()
        if len(foods) > 0:
            return min([(self.getMazeDistance(myPos, food), food) for food in foods])
        return None, None

    def getNearestCapsule(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0:
            return min([(self.getMazeDistance(myPos, cap), cap) for cap in capsules])
        return None, None

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

    def pureEnvBFS(self, gameState, goalPos):
        """
        Called when otherAgentState.scaredTimer > 5 (goal is nearest food)
         or otherAgent.isPacman in view && no mate around (goal is pacman):
        Use A Star Algorithm to get the best next action to eat the nearest dot.
        Using MazeDistance as heuristic value.
        """
        # ######################################
        # try to solve with BFS
        # ######################################
        queue, expended, path = util.Queue(), [], []
        queue.push((gameState, path))

        while not queue.isEmpty():
            currState, path = queue.pop()
            expended.append(currState)
            if currState.getAgentState(self.index).getPosition() == goalPos:
                return path[0]
            actions = currState.getLegalActions(self.index)
            actions.remove(Directions.STOP)
            for action in actions:
                succState = currState.generateSuccessor(self.index, action)
                if not succState in expended:
                    expended.append(succState)
                    queue.push((succState, path + [action]))
        return []


    def compareSuccDistToFood(self, gameState, nearestFood):
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        goodActions = []
        fvalues = []
        for action in actions:
            succState = gameState.generateSuccessor(self.index, action)
            succPos = succState.getAgentPosition(self.index)
            goodActions.append(action)
            fvalues.append(self.getMazeDistance(succPos, nearestFood))

        # Randomly chooses between ties.
        best = min(fvalues)
        ties = filter(lambda x: x[0] == best, zip(fvalues, goodActions))

        # # print 'eval time for defender agent %d: %.4f' % (self.index, time.time() - start)
        return random.choice(ties)[1]


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
        features['successorScore'] = self.getScore(successor)

        # Compute distance to capsule
        capsules = self.getCapsules(successor)
        if len(capsules) > 0:
            nearestCapsuleDist = min(self.getMazeDistance(myPos, capsule) for capsule in capsules)
            features["distanceToCapsule"] = nearestCapsuleDist

        # Compute distance to the nearest food
        nearestFoodDist, nearestFood = self.getNearestFood(successor)
        if nearestFood is not None:
            features['distanceToFood'] = nearestFoodDist

        # Compute distance to closest ghost
        nearestGhostDist, nearestGhost = self.getNearestGhost(successor)
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
        # #########################
        # Now our pacman will eat all capsules first, waste of capsules.
        # #########################
        # Weights normally used
        return {'distanceToCapsule': -3, 'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 220}

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        # Variables used to verify if the agent os locked
        self.numEnemyFood = "+inf"
        self.inactiveTime = 0

    # Implemente este metodo para pre-processamento (15s max).
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()

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
        myGhostDist, nearestGhost = self.getNearestGhost(gameState)
        capsuleDist, nearestCapsule = self.getNearestCapsule(gameState)
        mates = self.getTeam(gameState)
        mates.remove(self.index)

        if nearestGhost is not None and nearestGhost.isPacman:
            mateGhostDist = min(self.getMazeDistance(
                nearestGhost.getPosition(), gameState.getAgentState(mate).getPosition()) for mate in mates)
            if mateGhostDist > myGhostDist:
                # print "Help Mate!"
                print "BFS"
                return self.pureEnvBFS(gameState, nearestGhost.getPosition())

        if nearestGhost is None or nearestGhost.scaredTimer > 5:
            if gameState.getAgentState(self.index).numCarrying < 5:
                nearestFoodDist, nearestFood = self.getNearestFood(gameState)
                print "Compare Distance"
                return self.compareSuccDistToFood(gameState, nearestFood)

        if nearestGhost is not None and nearestCapsule is not None and capsuleDist < myGhostDist:
            print "Capsule"
            return self.compareSuccDistToFood(gameState, nearestCapsule)

        print "UCT"
        return self.getRationalActions(gameState)

class Defender(CaptureAgent):
    "Gera Monte, o agente defensivo."

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.target = None
        self.lastObservedFood = None
        # This variable will store our patrol points and
        # the agent probability to select a point as target.
        self.patrolDict = {}

    def distFoodToPatrol(self, gameState):
        """
        This method calculates the minimum distance from our patrol
        points to our pacdots. The inverse of this distance will
        be used as the probability to select the patrol point as
        target.
        """
        food = self.getFoodYouAreDefending(gameState).asList()
        total = 0

        # Get the minimum distance from the food to our
        # patrol points.
        for position in self.noWallSpots:
            closestFoodDist = "+inf"
            for foodPos in food:
                dist = self.getMazeDistance(position, foodPos)
                if dist < closestFoodDist:
                    closestFoodDist = dist
            # We can't divide by 0!
            if closestFoodDist == 0:
                closestFoodDist = 1
            self.patrolDict[position] = 1.0 / float(closestFoodDist)
            total += self.patrolDict[position]
        # Normalize the value used as probability.
        if total == 0:
            total = 1
        for x in self.patrolDict.keys():
            self.patrolDict[x] = float(self.patrolDict[x]) / float(total)

    def selectPatrolTarget(self):
        """
        Select some patrol point to use as target.
        """
        rand = random.random()
        sum = 0.0
        for x in self.patrolDict.keys():
            sum += self.patrolDict[x]
            if rand < sum:
                return x

    # Implemente este metodo para pre-processamento (15s max).
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()

        # Compute central positions without walls from map layout.
        # The defender will walk among these positions to defend
        # its territory.
        if self.red:
            centralX = (gameState.data.layout.width - 2) / 2
        else:
            centralX = ((gameState.data.layout.width - 2) / 2) + 1
        self.noWallSpots = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(centralX, i):
                self.noWallSpots.append((centralX, i))
        # Remove some positions. The agent do not need to patrol
        # all positions in the central area.
        while len(self.noWallSpots) > (gameState.data.layout.height - 2) / 2:
            self.noWallSpots.pop(0)
            self.noWallSpots.pop(len(self.noWallSpots) - 1)
        # Update probabilities to each patrol point.
        self.distFoodToPatrol(gameState)

    # Implemente este metodo para controlar o agente (1s max).
    def chooseAction(self, gameState):
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()

        # If some of our food was eaten, we need to update
        # our patrol points probabilities.
        if self.lastObservedFood and len(self.lastObservedFood) != len(self.getFoodYouAreDefending(gameState).asList()):
            self.distFoodToPatrol(gameState)

        mypos = gameState.getAgentPosition(self.index)
        if mypos == self.target:
            self.target = None

        # If we can see an invader, we go after him.
        x = self.getOpponents(gameState)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
        if len(invaders) > 0:
            positions = [agent.getPosition() for agent in invaders]
            self.target = min(positions, key=lambda x: self.getMazeDistance(mypos, x))
        # If we can't see an invader, but our pacdots were eaten,
        # we will check the position where the pacdot disappeared.
        elif self.lastObservedFood != None:
            eaten = set(self.lastObservedFood) - set(self.getFoodYouAreDefending(gameState).asList())
            if len(eaten) > 0:
                self.target = eaten.pop()

        # Update the agent memory about our pacdots.
        self.lastObservedFood = self.getFoodYouAreDefending(gameState).asList()

        # No enemy in sight, and our pacdots are not disappearing.
        # If we have only a few pacdots, let's walk among them.
        if self.target == None and len(self.getFoodYouAreDefending(gameState).asList()) <= 4:
            food = self.getFoodYouAreDefending(gameState).asList() \
                   + self.getCapsulesYouAreDefending(gameState)
            self.target = random.choice(food)
        # If we have many pacdots, let's patrol the map central area.
        elif self.target == None:
            self.target = self.selectPatrolTarget()

        # Choose action. We will take the action that brings us
        # closer to the target. However, we will never stay put
        # and we will never invade the enemy side.
        actions = gameState.getLegalActions(self.index)
        goodActions = []
        fvalues = []
        for a in actions:
            new_state = gameState.generateSuccessor(self.index, a)
            if not new_state.getAgentState(self.index).isPacman and not a == Directions.STOP:
                newpos = new_state.getAgentPosition(self.index)
                goodActions.append(a)
                fvalues.append(self.getMazeDistance(newpos, self.target))

        # Randomly chooses between ties.
        best = min(fvalues)
        ties = filter(lambda x: x[0] == best, zip(fvalues, goodActions))

        # print 'eval time for defender agent %d: %.4f' % (self.index, time.time() - start)
        return random.choice(ties)[1]
