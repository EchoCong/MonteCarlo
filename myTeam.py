#from __future__ import division
from captureAgents import CaptureAgent
from captureAgents import AgentFactory
from game import Directions
import random, time, util
from util import nearestPoint
import distanceCalculator
import sys
sys.path.append('teams/<your team>/')


import random, time, util
import datetime
from random import choice
from math import log, sqrt

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


    def getRationalActions(self, gameState):
        """
        EXPAND Step in Monte Carlo Search Tree:

        actions = gameState.getLegalActions(self.index)
        aBooleanValue = takeToEmptyAlley(self, gameState, action, depth)
        actions.remove(Directions.STOP and action lead pacman into a empty alley)
        """

        action, percent_wins = MCTSNode(gameState, self.index).get_play()

        '''
        You should change this in your own agent.
        '''

        legal = gameState.getLegalActions(self.index)
        legal.remove('Stop')
        moves_states = [(p, gameState.generateSuccessor(self.index, p)) for p in legal]

        foodlist = self.getFood(gameState).asList()

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in legal:
                successor = gameState.generateSuccessor(self.index, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        # if percentage of win is 0, go and eat the nearest food
        if percent_wins == 0.0:
            minDistance = 9999
            for p, S in moves_states:
                myPos = S.getAgentState(self.index).getPosition()
                temp = min([self.getMazeDistance(myPos, food) for food in foodlist])
                if temp < minDistance:
                    minDistance = temp
                    action = p
        return action

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

class Attacker(EvaluationBasedAgent):
    """Monte Carlo, o agent offensive."""

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

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        # Variables used to verify if the agent os locked
        self.numEnemyFood = "+inf"
        self.inactiveTime = 0

    # Implemente este metodo para pre-processamento (15s max).
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()
        self.start = gameState.getAgentPosition(self.index)



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
        myNearestGhostDist, nearestGhost = self.getNearestGhost(gameState)
        capsuleDist, capsule = min([(self.getMazeDistance(myPos, cap), cap) for cap in self.getCapsules(gameState)])
        mates = self.getTeam(gameState)
        mates.remove(self.index)

        if nearestGhost is not None and nearestGhost.isPacman:
            mateNearestGhostDist = min(self.getMazeDistance(
                nearestGhost.getPosition(), gameState.getAgentState(mate).getPosition()) for mate in mates)
            if mateNearestGhostDist > myNearestGhostDist:
                # print "Help Mate!"
                print "BFS"
                return self.pureEnvBFS(gameState, nearestGhost.getPosition())

        if nearestGhost is None or nearestGhost.scaredTimer > 5:
            if gameState.getAgentState(self.index).numCarrying < 5:
                nearestFoodDist, nearestFood = self.getNearestFood(gameState)
                print "Compare Distance"
                return self.compareSuccDistToFood(gameState, nearestFood)

        if nearestGhost is not None and capsuleDist < myNearestGhostDist:
            print "Capsule"
            return self.compareSuccDistToFood(gameState, capsule)

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

        # # print 'eval time for defender agent %d: %.4f' % (self.index, time.time() - start)
        return random.choice(ties)[1]


class MCTSNode():
  """
  build the MCTS tree
  """
  def __init__(self, game_state, player_index, **kwargs):

    #define the time allowed for simulation
    seconds = kwargs.get('time', 1)
    self.calculate_time = datetime.timedelta(seconds=seconds)

    self.max_moves = kwargs.get('max_moves', 90)
    self.states = [game_state]
    self.index = player_index
    self.wins = util.Counter()
    self.plays = util.Counter()
    self.C = kwargs.get('C', 1)
    self.distancer = distanceCalculator.Distancer(game_state.data.layout)

    if game_state.isOnRedTeam(self.index):
        self.enemies = game_state.getBlueTeamIndices()
        self.foodlist = game_state.getBlueFood()
        self.capsule = game_state.getBlueCapsules()
    else:
        self.enemies = game_state.getRedTeamIndices()
        self.foodlist = game_state.getRedFood()
        self.capsule = game_state.getRedCapsules()

  def update(self, state):
    self.states.append(state)

  def getMazeDistance(self, pos1, pos2):
      """
      Returns the distance between two points; These are calculated using the provided
      distancer object.

      If distancer.getMazeDistances() has been called, then maze distances are available.
      Otherwise, this just returns Manhattan distance.
      """
      d = self.distancer.getDistance(pos1, pos2)
      return d

  def get_play(self):
    # Causes the AI to calculate the best move from the
    # current game state and return it.
    state = self.states[-1]
    legal = state.getLegalActions(self.index)
    legal.remove('Stop')

    # Bail out early if there is no real choice to be made.
    if not legal:
      return
    if len(legal) == 1:
      return legal[0], 0.0

    games = 0

    begin = datetime.datetime.utcnow()
    while datetime.datetime.utcnow() - begin < self.calculate_time:
      self.run_simulation()
      games += 1
    # print "SIMULATION NUMBER", games

    moves_states = [(p, state.generateSuccessor(self.index, p)) for p in legal]

    # Display the number of calls of `run_simulation` and the
    # time elapsed.

    percent_wins, move = max((float(self.wins.get((self.index, S.getAgentState(self.index).getPosition()), 0)) / float(self.plays.get((self.index, S.getAgentState(self.index).getPosition()), 1)), p)for p, S in moves_states)

    return move, percent_wins



  def run_simulation(self):
    # Plays out a "random" game from the current position,
    # then updates the statistics tables with the result.
    state_copy = self.states[:]
    state = state_copy[-1]
    visited_states = set()
    visited_states.add(state)
    states_path = [state]

    expand = True
    for i in xrange(1,self.max_moves+1):
      state = state_copy[-1]
      # make i evaluates lazily
      legal_move = state.getLegalActions(self.index)
      legal_move.remove('Stop')

      # Bail out early if there is no real choice to be made.
      if not legal_move:
        return

      moves_states = [(p, state.generateSuccessor(self.index, p)) for p in legal_move]

      # check if all the results in the legal_move are in the plays dictionary
      # if they are, use UBT1 to make choice
      if all(self.plays.get((self.index, S.getAgentState(self.index).getPosition())) for p, S in moves_states):

        # the number of times state has been visited.
        if self.plays[(self.index, state.getAgentState(self.index).getPosition())] == 0:
            log_total = 1.0

        else:
            log_total = float(2 * log(self.plays[(self.index, state.getAgentState(self.index).getPosition())]))

        value, move, nstate = max(
          ((float(self.wins[(self.index, S.getAgentState(self.index).getPosition())]) / float(self.plays[(self.index, S.getAgentState(self.index).getPosition())])) +
           2 * self.C * sqrt(log_total / float(self.plays[(self.index, S.getAgentState(self.index).getPosition())])), p, S)
          for p, S in moves_states
        )
      else:
        # if not, make a random choice
        move, nstate = choice(moves_states)


      state_copy.append(nstate)
      states_path.append(nstate)

      if expand and (self.index, nstate.getAgentState(self.index).getPosition()) not in self.plays:
        # expand the tree
        expand = False
        self.plays[(self.index,nstate.getAgentState(self.index).getPosition())] = 0.0
        self.wins[(self.index, nstate.getAgentState(self.index).getPosition())] = 0.0

      visited_states.add(nstate)

      # Computes distance to enemies we can see
      enemies = [state.getAgentState(i) for i in self.enemies]
      ghost = [a for a in enemies if a.getPosition() != None and not a.isPacman]
      invaders = [a for a in enemies if a.isPacman]

      nenemies = [nstate.getAgentState(i) for i in self.enemies]
      nghost = [a for a in nenemies if a.getPosition() != None and not a.isPacman]


      if len(invaders) != 0:
          ## if see a invader and ate it, win +1
          ate = False
          for a in invaders:
              if nstate.getAgentState(self.index).getPosition() == a.getPosition():
                  ate = True
                  break
          if ate:
              # record number of wins
              for s in states_path:
                  if (self.index, s.getAgentState(self.index).getPosition()) not in self.plays:
                      continue
                  self.wins[(self.index, s.getAgentState(self.index).getPosition())] += 1
              # print self.index, "EAT GHOST +1"
              break

      x, y = nstate.getAgentState(self.index).getPosition()

      # ghost
      if len(ghost) > 0:
          dist_to_ghost = min(self.getMazeDistance((x, y), a.getPosition()) for a in nghost)

          if dist_to_ghost > 3:
              # record number of wins
              for s in states_path:
                  if (self.index, s.getAgentState(self.index).getPosition()) not in self.plays:
                      continue
                  self.wins[(self.index, s.getAgentState(self.index).getPosition())] += 1
              # print self.index, "AVOID NEARBY GHOST +1"
              break

      # Capsule
      if len(self.capsule) != 0:
          dist_to_capsule, a = min([(self.getMazeDistance((x, y), a), a) for a in self.capsule])

          if nstate.getAgentState(self.index).getPosition() == a:
              # record number of wins
              for s in states_path:
                  if (self.index, s.getAgentState(self.index).getPosition()) not in self.plays:
                      continue
                  self.wins[(self.index, s.getAgentState(self.index).getPosition())] += 0.0
              # print self.index, "EAT CAPSULE +1"
              break

      # Score
      if nstate.getScore() > 3:
          # record number of wins
          for s in states_path:
              if (self.index, s.getAgentState(self.index).getPosition()) not in self.plays:
                  continue
              self.wins[(self.index, s.getAgentState(self.index).getPosition())] += 0.8
          # print self.index, "RETURN FOOD +1"
          break

      # Food
      if ((nstate.getAgentState(self.index).numCarrying - state.getAgentState(self.index).numCarrying > 0) or nstate.getAgentState(self.index).numReturned > 4) and len(ghost) == 0:
          # record number of wins
          for s in states_path:
              if (self.index, s.getAgentState(self.index).getPosition()) not in self.plays:
                  continue
              self.wins[(self.index, s.getAgentState(self.index).getPosition())] += 0.0
          # print self.index, "AVOID GHOST +1"
          break


    for s in states_path:
      # record number of plays
      if (self.index, s.getAgentState(self.index).getPosition()) not in self.plays:
        continue
      self.plays[(self.index, s.getAgentState(self.index).getPosition())] += 1

