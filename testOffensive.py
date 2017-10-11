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
        nearestGhostDist, nearestGhost = self.getNearestGhost(gameState)
        mates = self.getTeam(gameState)
        mates.remove(self.index)

        if nearestGhost is not None and nearestGhost.isPacman:
            nearestMateDist = min(self.getMazeDistance(
                nearestGhost.getPosition(),gameState.getAgentState(mate).getPosition()) for mate in mates)
            if nearestMateDist > nearestGhostDist:
                return self.pureEnvBFS(gameState, nearestGhost.getPosition())

        if nearestGhost is None or nearestGhost.scaredTimer > 5:
            if gameState.getAgentState(self.index).numCarrying < 5:
                nearestFoodDist, nearestFood = self.getNearestFood(gameState)
                return self.pureEnvBFS(gameState, nearestFood)

        return self.getRationalActions(gameState)


class MCTSNode():
  """
  build the MCTS tree
  """
  def __init__(self, game_state, player_index, **kwargs):

    #define the time allowed for simulation
    seconds = kwargs.get('time', 1)
    self.calculate_time = datetime.timedelta(seconds=seconds)

    self.max_moves = kwargs.get('max_moves', 70)
    self.states = [game_state]
    self.index = player_index
    self.wins = util.Counter()
    self.plays = util.Counter()
    self.C = kwargs.get('C', 1)

    if game_state.isOnRedTeam(self.index):
        self.enemies = game_state.getBlueTeamIndices()
        self.foodlist = game_state.getBlueFood()
    else:
        self.enemies = game_state.getRedTeamIndices()
        self.foodlist = game_state.getRedFood()

  def update(self, state):
    self.states.append(state)

  def get_play(self):
    # Causes the AI to calculate the best move from the
    # current game state and return it.
    print "INDEX: ", self.index
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
    print "SIMULATION NUMBER: ", games

    print self.plays
    print self.wins

    moves_states = [(p, state.generateSuccessor(self.index, p)) for p in legal]

    # Display the number of calls of `run_simulation` and the
    # time elapsed.
    print games, datetime.datetime.utcnow() - begin

    # Pick the move with the highest percentage of wins.
    for p, S in moves_states:
        print "CURRENT WINS: ", self.wins.get((self.index, S.getAgentState(self.index).getPosition()))
        print "CURRENT PLAYS: ", self.plays.get((self.index, S.getAgentState(self.index).getPosition()))
        print "CURRENT PERCENTAGE: ", self.wins.get((self.index, S.getAgentState(self.index).getPosition()), 0) / self.plays.get((self.index, S.getAgentState(self.index).getPosition()), 1)
        print "CURRENET MOVE: ", p
        print " "

    percent_wins, move = max((self.wins.get((self.index, S.getAgentState(self.index).getPosition()), 0) / self.plays.get((self.index, S.getAgentState(self.index).getPosition()), 1), p)for p, S in moves_states)

    print "AGENT INDEX: ", self.index
    print "PERCENTAGE: ", percent_wins
    print "MOVE: ", move
    print " "
    print " "
    print " "

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
            log_total = 1

        else:
            log_total = 2 * log(self.plays[(self.index, state.getAgentState(self.index).getPosition())])

        value, move, nstate = max(
          ((self.wins[(self.index, S.getAgentState(self.index).getPosition())] / self.plays[(self.index, S.getAgentState(self.index).getPosition())]) +
           2 * self.C * sqrt(log_total / self.plays[(self.index, S.getAgentState(self.index).getPosition())]), p, S)
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
        self.plays[(self.index,nstate.getAgentState(self.index).getPosition())] = 0
        self.wins[(self.index, nstate.getAgentState(self.index).getPosition())] = 0

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
                  print self.index, "EAT GHOST +1"
              break

      x, y = nstate.getAgentState(self.index).getPosition()
      if len(ghost) > 0:
          dist_to_ghost = min(util.manhattanDistance((x, y), a.getPosition()) for a in nghost)

          if dist_to_ghost > 5:
              # record number of wins
              for s in states_path:
                  if (self.index, s.getAgentState(self.index).getPosition()) not in self.plays:
                      continue
                  self.wins[(self.index, s.getAgentState(self.index).getPosition())] += 1
                  print self.index, "AVOID NEARBY GHOST +1"
              break


      if nstate.getAgentState(self.index).numReturned > 4:
          # record number of wins
          for s in states_path:
              if (self.index, s.getAgentState(self.index).getPosition()) not in self.plays:
                  continue
              self.wins[(self.index, s.getAgentState(self.index).getPosition())] += 1
              print self.index, "RETURN FOOD +1"
          break


      if ((nstate.getAgentState(self.index).numCarrying - state.getAgentState(self.index).numCarrying > 0) or nstate.getAgentState(self.index).numReturned > 4) and len(ghost) == 0:
          # record number of wins
          for s in states_path:
              if (self.index, s.getAgentState(self.index).getPosition()) not in self.plays:
                  continue
              self.wins[(self.index, s.getAgentState(self.index).getPosition())] += 1
              print self.index, "AVOID GHOST +1"
          break


    for s in states_path:
      # record number of plays
      if (self.index, s.getAgentState(self.index).getPosition()) not in self.plays:
        continue
      self.plays[(self.index, s.getAgentState(self.index).getPosition())] += 1