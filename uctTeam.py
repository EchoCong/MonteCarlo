# myTeam.py
# ---------
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

from __future__ import division
from captureAgents import CaptureAgent
import random, time, util
import datetime
from random import choice
from math import log, sqrt


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
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

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    self.start = gameState.getAgentPosition(self.index)



  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
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




class Approximate_Q:

  def __init__(self, game_state, goal, player_index, prev_agent_pos, **kwargs):
    # define the time allowed for simulation
    seconds = kwargs.get('time', 2)
    self.calculate_time = datetime.timedelta(seconds=seconds)

    self.max_moves = kwargs.get('max_moves', 50)
    self.states = [game_state]
    self.index = player_index
    self.goal = goal
    self.wins = util.Counter()
    self.plays = util.Counter()
    self.C = kwargs.get('C', 1.4)
    self.prev_agent_pos = prev_agent_pos
    self.disc_factor = 0.9
    self.learning_rate = 0.004
    self.q_value = util.Counter()
    self.weight = {'distanceToFood': 0, 'distanceToGhost': 0}
    self.food_reward = 1
    self.ghost_reward = -500


  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    # features after taking the action
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = 1/minDistance

    # Compute distance to the nearest ghost

    return features


  def updateWeights(self, gameState, action):
    diff = + self.disc_factor*self.updateQValue[(gameState, action)] - self.q_value[(gameState, action)]

    gameState.generateSuccessor(self.index, action)

    self.weight['distanceToFood'] = self.weight['distanceToFood'] + self.learning_rate*diff*self.getFeatures(gameState, action)['distanceToFood']
    self.weight['distanceToGhost'] = self.weight['distanceToGhost'] + self.learning_rate * diff * self.getFeatures(
      gameState, action)['distanceToGhost']

    return self.weight

  def getBestAction(self, gameState):
    bestActoin = None
    max_q_value = 0.0
    legal_actions = gameState.getLegalActions(self.index)

    for action in legal_actions:
        if self.q_value[(gameState, action)] > max_q_value:
            max_q_value = self.q_value[(gameState, action)]
            bestActoin = action

    return bestActoin


  def updateQValue(self, gameState, action):
        self.q_value[(gameState, action)] = self.updateWeights(gameState, action)*self.getFeatures(gameState, action)







