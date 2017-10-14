from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint
import distanceCalculator
import random, util
import datetime
from random import choice
from math import log, sqrt

import sys
sys.path.append('teams/<your team>/')


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


#########################################################
#  Evaluation Based CaptureAgent.                       #
#  Provide functions used by both attacker and defender #
#########################################################

class EvaluationBasedAgent(CaptureAgent):
    # Implemente este metodo para pre-processamento (15s max).

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()
        self.start = gameState.getAgentPosition(self.index)
    #     self.entrance = self.analyseEntrance(gameState)
    #
    # def analyseEntrance(self, gameState):
    #     height = gameState.data.layout.height
    #     width = gameState.data.layout.width
    #     print "HEIGHT: ", height, " WIDTH: ", width
    #     print gameState.data.layout.walls
    #     walls = gameState.data.layout.walls.asList()
    #     boundPos = [(x,y) for (x,y) in walls if x == width/2 +1]
    #
    #
    #     print boundPos

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

    def getNearestGhost(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        ghostsAll = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghostsInRange = [ghost for ghost in ghostsAll if
                         ghost.getPosition() is not None and
                         util.manhattanDistance(myPos, ghost.getPosition()) <= 5]

        if len(ghostsInRange) > 0:
            return min([(self.getMazeDistance(myPos, ghost.getPosition()), ghost) for ghost in ghostsInRange])

        return None, None

    def getNearestFood(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        foods = self.getFood(gameState).asList()

        nearFoodsNum = max(len([food for food in foods if self.getMazeDistance(food, myPos) <= 5]), 0)

        if len(foods) > 0:
            return min([(self.getMazeDistance(myPos, food), food, nearFoodsNum) for food in foods])
        return None, None, None

    def getNearestCapsule(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0:
            return min([(self.getMazeDistance(myPos, cap), cap) for cap in capsules])
        return None, None

    def getBFSAction(self, gameState, goalPos):
        """
        BFS Algorithm to help ally chase enemy.
        """
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

    def getGreedyAction(self, gameState, nearestFood):
        """
        Greedy Algorithm to eat nearest goal.
        """
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
        ties = [combine for combine in zip(fvalues, goodActions) if combine[0] == best]

        return random.choice(ties)[1]

    def getUCTAction(self, gameState):
        """
        UCT algorithm to choose which action is more rational.
        """

        action, percent_wins = MCTSNode(gameState, self.index).get_play()

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


class Attacker(EvaluationBasedAgent):
    """
    Monte Carlo, offensive agent.
    """

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        # Variables used to verify if the agent is locked
        self.numEnemyFood = "+inf"
        self.inactiveTime = 0

    def chooseAction(self, gameState):
        """
        Choose next action according to strategy.
        """
        # begin = datetime.datetime.utcnow()

        '''Strategy 1: Give up last two foods.'''
        myGhostDist, nearestGhost = self.getNearestGhost(gameState)
        capsuleDist, nearestCapsule = self.getNearestCapsule(gameState)
        nearestFoodDist, nearestFood, nearFoodsNum = self.getNearestFood(gameState)

        if len(self.getFood(gameState).asList()) <= 2:
            uctAction = self.getUCTAction(gameState)
            # print "FINAL TWO FOOD TIME ", datetime.datetime.utcnow() - begin
            return uctAction

        '''Strategy 2: BFS eat enemy when ally not around.'''
        if nearestGhost is not None and nearestGhost.isPacman:
            mates = self.getTeam(gameState)
            mates.remove(self.index)
            mateGhostDist = min(self.getMazeDistance(
                nearestGhost.getPosition(), gameState.getAgentState(mate).getPosition()) for mate in mates)
            if mateGhostDist > myGhostDist:
                bfsAction = self.getBFSAction(gameState, nearestGhost.getPosition())
                # print "HELP MATE TIME ", datetime.datetime.utcnow() - begin
                return bfsAction

        '''Strategy 3: Greedy eat foods when safe.'''
        if nearestGhost is None or (nearestGhost is not None and myGhostDist >= 6) or nearestGhost.scaredTimer > 5:
            if (nearestFood is not None and nearFoodsNum >= 5 and gameState.getAgentState(self.index).numCarrying <= 15) \
                    or (nearestFood is not None and gameState.getAgentState(self.index).numCarrying < 5):
                eatAction = self.getGreedyAction(gameState, nearestFood)
                # print "EAT FOOD TIME ", datetime.datetime.utcnow() - begin
                return eatAction

        '''Strategy 4: Greedy eat capsule when half nearestGhostDistance closer than enemy.'''
        if nearestGhost is not None and not nearestGhost.isPacman and \
                        nearestCapsule is not None and capsuleDist <= myGhostDist/2:
            eatAction = self.getGreedyAction(gameState, nearestCapsule)
            # print "EAT CAPSULE TIME ", datetime.datetime.utcnow() - begin
            return eatAction

        '''Strategy 5: Other situation: UCT Algorithm trade off score, escape, capsule.'''
        uctAction = self.getUCTAction(gameState)
        # print "UCT ACTION TIME", datetime.datetime.utcnow() - begin
        return uctAction


class Defender(EvaluationBasedAgent):
    """
    Monte Carlo, agent defensive."
    """

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

        return random.choice(ties)[1]


class MCTSNode:
    """
    build the MCTS tree
    """

    def __init__(self, game_state, player_index, **kwargs):

        # define the time allowed for simulation
        seconds = kwargs.get('time', 0.5)
        self.calculate_time = datetime.timedelta(seconds=seconds)

        self.max_moves = kwargs.get('max_moves', 20)
        self.states = [game_state]
        self.index = player_index
        self.wins = util.Counter()
        self.plays = util.Counter()
        self.C = kwargs.get('C', 1)
        self.distancer = distanceCalculator.Distancer(game_state.data.layout)
        self.distancer.getMazeDistances()
        self.gameState = game_state

        if game_state.isOnRedTeam(self.index):
            self.enemies = game_state.getBlueTeamIndices()
            self.foodlist = game_state.getBlueFood()
            self.capsule = game_state.getBlueCapsules()
        else:
            self.enemies = game_state.getRedTeamIndices()
            self.foodlist = game_state.getRedFood()
            self.capsule = game_state.getRedCapsules()

    def getMazeDistance(self, pos1, pos2):
        """
        Returns the distance between two points; These are calculated using the provided
        distancer object.

        If distancer.getMazeDistances() has been called, then maze distances are available.
        Otherwise, this just returns Manhattan distance.
        """
        d = self.distancer.getDistance(pos1, pos2)
        return d

    def update(self, state):
        self.states.append(state)

    def takeToEmptyAlley(self, gameState, action, depth):
        """
        Verify if an action takes the agent to an alley with
        no pacdots.
        """
        old_score = gameState.getScore()
        new_state = gameState.generateSuccessor(self.index, action)
        new_score = new_state.getScore()
        if old_score < new_score or depth == 0:
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

    def get_play(self):
        # Causes the AI to calculate the best move from the
        # current game state and return it.
        # print "INDEX: ", self.index
        state = self.states[-1]
        legal = state.getLegalActions(self.index)
        legal.remove('Stop')

        for action in legal:
            if self.takeToEmptyAlley(self.gameState, action, 6):
                legal.remove(action)
                # print "EMPTY ALLEY REMOVED"

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
        # print "SIMULATION NUMBER: ", games

        moves_states = [(p, state.generateSuccessor(self.index, p)) for p in legal]

        # Display the number of calls of `run_simulation` and the
        # time elapsed.
        # print games, datetime.datetime.utcnow() - begin

        # Pick the move with the highest percentage of wins.
        # for p, S in moves_states:
        # print "CURRENT WINS: ", self.wins.get((self.index, S.getAgentState(self.index).getPosition()))
        # print "CURRENT PLAYS: ", self.plays.get((self.index, S.getAgentState(self.index).getPosition()))
        # print "CURRENT PERCENTAGE: ", float(self.wins.get((self.index, S.getAgentState(self.index).getPosition()), 0)) / float(self.plays.get((self.index, S.getAgentState(self.index).getPosition()), 1))
        # print "CURRENET MOVE: ", p
        # print " "

        percent_wins, move = max((float(
            self.wins.get((self.index, S.getAgentState(self.index).getPosition()), 0)) / float(
            self.plays.get((self.index, S.getAgentState(self.index).getPosition()), 1)), p) for p, S in moves_states)

        # print "AGENT INDEX: ", self.index
        # print "PERCENTAGE: ", percent_wins
        # print "MOVE: ", move
        # print " "
        # print " "
        # print " "

        return move, percent_wins

    def run_simulation(self):
        # Plays out a "random" game from the current position,
        # then updates the statistics tables with the result.
        state_copy = self.states[:]
        state = state_copy[-1]
        visited_states = set()
        visited_states.add(state)
        states_path = [state]

        enemies = [state.getAgentState(i) for i in self.enemies if state.getAgentState(i).scaredTimer < 6]
        ghost = [a for a in enemies if a.getPosition() != None and not a.isPacman]
        invaders = [a for a in enemies if a.isPacman]

        c, d = state.getAgentState(self.index).getPosition()

        expand = True
        for i in xrange(1, self.max_moves + 1):
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
                if self.plays[(self.index, state.getAgentState(self.index).getPosition())] == 0.0:
                    log_total = 0.5

                else:
                    log_total = float(
                        2.0 * log(self.plays[(self.index, state.getAgentState(self.index).getPosition())]))

                value, move, nstate = max(
                    ((float(self.wins[(self.index, S.getAgentState(self.index).getPosition())]) / float(
                        self.plays[(self.index, S.getAgentState(self.index).getPosition())])) +
                     2 * self.C * sqrt(
                         log_total / float(self.plays[(self.index, S.getAgentState(self.index).getPosition())])), p, S)
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
                self.plays[(self.index, nstate.getAgentState(self.index).getPosition())] = 0.0
                self.wins[(self.index, nstate.getAgentState(self.index).getPosition())] = 0.0

            visited_states.add(nstate)

            # Computes distance to enemies we can see

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
                        self.wins[(self.index, s.getAgentState(self.index).getPosition())] += 1.0
                        # print self.index, "EAT GHOST +1"
                    break

            x, y = nstate.getAgentState(self.index).getPosition()

            if len(ghost) > 0:

                cur_dist_to_ghost, a = min([(self.getMazeDistance((c, d), a.getPosition()), a) for a in ghost])

                if util.manhattanDistance((c, d), a.getPosition()) < 6:

                    next_dist_to_ghost = min((self.getMazeDistance((x, y), a.getPosition()) for a in ghost))

                    # print "CURRENT", cur_dist_to_ghost
                    # print "NEXT", next_dist_to_ghost

                    if next_dist_to_ghost - cur_dist_to_ghost > 3:
                        # record number of wins
                        for s in states_path:
                            if (self.index, s.getAgentState(self.index).getPosition()) not in self.plays:
                                continue
                            self.wins[(self.index, s.getAgentState(self.index).getPosition())] += 1.0
                            # print self.index, "AVOID NEARBY GHOST +1"
                        break

                    if next_dist_to_ghost < cur_dist_to_ghost:
                        break

            if len(self.capsule) != 0:
                dist_to_capsule, a = min([(self.getMazeDistance((x, y), a), a) for a in self.capsule])

                if nstate.getAgentState(self.index).getPosition() == a:
                    # record number of wins
                    for s in states_path:
                        if (self.index, s.getAgentState(self.index).getPosition()) not in self.plays:
                            continue
                        self.wins[(self.index, s.getAgentState(self.index).getPosition())] += 0.002
                        # print self.index, "EAT CAPSULE +1"
                    break

            if abs(nstate.getScore() - state.getScore()) > 3:
                # record number of wins
                for s in states_path:
                    if (self.index, s.getAgentState(self.index).getPosition()) not in self.plays:
                        continue
                    self.wins[(self.index, s.getAgentState(self.index).getPosition())] += 0.6
                    # print self.index, "RETURN FOOD +1"
                break

            """""

            if nstate.getAgentState(self.index).numCarrying - state.getAgentState(self.index).numCarrying > 0 and len(nghost) == 0:
                # record number of wins
                for s in states_path:
                    if (self.index, s.getAgentState(self.index).getPosition()) not in self.plays:
                        continue
                    self.wins[(self.index, s.getAgentState(self.index).getPosition())] += 0.0
                    # print self.index, "AVOID GHOST AND EAT DOTS +1"
                break
            """""

        for s in states_path:
            # record number of plays
            if (self.index, s.getAgentState(self.index).getPosition()) not in self.plays:
                continue
            self.plays[(self.index, s.getAgentState(self.index).getPosition())] += 1.0
