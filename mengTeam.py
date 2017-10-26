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

        action, percent_wins = MCTSNode(gameState, self.index).getUctAction()

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
                        nearestCapsule is not None and capsuleDist <= myGhostDist / 2:
            eatAction = self.getGreedyAction(gameState, nearestCapsule)
            # print "EAT CAPSULE TIME ", datetime.datetime.utcnow() - begin
            return eatAction

        """
        Strategy 5: other situations use UCT algorithm to trade off:
            1) Go home gain score
            2) Run away from ghost
            3) Eat capsule
        """
        uctAction = self.getUCTAction(gameState)
        # print "UCT ACTION TIME", datetime.datetime.utcnow() - begin
        return uctAction


class Defender(EvaluationBasedAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.target = None
        self.FoodLastRound = None
        self.patrolDict = {}

    def distFoodToPatrol(self, gameState):
        """
        This method calculates the minimum distance from our patrol
        points to our pacdots. The inverse of this distance will
        be used as the probability to select the patrol point as
        target.
        """
        foodListDefending = self.getFoodYouAreDefending(gameState).asList()
        # total = 0

        # Get the minimum distance from the food to our
        # patrol points.
        for position in self.patrolPosition:
            closestFoodDist = 1000
            for foodPos in foodListDefending:
                dist = self.getMazeDistance(position, foodPos)
                if dist < closestFoodDist:
                    closestFoodDist = dist
            # We can't divide by 0!
            # if closestFoodDist == 0:
            #     closestFoodDist = 1
            # self.patrolDict[position] = 1.0 / float(closestFoodDist)
            self.patrolDict[position] = closestFoodDist

            # total += self.patrolDict[position]
            # # Normalize the value used as probability.
            # if total == 0:
            #     total = 1
            # for x in self.patrolDict.keys():
            #     self.patrolDict[x] = float(self.patrolDict[x]) / float(total)

    """
        Remove some of patrol positions. 
        when the size (height) of maze is greater than 18,  leave 5 position to be un-patrolled
        when the size of maze is less than 18, leave half of the postions patrolled.
        :param height: height of the maze
    """

    def CleanPatrolPostions(self, height):

        if height > 18:
            for i in range(0, 5):
                self.patrolPosition.pop(0)
                self.patrolPosition.pop(len(self.patrolPosition) - 1)
        else:
            while len(self.patrolPosition) > (height - 2) / 2:
                self.patrolPosition.pop(0)
                self.patrolPosition.pop(len(self.patrolPosition) - 1)

    """
        get the initial patrol positons. 

        :param gameState: gameState
    """

    def getPatrolPosition(self, gameState):
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        if self.red:
            centralX = (width - 2) / 2
        else:
            centralX = ((width - 2) / 2) + 1
        self.patrolPosition = []
        for i in range(1, height - 1):
            if not gameState.hasWall(centralX, i):
                self.patrolPosition.append((centralX, i))

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()

        self.getPatrolPosition(gameState)
        self.CleanPatrolPostions(gameState.data.layout.height)

        print 'patrol points', self.patrolPosition
        # Update probabilities to each patrol point.
        self.distFoodToPatrol(gameState)

    """
    Update the minimum distance between patrol points to closest food
    """

    def updateMiniFoodDistance(self, gameState):
        if self.FoodLastRound and len(self.FoodLastRound) != len(self.getFoodYouAreDefending(gameState).asList()):
            self.distFoodToPatrol(gameState)

    def getInvaders(self, gameState):
        enemies = [gameState.getAgentState(each) for each in self.getOpponents(gameState)]
        return filter(lambda x: x.isPacman and x.getPosition() != None, enemies)

    def getDefendingTarget(self, gameState):
        return self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)

    def greedySearch(self, gameState):

        actions = gameState.getLegalActions(self.index)
        goodActions = []
        fvalues = []
        for a in actions:
            new_state = gameState.generateSuccessor(self.index, a)
            if not new_state.getAgentState(self.index).isPacman and not a == Directions.STOP:
                newpos = new_state.getAgentPosition(self.index)
                goodActions.append(a)
                fvalues.append(self.getMazeDistance(newpos, self.target))

        best = min(fvalues)
        ties = filter(lambda x: x[0] == best, zip(fvalues, goodActions))

        return random.choice(ties)[1]

    def chooseAction(self, gameState):
        # begin = datetime.datetime.utcnow()
        self.updateMiniFoodDistance(gameState)
        # print datetime.datetime.utcnow() - begin
        mypos = gameState.getAgentPosition(self.index)
        invaders = self.getInvaders(gameState)

        if mypos == self.target:
            self.target = None

        """
        if there is invaders: Go for the  nearest invader postion directly
        """
        if len(invaders) > 0:
            InvaderPositions = [eachInvader.getPosition() for eachInvader in invaders]
            self.target = min(InvaderPositions, key=lambda x: self.getMazeDistance(mypos, x))
        elif self.FoodLastRound is not None:
            FoodEatenPosition = set(self.FoodLastRound) - set(self.getFoodYouAreDefending(gameState).asList())
            if len(FoodEatenPosition) > 0:
                # print eaten
                self.target = FoodEatenPosition.pop()

        # record the food list in this current. it will be compared to next round's food list to determine
        # opponent's position
        self.FoodLastRound = self.getFoodYouAreDefending(gameState).asList()

        """
        when there are only 5 food dots remaining. defender patrol around these food rather than  boundary line.
        """
        if self.target is None and len(self.getFoodYouAreDefending(gameState).asList()) <= 5:
            self.target = random.choice(self.getDefendingTarget(gameState))

        # random to choose a position around the boundary to patrol.
        elif self.target is None:
            self.target = random.choice(self.patrolDict.keys())
            # print self.target

        return self.greedySearch(gameState)

##############################################################################
#  MCT applied UCB1 policy. Used to generate and return UCT move             #
#  Developed based on Jeff Bradberry's board game:                           #
#  https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/ #
##############################################################################

class MCTSNode:

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

    def getUctAction(self):
        # get the best move from the
        # current game state and return it.
        state = self.states[-1]
        legal = state.getLegalActions(self.index)
        legal.remove('Stop')

        for action in legal:
            if self.takeToEmptyAlley(self.gameState, action, 6):
                legal.remove(action)

        # return the action early if there is no other choice
        if not legal:
            return
        if len(legal) == 1:
            return legal[0], 0.0

        games = 0

        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < self.calculate_time:
            self.uctSimulation()
            games += 1

        moves_states = [(p, state.generateSuccessor(self.index, p)) for p in legal]

        percent_wins, move = max((float(
            self.wins.get(S.getAgentState(self.index).getPosition(), 0)) / float(
            self.plays.get(S.getAgentState(self.index).getPosition(), 1)), p) for p, S in moves_states)


        print move, percent_wins
        return move, percent_wins

    def uctSimulation(self):
        # simulate the moves from the current game state
        # and updates self.plays and self.wins
        state_copy = self.states[:]
        state = state_copy[-1]
        states_path = [state]

        # get the ghost and invaders the agent can see at the current game state
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
            if all(self.plays.get(S.getAgentState(self.index).getPosition()) for p, S in moves_states):

                # the number of times state has been visited.
                if self.plays[state.getAgentState(self.index).getPosition()] == 0.0:
                    log_total = 0.5

                else:
                    log_total = float(
                        2.0 * log(self.plays[state.getAgentState(self.index).getPosition()]))

                value, move, nstate = max(
                    ((float(self.wins[S.getAgentState(self.index).getPosition()]) / float(
                        self.plays[S.getAgentState(self.index).getPosition()])) +
                     2 * self.C * sqrt(
                         log_total / float(self.plays[S.getAgentState(self.index).getPosition()])), p, S)
                    for p, S in moves_states
                )
            else:
                # if not, make a random choice
                move, nstate = choice(moves_states)

            state_copy.append(nstate)
            states_path.append(nstate)

            if expand and nstate.getAgentState(self.index).getPosition() not in self.plays:
                # expand the tree
                expand = False
                self.plays[nstate.getAgentState(self.index).getPosition()] = 0.0
                self.wins[nstate.getAgentState(self.index).getPosition()] = 0.0


            if len(invaders) != 0:
                # if see a invader and ate it, win +1
                ate = False
                for a in invaders:
                    if nstate.getAgentState(self.index).getPosition() == a.getPosition():
                        ate = True
                        break
                if ate:
                    # record number of wins
                    for s in states_path:
                        if s.getAgentState(self.index).getPosition() not in self.plays:
                            continue
                        self.wins[s.getAgentState(self.index).getPosition()] += 1.0
                        print self.index, "EAT GHOST +1"
                    break

            x, y = nstate.getAgentState(self.index).getPosition()

            if len(ghost) > 0:

                cur_dist_to_ghost, a = min([(self.getMazeDistance((c, d), a.getPosition()), a) for a in ghost])

                if util.manhattanDistance((c, d), a.getPosition()) < 6:

                    next_dist_to_ghost = min((self.getMazeDistance((x, y), a.getPosition()) for a in ghost))

                    if next_dist_to_ghost - cur_dist_to_ghost > 3 and abs(nstate.getScore() - state.getScore()) > 0:
                        # record number of wins
                        for s in states_path:
                            if s.getAgentState(self.index).getPosition() not in self.plays:
                                continue
                            self.wins[s.getAgentState(self.index).getPosition()] += 1.0
                        break

                    if next_dist_to_ghost - cur_dist_to_ghost > 3:
                        # record number of wins
                        for s in states_path:
                            if s.getAgentState(self.index).getPosition() not in self.plays:
                                continue
                            self.wins[s.getAgentState(self.index).getPosition()] += 0.7
                        break

                    if next_dist_to_ghost < cur_dist_to_ghost:
                        break

            if len(self.capsule) != 0:
                dist_to_capsule, a = min([(self.getMazeDistance((x, y), a), a) for a in self.capsule])

                if nstate.getAgentState(self.index).getPosition() == a:
                    # record number of wins
                    for s in states_path:
                        if s.getAgentState(self.index).getPosition() not in self.plays:
                            continue
                        self.wins[s.getAgentState(self.index).getPosition()] += 0.002
                    break

            if abs(nstate.getScore() - state.getScore()) > 3:
                # record number of wins
                for s in states_path:
                    if s.getAgentState(self.index).getPosition() not in self.plays:
                        continue
                    self.wins[s.getAgentState(self.index).getPosition()] += 0.4
                break

        for s in states_path:
            # record number of plays
            if s.getAgentState(self.index).getPosition() not in self.plays:
                continue
            self.plays[s.getAgentState(self.index).getPosition()] += 1.0
