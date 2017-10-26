from captureAgents import CaptureAgent
import random, time, util, sys
from game import Directions
from game import Actions
from util import nearestPoint
import copy

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):

    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class AStarAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.start_state = gameState.getAgentPosition(self.index)
        self.walls = gameState.getWalls().asList()
        self.total_food_num = len(self.getFood(gameState).asList())

        self.back_to_home = random.choice(self.getFoodYouAreDefending(gameState).asList())   

        self.initial_food_list = copy.copy(self.getFoodYouAreDefending(gameState).asList())
        self.my_previous_food_list = copy.copy(self.getFoodYouAreDefending(gameState).asList())
        self.food_lost_position = []

        self.dead_position_list = []
  
        self.my_score = 0

    def getSuccessors(self, position, isChased):
        successors = []
        cur_state = self.getCurrentObservation()

        pos_block = copy.copy(self.walls)

        if isChased:
            pos_block.extend(self.dead_position_list)
        
        if cur_state.getAgentState(self.index).isPacman:
            enemies = [cur_state.getAgentState(opponent) for opponent in self.getOpponents(cur_state)]
            defenders = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None and enemy.scaredTimer <= 0]
            if len(defenders) > 0:
                defenders_pos = [defender.getPosition() for defender in defenders]
                for i in defenders_pos:
                    x, y = i
                    # we make the packman believe the pos around the defender is wall
                    defender_walls = [(x, y), (x -1, y), (x + 1, y), (x, y + 1), (x, y - 1), 
                                      (x -1, y + 1), (x - 1, y - 1), (x + 1, y + 1), (x + 1, y - 1)]
                    pos_block.extend(defender_walls)

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = position
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if (nextx, nexty) not in pos_block:
                nextState = (nextx, nexty)
                successors.append((nextState, action))
        return successors

    def getGoals(self, gameState, is_defender):
        
        my_scared_time = gameState.data.agentStates[self.index].scaredTimer

        if gameState.isOnRedTeam(self.index):
            other_team_index = gameState.getBlueTeamIndices()
            enemy_scared_time = gameState.data.agentStates[other_team_index[0]].scaredTimer
        else:
            other_team_index = gameState.getRedTeamIndices()
            enemy_scared_time = gameState.data.agentStates[other_team_index[0]].scaredTimer
   
        if is_defender:
            return self.defenderGoals(gameState, my_scared_time)
        else:
            return self.attackerGoals(gameState, enemy_scared_time)

    def attackerGoals(self, gameState, enemy_scared_time = 0, food_num_to_eat = 5):
        cur_state = self.getCurrentObservation()
        cur_pos = cur_state.getAgentPosition(self.index)
        food_list = self.getFood(gameState).asList()
        
        capsule_list = []

        if gameState.isOnRedTeam(self.index):
            capsule_list = gameState.getBlueCapsules()
        else:
            capsule_list = gameState.getRedCapsules()

        capsule_dis_list = []
        food_dis_list = []
        # capslule distance
        for capsule in capsule_list:
            capsule_dis_list.append(self.getMazeDistance(cur_pos, capsule))

        # food distance
        for food in food_list:
            food_dis_list.append(self.getMazeDistance(cur_pos, food))

        food_dis_list = sorted(food_dis_list)
        capsule_dis_list = sorted(capsule_dis_list)
        
        sorted_capsule_list = []
        sorted_food_list = []

        for i in range(0, len(food_dis_list)):
            for food in food_list:
                if self.getMazeDistance(cur_pos, food) == food_dis_list[i]:
                    sorted_food_list.append(food)
                    food_list.remove(food)

        for i in range(0, len(capsule_dis_list)):
            for capsule in capsule_list:
                if self.getMazeDistance(cur_pos, capsule) == capsule_dis_list[i]:
                    sorted_capsule_list.append(capsule)
                    capsule_list.remove(capsule)

        goals_food_list = []

        # first goal is the capsule
        if len(sorted_capsule_list) >= 1 and enemy_scared_time <= 10:
           goals_food_list.append(sorted_capsule_list[0])

        if (food_num_to_eat - 1) >= len(sorted_food_list):
            goal_num = len(sorted_food_list)
        else:
            goal_num = food_num_to_eat - 1
        for i in range(goal_num):
            goals_food_list.append(sorted_food_list[i])

        return goals_food_list
        
    def defenderGoals(self, gameState, my_scared_time = 0):
        my_cur_food_list = copy.copy(self.getFoodYouAreDefending(gameState).asList())
        cur_state = self.getCurrentObservation()
        cur_pos = cur_state.getAgentPosition(self.index)

        enemies = [cur_state.getAgentState(i) for i in self.getOpponents(cur_state)]
        invaders = [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]
        attackers = [enemy for enemy in enemies if enemy.isPacman]

        if len(invaders) > 0:
            dists = [self.getMazeDistance(cur_pos, invader.getPosition()) for invader in invaders]
            for i in range(0, len(invaders)):
                if self.getMazeDistance(cur_pos, invaders[i].getPosition()) == min(dists):
                    return [invaders[i].getPosition()]
         
        if len(self.my_previous_food_list) < len(my_cur_food_list):
            self.my_previous_food_list = my_cur_food_list

        # if i find food lost i will go to check the foodlosted position
        if len(self.my_previous_food_list) > len(my_cur_food_list) :
            previous_set = set(self.my_previous_food_list)
            current_food_set = set(my_cur_food_list)
            self.food_lost_position = list((previous_set.difference(current_food_set)))
            self.my_previous_food_list = my_cur_food_list
            # Understand the Range of the map.

     
        thisWalls = copy.copy(self.walls)
        xline = []
        yline = []
        for i in thisWalls:
            xline.append(i[0])
        largerX = max(xline)

        for i in thisWalls:
            yline.append(i[1])
        largery = max(yline)

        midwax = (largerX - 1) / 2
        gride = []
        newclearG = []

        # If it is Blue Team
        if self.index % 2 != 0:
            for i in range(midwax, largerX):
                for j in range(1, largery):
                    gride.append((i, j))
            gride = filter(lambda x: x not in thisWalls, gride)
            for i in gride:
                if (midwax + 2) == i[0] or (midwax + 3) == i[0] or (midwax + 4) == i[0] or (midwax + 5) == i[0]:
                    newclearG.append(i)

        # If it is Rea Team
        if not self.index % 2 != 0:
            for i in range(0, midwax):
                for j in range(1, largery):
                    gride.append((i, j))
            gride = filter(lambda x: x not in thisWalls, gride)
            for i in gride:
                if (midwax - 2) == i[0] or (midwax - 3) == i[0] or (midwax - 4) == i[0] or (midwax - 5) == i[0]:
                    newclearG.append(i)
        # Return Goal List
        if len(attackers)==0:
            self.food_lost_position = [random.choice(newclearG)]

        if len(my_cur_food_list) < len(self.initial_food_list):
            return self.food_lost_position

        goto =  random.choice(newclearG)
        if self.getMazeDistance(goto,cur_pos) < 3:
            goto = random.choice(newclearG)
        return [goto]
    
    # aStart search
    def aStarSearch(self, goals):
        cur_state = self.getCurrentObservation()
        cur_pos = cur_state.getAgentPosition(self.index)
        
        for goal in goals:
            prio_queue = util.PriorityQueue()
            prio_queue.push((cur_pos, []), self.getMazeDistance(cur_pos, goal))
            visited_list = []

            while (prio_queue.isEmpty() == False):
                cur_pos1, path = prio_queue.pop()
                if cur_pos1 in visited_list:
                    continue
                visited_list.append(cur_pos1)

                if cur_pos1 == goal:
                    if len(path) == 0:
                        return 'Stop'
                    else:
                       return path[0]
                cur_succ = self.getSuccessors(cur_pos1, cur_state)

                for s in cur_succ:
                    cost = len(path + [s[1]]) + self.getMazeDistance(s[0], goal)
                    prio_queue.push((s[0], path + [s[1]]), cost)
        return 'Stop'


class OffensiveAgent(AStarAgent):

    chase_by_defender = False
    is_changing_pos = False
    changing_pos = ()

    def chooseAction(self, gameState):

        food_ate = self.total_food_num - len(self.getFood(gameState).asList())
        self_cur_state = self.getCurrentObservation().getAgentState(self.index)
        cur_state = self.getCurrentObservation()

        #left 2 food and go home
        if len(self.getFood(gameState).asList()) <= 2:
            return self.aStarSearch([self.back_to_home])

        if not self_cur_state.isPacman:
            self.my_score = food_ate

        #move to another position near the middle
        if self.chase_by_defender == True and not self_cur_state.isPacman:

            if self.is_changing_pos == False:
                self.is_changing_pos = True
                biggestY = -1
                for x, y in self.walls:
                    if y > biggestY:
                        biggestY = y
                cur_x, cur_y = self_cur_state.getPosition()
                i = -5
                allChangingPos = []
                while cur_y + i < biggestY and i < 6:
                    if cur_y + i > 0 and (cur_x,cur_y + i) not in self.walls and i not in [-2,-1,0,1,2]:
                        allChangingPos.append((cur_x, cur_y+i))
                    i += 1
                if len(allChangingPos) > 0:
                    self.changing_pos = random.choice(allChangingPos)
                    return self.aStarSearch([self.changing_pos])

            else:
                if self_cur_state.getPosition() != self.changing_pos:
                    return self.aStarSearch([self.changing_pos])
                else:
                    self.is_changing_pos = False
                    self.chase_by_defender = False
                    self.changing_pos = ()


        #avoid defenders
        if self_cur_state.isPacman:
            # get defenders position
            enemies = [cur_state.getAgentState(i) for i in self.getOpponents(cur_state)]
            defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]

            if len(defenders) > 0:
                defendersPos = [i.getPosition() for i in defenders]

                for pos in defendersPos:
                    distance = self.getMazeDistance(pos,self_cur_state.getPosition()) - 2
                    if distance <= 1:
                        self.chase_by_defender = True
                        return self.aStarSearch([self.back_to_home])

        # get my and enemy scaredtime
        my_scared_time = gameState.data.agentStates[self.index].scaredTimer
        if gameState.isOnRedTeam(self.index):
            other_team_index = gameState.getBlueTeamIndices()
            enemy_scared_time = gameState.data.agentStates[other_team_index[0]].scaredTimer
        else:
            other_team_index = gameState.getRedTeamIndices()
            enemy_scared_time = gameState.data.agentStates[other_team_index[0]].scaredTimer

        # teammate position
        team = self.getTeam(gameState)
        teammate_pos = [gameState.getAgentPosition(teammate) for teammate in team if teammate != self.index]

        # if the time gonna to time up, the attacker go back to teammate pos
        if self_cur_state.isPacman and util.manhattanDistance(cur_state.getAgentPosition(self.index), teammate_pos[0]) >= (int(gameState.data.timeleft / 4) + 5):
            return self.aStarSearch([teammate_pos[0]])

        if food_ate >= 5 and self_cur_state.isPacman and len(defenders) > 0 and enemy_scared_time <= 15:
            return self.aStarSearch([self.back_to_home])

        if food_ate >= 5 and self_cur_state.isPacman and enemy_scared_time <= 15:
            return self.aStarSearch([self.back_to_home])

        if not self_cur_state.isPacman:
            self.total_food_num = len(self.getFood(gameState).asList())
        
        # if we alrady win 7 socre two become defender.
        if self.getScore(gameState) >= 7:
            return self.aStarSearch(self.getGoals(gameState, True))

        return self.aStarSearch(self.getGoals(gameState, False))

class DefensiveAgent(AStarAgent):
    def chooseAction(self, gameState):
        return self.aStarSearch(self.getGoals(gameState, True))

