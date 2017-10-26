Meng Qi, Linlin Cong, Zelong Cong

==================== OFFENDER STRATEGY ===============

Offender Strategy 1: give up last two foods.

Offender Strategy 2: pacman can BFS eat enemy capsule when:
    1) nearest enemy is pacman AND
    2) current agent is closer to enemy than allies

Offender Strategy 3: pacman can greedy eat foods when:
    1) there are no ghosts around OR
    2) observed ghost is 6 maze steps away from our pacman OR
    3) nearest ghost scared time more than 5
But carrying too many foods is dangerous, so limit food carrying:
    1) there are more than 5 dots within maze distance 5 and carrying less than 15 dots OR
    2) carry dots less than 5   

Offender Strategy 4: pacman can greedy eat capsule when:
    1) nearest enemy is not pacman AND
    2) pacman is half times closer to capsule than the enemy

Offender Strategy 5: other situations use UCT algorithm to trade off:
    1) Go home gain score
    2) Run away from ghost
    3) Eat capsule

==================== DEFENDER STRATEGY ===============

Defender Strategy:
By default:
    1. patrol along the boujndary of our territory
When a food we are defending is eaten:
    2. Go to that food's postion 
When there are only 5 foods left in our territory:
    3. Patrol around the dots if there less than 5 dots left
When there is enemies in sight:
    4. chase and eat the enemies in its sight

==================== UCT IMPLEMENTATION ===============

1. Initial two counters: self.palys and self.wins:

    self.plays
        keys: All the positons that the agent experienced during simulations.
        value: number of times that the agent has been to that position.

    self.wins:
        keys: All the positons that the agent experienced during simulations.
        value: the number of times the agent has found a goal on that positon(key).

2. For each action, UCT simulation process is restricted to 0.5s, each simulation explores up to 20 gameStates. simulation stops when the goal is found,corresponding rewards will be added in self.wins. All the positons stored in self.plays will be added 1 nomatter the agent found the goal or not. During the simulation, we apply UCB1 policy to balance whether to encourage more exploration or not.

3. When the time slot is over, we get all legal successor states of current state. For each successor state, we use its position to find its corresponding values in self.plays and self.wins. Then the winning rate is calculated by self.wins divided by self.plays. Finally, we select the successor state with the highest winning rate and return the corresponding move.

==================== GREEDY SEARCH IMPLEMENTATION ===============

The Greedy Search is based on the maze distance between the successor position and the goal (ghosts/ food dots/ capsule)
