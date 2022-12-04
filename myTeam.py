# smartTeam.py
# ---------------
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


# smartTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
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
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class SmartAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.myAgents = CaptureAgent.get_team(self, game_state)
        self.opAgents = CaptureAgent.get_opponents(self, game_state)
        self.myFoods = CaptureAgent.get_food(self, game_state).as_list()
        self.opFoods = CaptureAgent.get_food_you_are_defending(self, game_state).as_list()
        self.alpha = 0.2
        self.discountRate = 0.8
        self.weightsOffense = {'closest-food': 0, 'bias': 0, '#-of-ghosts-1-step-away': 0, 'successorScore': 0, 'eats-food': 0}
        CaptureAgent.register_initial_state(self, game_state)
		

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveAgent(SmartAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def get_features(self, game_state, action):
        # Extract the grid of food 
        food = self.myFoods
        ghosts = []
        opAgents = CaptureAgent.get_opponents(self, game_state)
        # Get ghost locations and states if observable
        if opAgents:
            for opponent in opAgents:
                opPos = game_state.get_agent_position(opponent)
                opIsPacman = game_state.get_agent_state(opponent).is_pacman
                if opPos and not opIsPacman: 
                    ghosts.append(opPos)
		
        # Initialize features
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        # Successor Score
        features['successorScore'] = self.get_score(successor)

        # Bias
        features["bias"] = 1.0
		
        # compute the location of pacman after he takes the action
        x, y = game_state.get_agent_position(self.index)
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)
		
        # Number of Ghosts 1-step away
        #features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in game_state.get_agent_position(opAgents) for g in ghosts)
        # if there is no danger of ghosts then add the food feature
        #if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            #features["eats-food"] = 1.0

        # Number of Ghosts scared
        #features['#-of-scared-ghosts'] = sum(game_state.get_agent_state(opponent).scared_timer != 0 for opponent in opAgents)
		
        if len(food) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, next_food) for next_food in food])
            features['distance_to_food'] = min_distance

        # Normalize and return
        features.divideAll(10.0)
        return features

    """
    Iterate through all features and for each feature, update
    its weight values using the following formula:
    w(i) = w(i) + alpha((reward + discount*value(nextState)) - Q(s,a)) * f(i)(s,a)
    """
    def get_weights(self, game_state, action):
        features = self.get_features(game_state, action)
        nextState = self.get_successor(game_state, action)

        # Calculate the reward. NEEDS WORK
        reward = nextState.get_score() - game_state.get_score()

        for feature in features:
            correction = (reward + self.discountRate*self.getValue(nextState)) - SmartAgent.evaluate(self, game_state, action)
            self.weights[feature] = self.weights[feature] + self.alpha * correction * features[feature]
    
    

    """
    Iterate through all q-values that we get from all
    possible actions, and return the highest q-value
    """
    def getValue(self, game_state):
        qVals = []
        legalActions = game_state.get_legal_actions(self.index)
        if len(legalActions) == 0:
            return 0.0
        else:
            for action in legalActions:
                qVals.append(SmartAgent.evaluate(self, game_state, action))
            return max(qVals)


class DefensiveAgent(SmartAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        myState = successor.get_agent_state(self.index)
        myPos = myState.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.is_pacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(myPos, a.get_position()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
        
