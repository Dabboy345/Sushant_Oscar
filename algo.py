# baseline_team.py
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


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point
from game import Actions
from util import PriorityQueue


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
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

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
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
        if pos != nearest_point(pos):
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

"""
class OffensiveReflexAgent(ReflexCaptureAgent):

  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.


    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):

    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.


    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}


"""


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    An offensive agent using minimax with alpha-beta pruning.
    """

    def choose_action(self, game_state):
        """
        Uses minimax with alpha-beta pruning to choose action.
        """
        def minimax(state, depth, agent_index, alpha, beta):
            # Base case: max depth or terminal state
            if depth == 0 or state.is_over():
                return self.evaluate_state(state), None
            
            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluate_state(state), None
            
            # Get opponents
            opponents = self.get_opponents(state)
            num_agents = len(opponents) + 1  # self + opponents
            next_agent = (agent_index + 1) % num_agents
            
            # Map indices correctly
            if agent_index == 0:  # Our turn (maximizing)
                next_agent_real = opponents[0] if len(opponents) > 0 else self.index
            else:
                next_agent_real = opponents[next_agent - 1] if next_agent > 0 else self.index
            
            # Check if next agent is ours or we need to go deeper
            is_our_turn = (agent_index == 0)
            
            if is_our_turn:  # Maximizing player
                max_val = float('-inf')
                best_action = None
                for action in legal_actions:
                    successor = state.generate_successor(self.index, action)
                    val, _ = minimax(successor, depth, 1, alpha, beta)
                    if val > max_val:
                        max_val = val
                        best_action = action
                    alpha = max(alpha, val)
                    if beta <= alpha:
                        break
                return max_val, best_action
            else:  # Minimizing player (opponent)
                min_val = float('inf')
                # Assume opponent moves optimally
                for action in legal_actions:
                    successor = state.generate_successor(opponents[agent_index - 1], action)
                    # After all opponents move, reduce depth and go back to our turn
                    if agent_index >= len(opponents):
                        val, _ = minimax(successor, depth - 1, 0, alpha, beta)
                    else:
                        val, _ = minimax(successor, depth, agent_index + 1, alpha, beta)
                    min_val = min(min_val, val)
                    beta = min(beta, val)
                    if beta <= alpha:
                        break
                return min_val, None
        
        # Start minimax
        _, best_action = minimax(game_state, 2, 0, float('-inf'), float('inf'))
        
        if best_action is None:
            legal_actions = game_state.get_legal_actions(self.index)
            return random.choice(legal_actions) if legal_actions else Directions.STOP
        
        return best_action
    
    def evaluate_state(self, game_state):
        """
        Evaluate the game state for offensive play.
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        score = self.get_score(game_state) * 100
        
        # Food carrying
        food_carrying = my_state.num_carrying
        score += food_carrying * 10
        
        # Distance to food
        food_list = self.get_food(game_state).as_list()
        if food_list:
            min_food_dist = min([self.get_maze_distance(my_pos, food) for food in food_list])
            score -= min_food_dist
        
        # Distance to home if carrying food
        if food_carrying > 0:
            dist_to_home = self.get_maze_distance(my_pos, self.start)
            score -= dist_to_home * 2
        
        # Avoid ghosts
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None]
        if my_state.is_pacman and ghosts:
            ghost_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]
            min_ghost_dist = min(ghost_dists)
            if min_ghost_dist < 3:
                score -= 100 / (min_ghost_dist + 1)
        
        return score
    


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A defensive agent that patrols borders and uses A* to chase invaders.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.border_points = []
        self.patrol_index = 0

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        # Calculate border points
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        
        if self.red:
            mid = width // 2 - 1
        else:
            mid = width // 2
        
        self.border_points = []
        for y in range(height):
            if not game_state.has_wall(mid, y):
                self.border_points.append((mid, y))
    
    def choose_action(self, game_state):
        """
        Patrol borders or chase invaders using A*.
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        # Check for invaders
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        
        if invaders:
            # Chase nearest invader using A*
            target = min(invaders, key=lambda e: self.get_maze_distance(my_pos, e.get_position())).get_position()
            path = self.a_star_search(game_state, my_pos, target)
            if path and len(path) > 0:
                return path[0]
        
        # Patrol border
        target_patrol = self.border_points[self.patrol_index % len(self.border_points)]
        if self.get_maze_distance(my_pos, target_patrol) < 2:
            self.patrol_index += 1
            target_patrol = self.border_points[self.patrol_index % len(self.border_points)]
        
        path = self.a_star_search(game_state, my_pos, target_patrol)
        if path and len(path) > 0:
            return path[0]
        
        # Fallback
        legal_actions = game_state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else Directions.STOP
    
    def a_star_search(self, game_state, start, goal):
        """
        A* search to find path from start to goal.
        Returns list of actions.
        """
        # Get initial successors
        start_successors = self.get_successors(game_state, start)
        
        # Initialize frontier
        frontier = PriorityQueue()
        
        # Initialize expanded nodes
        expanded_nodes = []
        expanded_nodes.append(start)
        
        # Put first successors in frontier with cost + heuristic
        for position, action, cost in start_successors:
            path = [action]
            total_cost = cost
            heuristic = self.get_maze_distance(position, goal)
            frontier.push((position, path, total_cost), total_cost + heuristic)
        
        # While frontier not empty
        while not frontier.is_empty():
            # Variable to check if node has been expanded
            expanded = False
            current_pos, current_path, current_cost = frontier.pop()
            
            # Check if node is already expanded
            for exp_pos in expanded_nodes:
                if exp_pos == current_pos:
                    expanded = True
                    break
            
            # If not expanded
            if not expanded:
                # Add to expanded
                expanded_nodes.append(current_pos)
                
                # Check if this is the goal node
                if current_pos == goal or self.get_maze_distance(current_pos, goal) <= 1:
                    return current_path
                
                # Get node successors
                next_successors = self.get_successors(game_state, current_pos)
                
                # Add successors to frontier with cost + heuristic
                for next_pos, next_action, step_cost in next_successors:
                    new_path = current_path + [next_action]
                    new_cost = current_cost + step_cost
                    heuristic = self.get_maze_distance(next_pos, goal)
                    frontier.push((next_pos, new_path, new_cost), new_cost + heuristic)
        
        return []
    
    def get_successors(self, game_state, position):
        """
        Returns list of (successor_position, action, cost) tuples.
        """
        successors = []
        x, y = int(position[0]), int(position[1])
        
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.direction_to_vector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            
            if not game_state.has_wall(next_x, next_y):
                next_pos = (next_x, next_y)
                successors.append((next_pos, action, 1))
        
        return successors
