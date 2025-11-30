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
from util import PriorityQueue
from game import Actions


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




class OffensiveReflexAgent(ReflexCaptureAgent):
   """
 A reflex agent that seeks food. This is an agent
 we give you to get an idea of what an offensive agent might look like,
 but it is by no means the best or only way to build an offensive agent.
 """


   def choose_action(self, game_state):
       """
       Simple offense:
       - If carrying more than 5 food, go back to base (start).
       - Else, try to take a capsule first (if closer than opponents).
       - Else, chase nearest food using A*.
       """
       state = game_state.get_agent_state(self.index)
       my_pos = state.get_position()


       # Return to base if carrying enough food
       if state.num_carrying > 4:
           path = self.a_star_search(game_state, my_pos, self.start)
           if path:
               return path[0]
           actions = game_state.get_legal_actions(self.index)
           return random.choice(actions) if actions else Directions.STOP


       # Try to take a capsule first if advantageous
       target = None
       capsules = []
       try:
           capsules = self.get_capsules(game_state)
       except Exception:
           capsules = []


       if capsules:
           opponents = self.get_opponents(game_state)
           opp_positions = []
           for i in opponents:
               s = game_state.get_agent_state(i)
               p = s.get_position()
               if p is not None:
                   opp_positions.append(p)
           # Choose a capsule where we are closer than any visible opponent
           best_margin = -float('inf')
           for cap in capsules:
               my_d = self.get_maze_distance(my_pos, cap)
               if opp_positions:
                   opp_d = min(self.get_maze_distance(p, cap) for p in opp_positions)
               else:
                   opp_d = float('inf')
               margin = opp_d - my_d
               if margin > 0 and margin > best_margin:
                   best_margin = margin
                   target = cap


       # If no good capsule target, chase nearest food
       if target is None:
           food_list = self.get_food(game_state).as_list()
           if not food_list:
               return Directions.STOP
           best_dist = float('inf')
           for food in food_list:
               d = self.get_maze_distance(my_pos, food)
               if d < best_dist:
                   best_dist = d
                   target = food


       if target is None:
           actions = game_state.get_legal_actions(self.index)
           return random.choice(actions) if actions else Directions.STOP


       path = self.a_star_search(game_state, my_pos, target)
       if path:
           return path[0]
       actions = game_state.get_legal_actions(self.index)
       return random.choice(actions) if actions else Directions.STOP


   def a_star_search(self, game_state, start, goal):
       """
       Returns list of actions from start to goal using A*.
       Heuristic: util.manhattan_distance (faster than maze distance).
       """
       if start == goal:
           return []


       frontier = PriorityQueue()
       frontier.push((start, [], 0), 0)
       visited = set()


       while not frontier.is_empty():
           pos, path, cost_so_far = frontier.pop()
           if pos in visited:
               continue
           visited.add(pos)


           if pos == goal:
               return path


           for next_pos, action, step_cost in self.get_successors(game_state, pos):
               if next_pos in visited:
                   continue
               new_path = path + [action]
               new_cost = cost_so_far + step_cost
               try:
                   heuristic = util.manhattan_distance(next_pos, goal)
               except AttributeError:
                   # Fallback if name differs
                   heuristic = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
               frontier.push((next_pos, new_path, new_cost), new_cost + heuristic)


       return []
  


   def get_successors(self, game_state, position):
       """
       Returns list of (successor_position, action, cost).
       """
       successors = []
       x, y = int(position[0]), int(position[1])
       for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
           dx, dy = Actions.direction_to_vector(action)
           nx, ny = int(x + dx), int(y + dy)
           if not game_state.has_wall(nx, ny):
               successors.append(((nx, ny), action, 1))
       return successors




class DefensiveReflexAgent(ReflexCaptureAgent):
   """
   A reflex agent that keeps its side Pacman-free. Again,
   this is to give you an idea of what a defensive agent
   could be like.  It is not the best or only way to make
   such an agent.
   Camper ghost: sit on a chosen border (camp) point.
   If an invader (enemy Pacman) is visible, chase it with A*.
   Otherwise stay at (or return to) camp.
   """


   def __init__(self, index, time_for_computing=.1):
       super().__init__(index, time_for_computing)
       self.camp_position = None


   def register_initial_state(self, game_state):
       super().register_initial_state(game_state)
       width = game_state.data.layout.width
       height = game_state.data.layout.height
       mid_x = (width // 2 - 1) if self.red else (width // 2)
       border_points = [(mid_x, y) for y in range(height) if not game_state.has_wall(mid_x, y)]
       # Pick central usable point to camp
       if border_points:
           self.camp_position = border_points[len(border_points) // 2]
       else:
           self.camp_position = self.start  # Fallback


   def choose_action(self, game_state):
       my_pos = game_state.get_agent_state(self.index).get_position()


       # Detect visible invaders
       enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
       invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]


       if invaders:
           # Chase closest invader with A*
           target = None
           best_dist = float('inf')
           for e in invaders:
               pos = e.get_position()
               dist = self.get_maze_distance(my_pos, pos)
               if dist < best_dist:
                   best_dist = dist
                   target = pos
           path = self.a_star_search(game_state, my_pos, target)
           if path:
               return path[0]
           # Fallback: any legal action
           legal = game_state.get_legal_actions(self.index)
           return random.choice(legal) if legal else Directions.STOP


       # No invader: stay or return to camp
       if my_pos != self.camp_position:
           path = self.a_star_search(game_state, my_pos, self.camp_position)
           if path:
               return path[0]
       # Already camping or no path: do nothing
       return Directions.STOP
  


   def a_star_search(self, game_state, start, goal):
       """
       Defensive agent A*: now uses maze distance heuristic.
       """
       if start == goal:
           return []


       frontier = PriorityQueue()
       frontier.push((start, [], 0), 0)
       visited = set()


       while not frontier.is_empty():
           pos, path, cost_so_far = frontier.pop()
           if pos in visited:
               continue
           visited.add(pos)


           if pos == goal:
               return path


           for next_pos, action, step_cost in self.get_successors(game_state, pos):
               if next_pos in visited:
                   continue
               new_path = path + [action]
               new_cost = cost_so_far + step_cost
               heuristic = self.get_maze_distance(next_pos, goal)
               frontier.push((next_pos, new_path, new_cost), new_cost + heuristic)


       return []
  


   def get_successors(self, game_state, position):
       """
       Returns list of (successor_position, action, cost).
       """
       successors = []
       x, y = int(position[0]), int(position[1])
       for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
           dx, dy = Actions.direction_to_vector(action)
           nx, ny = int(x + dx), int(y + dy)
           if not game_state.has_wall(nx, ny):
               successors.append(((nx, ny), action, 1))
       return successors



