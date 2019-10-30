import numpy as np
import pandas as pd
import copy
import csv
import random


class Agent:
    ### START CODE HERE ###

    def __init__(self, actions):
        self.actions = actions 
        self.epsilon = 1

    def choose_action(self, observation):
        '''
        obeservation :
        [5.0, 5.0, 35.0, 35.0, False]
        '''
        action = np.random.choice(self.actions)
        return 2

    ### END CODE HERE ###

class D_Q_Agent:

    def __init__(self,N):

        self.N_episode = N
        self.discount = 0.8
        self.epsilon = 0.2
        self.current_epsilon = self.epsilon
        self.learning_rate = 0.1
        self.living_cost = -0.01
        self.if_rewarded = False
        self.if_rewarded_in_the_whole_training = False

        self.Q_value_table1 = np.zeros((4, 6, 6))
        self.Q_value_table1[0,:,0] = -100
        self.Q_value_table1[1,:,5] = -100
        self.Q_value_table1[2,5,:] = -100
        self.Q_value_table1[3,0,:] = -100

        self.Q_value_table2 = copy.deepcopy(self.Q_value_table1)
        
        self.Q_value_table1_prev = copy.deepcopy(self.Q_value_table1)
        self.Q_value_table2_prev = copy.deepcopy(self.Q_value_table1)
        
        self.reward_table1 = np.zeros((6, 6))
        self.reward_table2 = np.zeros((6, 6))
        self.visited = {}
        

            

    def MaxQ_direction(self,location):
        '''
        find the max Q-value 
        '''
        x,y = int((location[0]-5)/40), int((location[1]-5)/40)
        # print(x)
        if self.if_rewarded:
            up = self.Q_value_table2[0][x][y] if y>0 else float('-Inf')
            down = self.Q_value_table2[1][x][y] if y<5 else float('-Inf')
            right = self.Q_value_table2[2][x][y] if x<5 else float('-Inf')
            left = self.Q_value_table2[3][x][y] if x>0 else float('-Inf')
            four_direction_value = [up, down, right, left]
        else:
            up = self.Q_value_table1[0][x][y] if y>0 else float('-Inf')
            down = self.Q_value_table1[1][x][y] if y<5 else float('-Inf')
            right = self.Q_value_table1[2][x][y] if x<5 else float('-Inf')
            left = self.Q_value_table1[3][x][y] if x>0 else float('-Inf')
            four_direction_value = [up, down, right, left]
        # print(four_direction_value)
        max_value = float('-Inf')
        max_direction = 1
        other_potential_direction = []
        for i in range(0,4):
            if four_direction_value[i] >= max_value:
                max_value = four_direction_value[i]
                max_direction = i
            if four_direction_value[i] != float('-Inf') :
                other_potential_direction.append(i)

        other_potential_direction.remove(max_direction)

        return max_direction, other_potential_direction


    def update_Q_value(self, location, direction, reward = None):
        x,y = int((location[0]-5)/40), int((location[1]-5)/40)
        if self.if_rewarded:
            if reward is not None:
                if direction==0:
                    self.reward_table2[x,y-1] = reward
                if direction==1:
                    self.reward_table2[x,y+1] = reward
                if direction==2:
                    self.reward_table2[x+1,y] = reward
                if direction==3:
                    self.reward_table2[x-1,y] = reward

            if ('({},{})'.format(x,y) in self.visited):
                if direction not in self.visited['({},{})'.format(x,y)]:
                    self.visited['({},{})'.format(x,y)].append(direction)
            else:
                self.visited['({},{})'.format(x,y)] = [direction] 

            self.Q_value_table2[direction,x,y] = (1-self.learning_rate)*self.Q_value_table2_prev[direction,x,y]
            if direction==0:
                self.Q_value_table2[direction,x,y]  += self.learning_rate*(self.reward_table2[x,y-1] + self.living_cost + max(self.Q_value_table2_prev[:,x,y-1])*self.discount) 
            if direction==1:
                self.Q_value_table2[direction,x,y]  += self.learning_rate*(self.reward_table2[x,y+1] + self.living_cost + max(self.Q_value_table2_prev[:,x,y+1])*self.discount)
            if direction==2:
                self.Q_value_table2[direction,x,y]  += self.learning_rate*(self.reward_table2[x+1,y] + self.living_cost + max(self.Q_value_table2_prev[:,x+1,y])*self.discount)
            if direction==3:
                self.Q_value_table2[direction,x,y]  += self.learning_rate*(self.reward_table2[x-1,y] + self.living_cost + max(self.Q_value_table2_prev[:,x-1,y])*self.discount)
            
            self.Q_value_table2_prev = copy.deepcopy(self.Q_value_table2)

        else :

            if reward is not None:
                if direction==0:
                    self.reward_table1[x,y-1] = reward
                if direction==1:
                    self.reward_table1[x,y+1] = reward
                if direction==2:
                    self.reward_table1[x+1,y] = reward
                if direction==3:
                    self.reward_table1[x-1,y] = reward

            if ('({},{})'.format(x,y) in self.visited):
                if direction not in self.visited['({},{})'.format(x,y)]:
                    self.visited['({},{})'.format(x,y)].append(direction)
            else:
                self.visited['({},{})'.format(x,y)] = [direction] 

            self.Q_value_table1[direction,x,y] = (1-self.learning_rate)*self.Q_value_table1_prev[direction,x,y]
            if direction==0:
                self.Q_value_table1[direction,x,y]  += self.learning_rate*(self.reward_table1[x,y-1] + self.living_cost + max(self.Q_value_table1_prev[:,x,y-1])*self.discount) 
            if direction==1:
                self.Q_value_table1[direction,x,y]  += self.learning_rate*(self.reward_table1[x,y+1] + self.living_cost + max(self.Q_value_table1_prev[:,x,y+1])*self.discount)
            if direction==2:
                self.Q_value_table1[direction,x,y]  += self.learning_rate*(self.reward_table1[x+1,y] + self.living_cost + max(self.Q_value_table1_prev[:,x+1,y])*self.discount)
            if direction==3:
                self.Q_value_table1[direction,x,y]  += self.learning_rate*(self.reward_table1[x-1,y] + self.living_cost + max(self.Q_value_table1_prev[:,x-1,y])*self.discount)
            
            self.Q_value_table1_prev = copy.deepcopy(self.Q_value_table1)


    def choose_action(self, location,current_episode, demonstration = False):
        

        max_Q_direction, other_potential_direction = self.MaxQ_direction(location)
        num = np.random.uniform()

        # for demonstration
        if demonstration:
            return max_Q_direction

        # finding the reward in the training, but not yet in this episode
        if self.if_rewarded_in_the_whole_training and not self.if_rewarded:
            return max_Q_direction

        # after reward, exploration decay
        if self.if_rewarded:
            self.current_epsilon = self.epsilon *(1 - current_episode/self.N_episode)

        # before finding reward or after finding reward and try to find exit
        
        if num  > self.current_epsilon:
            return max_Q_direction
        else :
            return (np.random.choice(other_potential_direction))

    def simulative_training(self,N):
        
        for _ in range(N):
            location = random.choice(list(self.visited))
            x,y = int(location[1]), int(location[3])
            direction = np.random.choice(self.visited[location])
            
            self.update_Q_value([40*x+5,40*y+5], direction)







