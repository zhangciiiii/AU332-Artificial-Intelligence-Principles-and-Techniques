from agent import Agent, D_Q_Agent
import time

maze = '2'

if maze == '1':
    from maze_env1 import Maze
elif maze == '2':
    from maze_env2 import Maze


if __name__ == "__main__":
    ### START CODE HERE ###
    # This is an agent with random policy. You can learn how to interact with the environment through the code below.
    # Then you can delete it and write your own code.

    env = Maze()
    training_epoch = 100 if maze == '1' else 1000
    agent = D_Q_Agent(training_epoch)
    
    for episode in range(training_epoch):
        agent.if_rewarded = False
        s = env.reset()
        while True:
            # env.render()                 # You can comment all render() to turn off the graphical interface in training process to accelerate your code.
            chosen_direction = agent.choose_action(s,episode)
            s_, r, done = env.step(chosen_direction)
            agent.update_Q_value(s, chosen_direction, r)
            if s_[-1]:
                agent.if_rewarded = True
                agent.if_rewarded_in_the_whole_training = True
            s = s_
            agent.simulative_training(100)    

            if done:
                #env.render()
                time.sleep(0.5)
                break
        
        print('episode:', episode)
    print('Training Finished! Now Demonstrate the Optimal Policy:')
    while True :
        s = env.reset()
        agent.if_rewarded = False
        while True:
            env.render() 
            chosen_direction = agent.choose_action(s,episode,demonstration = True)
            s_, r, done = env.step(chosen_direction)
            s = s_
            if s_[-1]:
                agent.if_rewarded = True
            if done:
                env.render()
                time.sleep(0.5)
                break


    ### END CODE HERE ###

    print('\ntraining over\n')
