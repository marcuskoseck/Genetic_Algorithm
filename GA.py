"""
Author: Marcus Koseck

This code is based on the following paper:
https://arxiv.org/pdf/1712.06567.pdf

DESIGN CHOICES:
You will notice that I save all of the models in memory before
I begin training. The Parallel function doesn't allow me to 
cache the keras models and that is my work around.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import tensorflow as tf
import keras
from keras import layers
import time
import concurrent.futures
import sys
from joblib import Parallel, delayed, Memory

def make_model(num_observations,num_actions):
    '''
    Input:
    num_observations: this is an integer and is the size of the array containing the state
    num_actions: This is an integer and is the size of the action space.

    Description:
    This function initializes a keras model

    Return:
    A keras Sequential Object.
    '''
    model = keras.Sequential(
        [
            layers.Input(shape = num_observations),
            layers.Dense(64,activation = "relu"),
            layers.Dense(64,activation = "relu"),
            layers.Dense(num_actions),
            layers.Dense(1,activation="softmax")
        ])
    #model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss = "mean_absolute_error",metrics=["accuracy"])
    return model

def save_model(model,directory,filename):
    '''
    Input:
    model: this is the model that is being saved
    directory: This is the save folder
    filename: this is what the filename will be

    Description:
    This function saves a keras model into a specific directory

    return:
    the path where the model is saved
    '''
    if not os.path.isdir(directory):
        os.makedirs(directory)
    full_path = directory+filename
    model.save(full_path,include_optimizer=False)
    return full_path

def load_model(path):
    '''
    Input: 
    path: model path

    Description:
    This function loads a function from memory

    return:
    A keras sequential object 
    '''
    return tf.keras.models.load_model(path,compile=False)

def rename_model(current_path,directory):
    '''
    Input:
    current_path: This is the current location of the model
    directory: This is where the model will be renamed and saved to

    Description:
    This function saves a model and saves it in a specified location

    return:
    the path where the model has been saved.
    '''
    replacement_path = r"\elite.h5"
    os.rename(current_path,f"{directory}\{replacement_path}")
    return f"{directory}\{replacement_path}"

def predict(path,state):
    '''
    Input: 
    path: the path of the model being used
    state: the state that the agent is currently located

    Description:
    This function calculates the probability of an action being selected

    return:
    an array of probabilities that an action will be taken.
    '''
    model = load_model(path)
    return model(state)
    #return model.predict(state,verbose=0,steps=1)

def generate_initial_generation(num_agents,num_observations,num_actions):
    '''
    Input: 
    num_agents: The number of agents for this specfic generation
    num_observations: this is an integer and is the size of the array containing the state
    num_actions: This is an integer and is the size of the action space.

    Description: 
    This function makes "num_agents" number of models and returns them in a list.

    return:
    a list of keras sqeuenatial models.
    '''
    return [make_model(num_observations,num_actions) for _ in range(num_agents)]

def generate_children_models(model_path,num_agents,num_observations,num_actions):
    '''
    input:
    model_path: this is the model that has been selected to produce offspring
    num_agents: the number of agents needed to be produced in the next generation
    num_observations: this is an integer and is the size of the array containing the state
    num_actions: This is an integer and is the size of the action space.
    
    Description:
    This function produces a finite number of models by randomly sampling from a gaussian
    distribution and adding the sampled matrix to the weights of the neural network.

    return:
    A list of keras sequenatial models.
    '''
    #generate offspring
    model = load_model(model_path)
    next_generation = [make_model(num_observations,num_actions) for _ in range(num_agents)]
    for child in next_generation:
        for layer_idx,layer_object in enumerate(model.layers):
            weights_size = layer_object.get_weights()[0].shape
            bias_size = layer_object.get_weights()[1].shape

            new_weights = layer_object.get_weights()[0]+0.02*np.random.normal(0,1,weights_size)
            new_bias = layer_object.get_weights()[1]+0.02*np.random.normal(0,1,bias_size)
            child.layers[layer_idx].set_weights([new_weights,new_bias])
    next_generation.append(model)
    return next_generation

def save_frames_as_gif(frames,generation, path, filename):
    '''
    Input:
    frames: this is a list of images in (R,G,B) format
    generation: this is an integer that specifies what generation is being rendered
    path: this is the directory path for saving the mp4 files
    filename: this is the name of the mp4 file.

    Description:
    This function creates mp4 files of a sequence of images provided in a list

    return:
    None
    '''
    if not os.path.isdir(path):
        os.makedirs(path)

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')
    plt.title(f"Run from generation {generation}")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps = 30,metadata={'artist':'Me'},bitrate=1800)
    anim.save(path + filename, writer)
    #plt.clf()
    plt.close()

def run_simulation(path,env):
    '''
    input:
    path: the path of the agent being used
    env: a copy of the reinforcement learning environment

    Description:
    This function runs through the simulation of the environment.

    return:
    the reward after finishing the episode.
    '''
    state, info = env.reset()
    state = tf.convert_to_tensor(np.reshape(state,(1,state.shape[0])))
    running_reward = 0
    while True:
        action = np.random.choice(range(env.action_space.n),1,predict(path,state)).item()
        new_state,reward,done,_,info = env.step(action)
        running_reward+=reward
        state = new_state
        state = tf.convert_to_tensor(np.reshape(state,(1,state.shape[0])))
        if done:
            break
    return running_reward

def find_elite_model_parallel(models,envs,NUMBER_OF_REEVAL_STEPS):
    '''
    input:
    models: the models that are being reevaluated
    envs: the list of environments that correspond to the models
    NUMBER_OF_REEVAL_STEPS: the number of times needed to reevaluate the model

    Description:
    this function reevaluates the models a finite number of times to account for 
    any stochastic behavior in the environment. Then, an average is taken to determine
    what the best model is on average. The model with the highest reward on average is
    considered the best model.

    return:
    The model with the highest reward on average.
    '''
    for i in range(NUMBER_OF_REEVAL_STEPS):
        if i==0:
            fitness_of_each_model = Parallel(n_jobs=10)(delayed(run_simulation)(agent, env) for agent,env in zip(best_agents,envs))
        else:
            next_row = Parallel(n_jobs=10)(delayed(run_simulation)(agent, env) for agent,env in zip(best_agents,envs))
            fitness_of_each_model = np.vstack((fitness_of_each_model,next_row))
    means = fitness_of_each_model.mean(axis=0)
    means = means/NUMBER_OF_REEVAL_STEPS
    return models[np.argmax(means)] #This model is the best of the best

if __name__ == "__main__" :
    #THESE ARE CHANGABLE PARAMETERS
    num_agents = 100
    best_performers = 10 #This is the number of selected individuals
    NUMBER_OF_REEVAL_STEPS = 30 #NUMBER_OF_REEVAL_STEPS
    NUMBER_OF_GENERATIONS = 100
    #####################################
    # DON'T CHANGE CODE BELOW THIS POINT#
    #####################################
    setup = gym.make("MsPacman-ramNoFrameskip-v4",render_mode="rgb_array")
    setup.metadata["render_fps"] = 30
    num_actions = setup.action_space.n
    num_observations = setup.observation_space.shape
    envs = [gym.make("MsPacman-ramNoFrameskip-v4",render_mode="rgb_array") for _ in range(num_agents)]
    highest_reward_list = []
    for gen in range(NUMBER_OF_GENERATIONS):
        start_time = time.perf_counter()
        parent_directory_for_models = rf"C:\Users\marcu\Documents\machine_learning_projects\Genetic_Algorithms\saved_models\generation_{gen}"
        if gen == 0:
            agents = generate_initial_generation(num_agents,num_observations,num_actions)
        else:
            #randomly_selected_parent = np.random.choice(agents[:best_performers])
            randomly_selected_parent = agents
            agents = generate_children_models(randomly_selected_parent,num_agents,num_observations,num_actions) 
        agent_path = [save_model(agent,rf"{parent_directory_for_models}",rf"\model{id(agent)}.h5") for agent in agents]
        
        fitness_of_each_model = Parallel(n_jobs=10)(delayed(run_simulation)(agent, env) for agent,env in zip(agent_path,envs))
        
        best_agents_idx = np.argsort(fitness_of_each_model)[::-1][:best_performers]
        highest_reward_list.append(fitness_of_each_model[best_agents_idx[0]])
        if gen == 0:
            best_agents = [agent_path[best_agents_idx[idx]] for idx in range(best_performers)]
        else:
            best_agents = [agent_path[best_agents_idx[idx]] for idx in range(best_performers-1)]
        elite = find_elite_model_parallel(best_agents,envs,NUMBER_OF_REEVAL_STEPS)
        elite = rename_model(elite,parent_directory_for_models)
        best_agents.append(elite)
        agents = elite
        print(f"Generation {gen+1}/{NUMBER_OF_GENERATIONS} took {time.perf_counter()-start_time} with highest reward being {highest_reward_list[gen]}")
    plt.plot([_ for _ in range(len(highest_reward_list))],highest_reward_list)
    plt.xlabel("Generation")
    plt.ylabel("Reward Achieved")
    plt.title("Generation V Reward")
    plt.savefig("generationsVrewards.png")
