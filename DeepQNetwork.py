
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from PIL import Image
import random


def prepro(image_seq):
	processed = []
	for image in image_seq:
		color = Image.fromarray(image)
		color = color.convert("L")
		color.save("./color.jpg")
		grey = np.dot(image[...,:3], [0.299, 0.587, 0.114])
		grey = grey[50:, :]
		grey = grey[::2,::2]
		black = Image.fromarray(grey)
		black=black.convert("L")
		black.save("./grey.jpg")
		processed.append(grey)
	return np.stack(processed, axis=-1)


class Q_network():

	def __init__(self, env, epsilon=0.1, gamma=0.99, mem_size=15000):		
		#initialize some hyperparameters
		self.memory = []
		self.mem_size = mem_size
		self.greedy = epsilon
		self.gamma = gamma
		self.env = env
		self.hundred_ep_reward = []
		
		
		self.input_layer = tf.placeholder(tf.float32, shape=[None, 80, 80, 4])
		self.conv1=tf.layers.conv2d(self.input_layer, filters=16, kernel_size=[8, 8],
									strides=(4, 4), padding="same", activation=tf.nn.relu)

		self.conv2=tf.layers.conv2d(self.conv1, filters=32, kernel_size=[4, 4],
									strides=(2,2),padding="same", activation=tf.nn.relu)
		self.flattened=tf.reshape(self.conv2,[-1, 32*10*10])
		self.fc1 = tf.layers.dense(self.flattened, units=256, activation=tf.nn.relu)
		self.logits = tf.layers.dense(self.fc1, units=3) #check this line0
		self.output = tf.argmax(self.logits)



		self.target_q = tf.placeholder(tf.float32, shape=[None])
		self.actions = tf.cast(tf.placeholder(tf.int32, shape=[None, 3]), tf.float32)

		self.Q = tf.reduce_sum(tf.multiply(self.logits, self.actions), 1)
		
		self.loss = tf.reduce_sum(tf.square(self.target_q - self.Q))
		
		self.opt = tf.train.AdamOptimizer()
		self.update=self.opt.minimize(self.loss)
		self.saver = tf.train.Saver()


	

	def run_episode(self, render, episode):
		update_frequency=3
		s1 = [self.env.reset()]*4
		self.env.step(1)
		done = False
		episode_reward = 0
		n=0
		while not done:
			pr1 = prepro(s1)
			if(np.random.random_sample() > self.greedy):
				a, l = self.sess.run([self.output, self.logits], feed_dict={self.input_layer:[pr1]})
				a = np.argmax(l)

			else:
				a = np.random.choice([0,1,2])
			a +=1
			s2, r, done, _ = self.env.step(a)
			a -=1
			episode_reward += r
			if render == True:
				self.env.render()
			if done == True:
				s2 = np.zeros((210, 160, 3))
			new_state = s1[1:]
			new_state.append(s2)
			pr2 = prepro(new_state)
			transition = (pr1, a, r, pr2)
			self.add_to_memory(transition)
			s1=new_state
			n+=1
			if n%update_frequency==0 and episode != 0:
				self.update_network()

		

		if len(self.hundred_ep_reward) >=100:
			self.hundred_ep_reward.pop(0)
			self.hundred_ep_reward.append(episode_reward)	
		else:
			self.hundred_ep_reward.append(episode_reward)
		
		



	def add_to_memory(self, transition):
		if len(self.memory) > self.mem_size:
			self.memory.pop(0)
			self.memory.append(transition)
		else:
			self.memory.append(transition)


	def update_network(self):
		if len(self.memory)>31:
			batches = random.sample(self.memory, 32)
		else: batches=self.memory[:]
		inputs = [np.reshape(transition[0], (80, 80, 4)) for transition in batches]
		rewards, actions = self.discount_rewards(batches)
		self.sess.run([self.update],feed_dict={self.input_layer:inputs, self.actions:actions, self.target_q: rewards})

	
	def discount_rewards(self, batches):
		rewards = []
		actions = []
		#transitiong = (xt, at, rt, xt+1)
		for transition in batches:
			#print(transition[2])
			actions.append(self.one_hot(transition[1]))
			if np.count_nonzero(transition[3]) == 0:
				rewards.append(transition[2])
			else:
				image = np.reshape(transition[3], (80, 80, 4))#why does this happen
				target = self.sess.run([self.logits], feed_dict={self.input_layer:[image]})
				rewards.append((transition[2] + np.multiply(self.gamma, np.max(target[0][0]) ) ) )
		a = np.reshape(actions, (len(actions), 3))
		return np.reshape(rewards, (len(rewards))), a

	def one_hot(self,a):
		one_h = np.zeros(3)
		one_h[a]=1
		return one_h

	def train(self, num_episodes, render=False):
		n = 0
		reporting_frequency = 50
		annealing_frequency = 299
	
		while(n<num_episodes):
			if n%annealing_frequency==0 and n!= 0:
				if self.greedy >= 0.05:
					self.greedy -= 0.01

				save_path=self.saver.save(self.sess, "./models/breakout.ckpt")
				print("Model saved in file: %s" % save_path)
	
			if (n%reporting_frequency== 0) and n!=0:
				print(sum(self.hundred_ep_reward)/len(self.hundred_ep_reward))
			self.run_episode(render, n)
			n+=1
			if n%10 == 0:
				print("Running eposide #" + str(n))


	def init_session(self, growth=True, restore=False):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=growth
		config.gpu_options.per_process_gpu_memory_fraction=0.7
		config.log_device_placement=True
		self.sess=tf.Session(config=config)
		if restore:
			self.saver.restore(self.sess, "./models/breakout.ckpt")
			print("Model restored.")
		else:
			init = tf.global_variables_initializer()
			trainables = tf.trainable_variables()
			self.sess.run(init)
			print("Model Initialized")



env = gym.make("Breakout-v0")
tf.reset_default_graph()
network = Q_network(env)
network.init_session(restore=True)
network.train(7000, render=False)


