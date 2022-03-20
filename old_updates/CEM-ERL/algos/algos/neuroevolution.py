import random
import numpy as np
import math
import core.utils as utils

class SSNE:

	def __init__(self, args):
		self.gen = 0
		self.args = args;
		self.population_size = self.args.pop_size
		self.writer = args.writer

		#RL TRACKERS
		self.rl_policy = None
		self.selection_stats = {'elite': 0, 'selected': 0, 'discarded': 0, 'total': 0}


	def selection_tournament(self, index_rank, num_offsprings, tournament_size):
		"""Conduct tournament selection

			Parameters:
				  index_rank (list): Ranking encoded as net_indexes
				  num_offsprings (int): Number of offsprings to generate
				  tournament_size (int): Size of tournament

			Returns:
				  offsprings (list): List of offsprings returned as a list of net indices

		"""


		total_choices = len(index_rank)
		offsprings = []
		for i in range(num_offsprings):
			winner = np.min(np.random.randint(total_choices, size=tournament_size))
			offsprings.append(index_rank[winner])

		offsprings = list(set(offsprings))  # Find unique offsprings
		if len(offsprings) % 2 != 0:  # Number of offsprings should be even
			offsprings.append(index_rank[winner])
		return offsprings

	def list_argsort(self, seq):
		"""Sort the list

			Parameters:
				  seq (list): list

			Returns:
				  sorted list

		"""
		return sorted(range(len(seq)), key=seq.__getitem__)

	def regularize_weight(self, weight, mag):
		"""Clamps on the weight magnitude (reguralizer)

			Parameters:
				  weight (float): weight
				  mag (float): max/min value for weight

			Returns:
				  weight (float): clamped weight

		"""
		if weight > mag: weight = mag
		if weight < -mag: weight = -mag
		return weight

	def crossover_inplace(self, gene1, gene2):
		"""Conduct one point crossover in place

			Parameters:
				  gene1 (object): A pytorch model
				  gene2 (object): A pytorch model

			Returns:
				None

		"""


		keys1 =  list(gene1.state_dict())
		keys2 = list(gene2.state_dict())

		for key in keys1:
			if key not in keys2: continue

			# References to the variable tensors
			W1 = gene1.state_dict()[key]
			W2 = gene2.state_dict()[key]

			if len(W1.shape) == 2: #Weights no bias
				num_variables = W1.shape[0]
				# Crossover opertation [Indexed by row]
				try: num_cross_overs = random.randint(0, int(num_variables * 0.3))  # Number of Cross overs
				except: num_cross_overs = 1
				for i in range(num_cross_overs):
					receiver_choice = random.random()  # Choose which gene to receive the perturbation
					if receiver_choice < 0.5:
						ind_cr = random.randint(0, W1.shape[0]-1)  #
						W1[ind_cr, :] = W2[ind_cr, :]
					else:
						ind_cr = random.randint(0, W1.shape[0]-1)  #
						W2[ind_cr, :] = W1[ind_cr, :]

			elif len(W1.shape) == 1: #Bias or LayerNorm
				if random.random() <0.8: continue #Crossover here with low frequency
				num_variables = W1.shape[0]
				# Crossover opertation [Indexed by row]
				#num_cross_overs = random.randint(0, int(num_variables * 0.05))  # Crossover number
				for i in range(1):
					receiver_choice = random.random()  # Choose which gene to receive the perturbation
					if receiver_choice < 0.5:
						ind_cr = random.randint(0, W1.shape[0]-1)  #
						W1[ind_cr] = W2[ind_cr]
					else:
						ind_cr = random.randint(0, W1.shape[0]-1)  #
						W2[ind_cr] = W1[ind_cr]

	def mutate_inplace(self, gene):
		"""Conduct mutation in place

			Parameters:
				  gene (object): A pytorch model

			Returns:
				None

		"""
		mut_strength = 0.1
		num_mutation_frac = 0.05
		super_mut_strength = 10
		super_mut_prob = 0.05
		reset_prob = super_mut_prob + 0.02

		num_params = len(list(gene.parameters()))
		ssne_probabilities = np.random.uniform(0, 1, num_params) * 2

		for i, param in enumerate(gene.parameters()):  # Mutate each param

			# References to the variable keys
			W = param.data
			if len(W.shape) == 2:  # Weights, no bias

				num_weights = W.shape[0] * W.shape[1]
				ssne_prob = ssne_probabilities[i]

				if random.random() < ssne_prob:
					num_mutations = random.randint(0,
						int(math.ceil(num_mutation_frac * num_weights)))  # Number of mutation instances
					for _ in range(num_mutations):
						ind_dim1 = random.randint(0, W.shape[0]-1)
						ind_dim2 = random.randint(0, W.shape[-1]-1)
						random_num = random.random()

						if random_num < super_mut_prob:  # Super Mutation probability
							W[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * W[ind_dim1, ind_dim2])
						elif random_num < reset_prob:  # Reset probability
							W[ind_dim1, ind_dim2] = random.gauss(0, 0.1)
						else:  # mutauion even normal
							W[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * W[ind_dim1, ind_dim2])

						# Regularization hard limit
						W[ind_dim1, ind_dim2] = self.regularize_weight(W[ind_dim1, ind_dim2],
																	   self.args.weight_magnitude_limit)

			elif len(W.shape) == 1:  # Bias or layernorm
				num_weights = W.shape[0]
				ssne_prob = ssne_probabilities[i]*0.04 #Low probability of mutation here

				if random.random() < ssne_prob:
					num_mutations = random.randint(0,
						int(math.ceil(num_mutation_frac * num_weights)))  # Number of mutation instances
					for _ in range(num_mutations):
						ind_dim = random.randint(0, W.shape[0]-1)
						random_num = random.random()

						if random_num < super_mut_prob:  # Super Mutation probability
							W[ind_dim] += random.gauss(0, super_mut_strength * W[ind_dim])
						elif random_num < reset_prob:  # Reset probability
							W[ind_dim] = random.gauss(0, 1)
						else:  # mutauion even normal
							W[ind_dim] += random.gauss(0, mut_strength * W[ind_dim])

						# Regularization hard limit
						W[ind_dim] = self.regularize_weight(W[ind_dim], self.args.weight_magnitude_limit)

	def reset_genome(self, gene):
		"""Reset a model's weights in place

			Parameters:
				  gene (object): A pytorch model

			Returns:
				None

		"""
		for param in (gene.parameters()):
			param.data.copy_(param.data)

	def epoch(self, gen, pop, fitness_evals, migration):


		self.gen+= 1; num_elitists = int(self.args.elite_fraction * len(fitness_evals))
		if num_elitists < 2: num_elitists = 2

		# Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
		index_rank = self.list_argsort(fitness_evals); index_rank.reverse()
		elitist_index = index_rank[:num_elitists]  # Elitist indexes safeguard

		# Selection step
		offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - len(elitist_index) - len(migration), tournament_size=3)

		#Figure out unselected candidates
		unselects = []; new_elitists = []
		for net_i in range(len(pop)):
			if net_i in offsprings or net_i in elitist_index:
				continue
			else:
				unselects.append(net_i)
		random.shuffle(unselects)

		#Migration Tracker
		if self.rl_policy != None:  # RL Transfer happened
			self.selection_stats['total'] += 1.0
			if self.rl_policy in elitist_index:
				self.selection_stats['elite'] += 1.0
			elif self.rl_policy in offsprings:
				self.selection_stats['selected'] += 1.0
			elif self.rl_policy in unselects:
				self.selection_stats['discarded'] += 1.0
			self.rl_policy = None
			self.writer.add_scalar('elite_rate', self.selection_stats['elite']/self.selection_stats['total'], gen)
			self.writer.add_scalar('selection_rate', (self.selection_stats['elite']+self.selection_stats['selected'])/self.selection_stats['total'], gen)
			self.writer.add_scalar('discard_rate', self.selection_stats['discarded']/self.selection_stats['total'], gen)

		#Inheritance step (sync learners to population) --> Migration
		for policy in migration:
			replacee = unselects.pop(0)
			utils.hard_update(target=pop[replacee], source=policy)
			self.rl_policy = replacee

		# Elitism step, assigning elite candidates to some unselects
		for i in elitist_index:
			try: replacee = unselects.pop(0)
			except: replacee = offsprings.pop(0)
			new_elitists.append(replacee)
			utils.hard_update(target=pop[replacee], source=pop[i])


		# Crossover for unselected genes with 100 percent probability
		if len(unselects) % 2 != 0:  # Number of unselects left should be even
			unselects.append(unselects[random.randint(0, len(unselects)-1)])
		for i, j in zip(unselects[0::2], unselects[1::2]):
			off_i = random.choice(new_elitists);
			off_j = random.choice(offsprings)
			utils.hard_update(target=pop[i], source=pop[off_i])
			utils.hard_update(target=pop[j], source=pop[off_j])
			self.crossover_inplace(pop[i], pop[j])


		# Crossover for selected offsprings
		for i, j in zip(offsprings[0::2], offsprings[1::2]):
			if random.random() < self.args.crossover_prob:
				self.crossover_inplace(pop[i], pop[j])


		# Mutate all genes in the population except the new elitists
		for net_i in range(len(pop)):
			if net_i not in new_elitists:  # Spare the new elitists
				if random.random() < self.args.mutation_prob:
					self.mutate_inplace(pop[net_i])

class CovMatrix():
    def __init__(self, sigma, noise_multiplier, seed, diag_cov=False):
        self.sigma = sigma
        self.noise_multiplier = noise_multiplier
        self.diag_cov = diag_cov
        
        # Random number generator to sample weights
        # from the Gaussian distribution     
        self.rng = np.random.default_rng(seed)

    def init_covariance(self, centroid: np.ndarray) -> None:
        self.policy_dim = len(centroid)
        self.noise_matrix = np.diag(np.ones(self.policy_dim) * self.sigma)
        self.cov = np.diag(np.ones(self.policy_dim) * np.var(centroid)) + self.noise_matrix

    def update_noise(self) -> None:
        self.noise_matrix = self.noise_matrix * self.noise_multiplier

    def generate_weights(self, centroid, pop_size) -> np.ndarray:
        if self.diag_cov:
            # Separable CEM, useful when self.policy_dim >> 100
            # Use only diagonal of the covariance matrix
            param_noise = np.random.randn(self.pop_size, self.policy_dim)
            weights = centroid + param_noise * np.sqrt(np.diagonal(self.cov))
        else:
            # The params of policies at iteration t+1 are drawn according to a multivariate
            # Gaussian whose center is centroid and whose shape is defined by cov
            weights = self.rng.multivariate_normal(centroid, self.cov, pop_size)
        return weights

    def update_covariance(self, elites_weights: np.ndarray) -> None:
        self.cov = np.cov(elites_weights, rowvar=False) + self.noise_matrix

class CEM:

	def __init__(self, args):
		self.gen = 0
		self.args = args;
		self.pop_size = self.args.pop_size
		self.writer = args.writer
		self.matrix = CovMatrix(self.args.sigma, self.args.noise_multiplier, self.args.seed, self.args.diag_cov)

		#RL TRACKERS
		self.rl_policy = None
		self.selection_stats = {'elite': 0, 'selected': 0, 'discarded': 0, 'total': 0}

	def selection_tournament(self, index_rank, num_offsprings, tournament_size):
		"""Conduct tournament selection

			Parameters:
				  index_rank (list): Ranking encoded as net_indexes
				  num_offsprings (int): Number of offsprings to generate
				  tournament_size (int): Size of tournament

			Returns:
				  offsprings (list): List of offsprings returned as a list of net indices

		"""


		total_choices = len(index_rank)
		offsprings = []
		for i in range(num_offsprings):
			winner = np.min(np.random.randint(total_choices, size=tournament_size))
			offsprings.append(index_rank[winner])

		offsprings = list(set(offsprings))  # Find unique offsprings
		if len(offsprings) % 2 != 0:  # Number of offsprings should be even
			offsprings.append(index_rank[winner])
		return offsprings

	def create_next_gen(self, centroid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		# The scores are initialized
        scores = np.zeros(self.pop_size)

		weights = self.matrix.generate_weights(centroid, self.pop_size)
		#useless here ?
		"""
		for i in range(self.pop_size):
            # Evaluate each individual
            self.set_params(self.train_policy, weights[i])
            episode_rewards, episode_lengths = evaluate_policy(
                self.train_policy,
                self.env,
                n_eval_episodes=self.n_eval_episodes,
                return_episode_rewards=True,
            )

            scores[i] = np.mean(episode_rewards)

            # Save best params
            self.update_best_policy(scores[i], weights[i])

            # Mimic Monitor Wrapper
            infos = [
                {"episode": {"r": episode_reward, "l": episode_length}}
                for episode_reward, episode_length in zip(episode_rewards, episode_lengths)
            ]

            self._update_info_buffer(infos)

            # Update the time steps and episode counters
            self.num_timesteps += sum(episode_lengths)
            self._episode_num += len(episode_lengths)

            # Print the scores of the individual
            if self.verbose > 0:
                print(f"Indiv: {i + 1} score {scores[i]:.2f}")

        # Log for monitoring
        self.log_scores(scores)
        return weights, scores
		"""

	def epoch(self, gen, pop, fitness_evals, migration):
		self.gen+= 1; num_elitists = int(self.args.elite_fraction * len(fitness_evals))
		if num_elitists < 2: num_elitists = 2

		# Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
		index_rank = self.list_argsort(fitness_evals); index_rank.reverse()
		elitist_index = index_rank[:num_elitists]  # Elitist indexes safeguard
'''	
		#Sigaud Version 

		# Keep only best individuals to compute the new centroid
        elites_idxs = scores.argsort()[-self.elites_nb :]
        elites_weights = weights[elites_idxs]
'''
