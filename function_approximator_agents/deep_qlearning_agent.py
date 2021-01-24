import numpy as np
import tensorflow as tf
import gym

from .neural_network_agent import NeuralNetworkAgent
import utils


@utils.export
class DeepQLearningAgent(NeuralNetworkAgent):

    def __init__(self,
                 env,
                 save_model_dir=None,
                 batch_size=16,
                 epsilon_scheduler=utils.ConstEpsilonScheduler(0.1),
                 learning_rate_scheduler=None,
                 policy='eps_greedy',
                 experience_replay=True,
                 size_memory=1024,
                 fixed_target_weights=True,
                 update_period_fixed_target_weights=400,
                 gamma=1,
                 self_play=False):

        super().__init__(env=env,
                         policy=policy,
                         epsilon_scheduler=epsilon_scheduler,
                         learning_rate_scheduler=learning_rate_scheduler,
                         gamma=gamma,
                         self_play=self_play,
                         save_model_path=save_model_dir)

        if type(env.observation_space) is gym.spaces.discrete.Discrete:
            self.Q_input_shape = (env.observation_space.n, )
        elif type(env.observation_space) is gym.spaces.tuple.Tuple:
            self.Q_input_shape = (len(env.observation_space), )

        self.Q = utils.create_Sequential_Dense_net1(
            input_shape=self.Q_input_shape,
            n_outputs=env.action_space.n,
            layers=10,
            neurons=256,
            p_dropout=0.1,
            lambda_regularization=10**(-4),
        )

        self.starting_learning_rate = 5*10**(-6)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.starting_learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            #amsgrad=True
        )

        #self.loss = 'mean_absolute_error'
        #self.loss = 'mean_squared_error'
        self.loss = tf.keras.losses.Huber()

        self.callbacks = []
        if self.learning_rate_scheduler:
            scheduler = tf.keras.callbacks.LearningRateScheduler(self.learning_rate_scheduler)
            self.callbacks.append(scheduler)

        self.Q.compile(
            optimizer=optimizer,
            loss=self.loss,
        )

        self.episodes = 0


        self.experience_replay = experience_replay
        self.fixed_target_weights = fixed_target_weights

        # reserve space for the experienced episodes, if desired
        if experience_replay:

            self.memory = utils.NumpyArrayMemory(
                size=size_memory,
                input_shape=self.Q_input_shape[0],
                nb_actions=self.nb_actions,
                data_dir=self.save_model_path
            )

        # here, we build a second network that will generate the predictions
        # to create the Q learning target. This is done to create more stability.
        if fixed_target_weights:
            self.Q_fixed_weights = tf.keras.models.clone_model(self.Q)
            self.update_period_fixed_target_weights = update_period_fixed_target_weights

    def training_data(self):
        # TODO deprecated
        # one Q_memory row consists of [state, Qvalues(dim=nb_actions)]
        states = self.Q_memory[:, :-self.nb_actions]
        targets = self.Q_memory[:, -self.nb_actions:]

        return states, targets

    def update_Q(self, batch_size=128, episodes=2):

        if self.memory.ready():
        #f self.memory_ready():

            states, targets = self.memory.complete_training_data()
            #states, targets = self.training_data()

            fit_result = self.Q.fit(
                x=states,
                y=targets,
                batch_size=batch_size,
                epochs=episodes,
                callbacks=self.callbacks,
                verbose=False)

            return fit_result
            # self.Q.train_on_batch(states, rewards)
        else:
            return False  # dont train here

    def update_fixed_target_weights(self):
        """
        Updates the fixed weights of the prediction network Q_fixed_targets.
        """
        current_weights = self.Q.get_weights()

        # fresh optimizer, since we will be updating towards new goals. (Old opt. weights are irrelevant.)
        opt = tf.keras.optimizers.Adam(
            learning_rate=self.starting_learning_rate
        )

        self.Q_fixed_weights = tf.keras.models.clone_model(self.Q)
        self.Q_fixed_weights.set_weights(current_weights)

        self.Q_fixed_weights.compile(
            optimizer=opt,
            loss=self.loss
        )

    def train_one_episode(self):
        # initialize
        self.episodes += 1

        state = self.env.reset()
        losses = []

        fit_info = None
        terminal = False

        if self.fixed_target_weights:
            if self.episodes % self.update_period_fixed_target_weights == 0:
                self.update_fixed_target_weights()

        # train the neural network
        if self.episodes % 20 == 0:
            fit_info = self.update_Q(episodes=20)
            if fit_info:
                history = fit_info.history
                losses.extend(history['loss'])
                if losses[-1] == 0:
                    print('loss was zero. ', fit_info)

        while not terminal:

            action = self.policy(state)

            if self.self_play and self.episodes > self.memory.size:
                opponent_action = self.get_ideal_opponent_action
            else:
                opponent_action = self.get_random_action

            next_state, reward, terminal, info = self.env.step(action, opponent_action)

            if not terminal:
                maxQ = self.get_maxQ(next_state, network='Q_fixed_weights')
                target = reward + self.gamma * maxQ
            else:
                target = reward

            if self.experience_replay:

                qvalues = self.predict(state, network='Q')
                qvalues[action] = target

                self.memory.add(state, qvalues)

                #self.Q_memory[self.episodes % self.memory.size] = list(state) + list(qvalues)

            state = next_state

        mean_loss = np.mean(losses) if len(losses) > 0 else None
        train_info = {'mean_loss': mean_loss, 'last_reward': reward}

        return train_info

    def play_one_episode(self):
        """
        The agent plays one episode.
        """

        # setup
        terminal = False
        reward = None
        state = self.env.reset()
        episode_memory = []

        while not terminal:
            action = self.policy(state)

            # episode_memory.append((state, action, reward, terminal))

            next_state, reward, terminal, info = self.env.step(action, self.opponent_policy)

            state = next_state

        info['last_reward'] = reward
        info['episode_memory'] = episode_memory
        return info



