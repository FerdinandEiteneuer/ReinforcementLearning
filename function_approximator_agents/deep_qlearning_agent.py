import numpy as np
import tensorflow as tf
import gym

from .neural_network_agent import NeuralNetworkAgent
import utils

@utils.export
class DeepQLearningAgent(NeuralNetworkAgent):

    def __init__(
            self,
            env,
            Q=None,
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
            self_play=False,
            starting_learning_rate=2*10**(-5),
            success_condition=None,
        ):

        super().__init__(
            env=env,
            policy=policy,
            epsilon_scheduler=epsilon_scheduler,
            learning_rate_scheduler=learning_rate_scheduler,
            gamma=gamma,
            self_play=self_play,
            save_model_dir=save_model_dir,
            size_memory=size_memory,
            experience_replay=experience_replay,
            success_condition=success_condition,
        )

        if Q is not None:
            self.Q = Q
        else:
            self.Q = utils.create_sequential_dense_net(
                input_shape=self.Q_input_shape,
                n_outputs=self.nb_actions,
                hidden_layers=7,
                neurons_per_layer=128,
                p_dropout=0.1,
                lambda_regularization=10**(-4),
                hidden_activation_function='relu',
                final_activation_function='relu'
            )

        self.starting_learning_rate = starting_learning_rate

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.starting_learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            #amsgrad=True
        )

        #self.loss = 'mean_absolute_error'
        self.loss = 'mean_squared_error'
        #self.loss = tf.keras.losses.Huber()

        self.callbacks = []
        if self.learning_rate_scheduler:
            scheduler = tf.keras.callbacks.LearningRateScheduler(
                self.learning_rate_scheduler
            )
            self.callbacks.append(scheduler)

        self.Q.compile(
            optimizer=optimizer,
            loss=self.loss,
        )

        # reserve space for the experienced episodes, if desired

        # here, we build a second network that will generate the predictions
        # to create the Q learning target. This is done to create more stability.
        self.fixed_target_weights = fixed_target_weights
        if fixed_target_weights:
            self.Q_fixed_weights = tf.keras.models.clone_model(self.Q)
            self.update_period_fixed_target_weights = update_period_fixed_target_weights


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
        episode_length = 0
        total_reward = 0

        if self.fixed_target_weights:
            if self.episodes % self.update_period_fixed_target_weights == 0:
                self.update_fixed_target_weights()

        # train the neural network
        if self.episodes % 10 == 0:
            fit_info = self.update_Q(episodes=4)
            if fit_info:
                history = fit_info.history
                losses.extend(history['loss'])
                if losses[-1] == 0:
                    print('loss was zero. ', fit_info)

        while not terminal:

            #if self.episodes % 5 == 0:
            #   self.env.render()
            action = self.policy(state)

            if self.self_play:
                next_state, reward, terminal, info = self.env.step(action, self.opponent_policy)
            else:
                next_state, reward, terminal, info = self.env.step(action)

            if not terminal:
                network = 'Q_fixed_weights' if self.fixed_target_weights else 'Q'
                maxQ = self.get_maxQ(next_state, network=network)
                target = reward + self.gamma * maxQ
            else:
                target = reward

            if self.experience_replay:
                qvalues = self.predict(state, network='Q')
                qvalues[action] = target

                self.memory.add(state, qvalues)

            state = next_state
            episode_length += 2
            total_reward += reward

        mean_loss = np.mean(losses) if len(losses) > 0 else None

        train_info = {
            'mean_loss': mean_loss,
            'last_reward': reward,
            'episode_length': episode_length,
            'total_reward': total_reward,
        }

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

        episode_length = 0
        total_reward = 0

        while not terminal:
            action = self.policy(state)

            # episode_memory.append((state, action, reward, terminal))

            if self.self_play:
                next_state, reward, terminal, info = self.env.step(action, self.opponent_policy)
                episode_length += 2
            else:
                next_state, reward, terminal, info = self.env.step(action)
                episode_length += 1

            total_reward += reward
            state = next_state

        info = {
            'last_reward': reward,
            'total_reward': total_reward,
            'episode_memory': episode_memory,
            'episode_length': episode_length,
        }

        return info



