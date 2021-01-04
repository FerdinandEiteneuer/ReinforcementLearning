* "size_Q_memory" from (1024, 10) to 1024, i.e dont give the second dimensions, only number of datapoints
* in deep_qlearning_agent: functions train,play and _loop into neuralnetwork agent
* in neural_network_agent: analsye_maxQ just assumes states with value 0 are valid and nonzero are not valid.... use env's get_valid_actions?