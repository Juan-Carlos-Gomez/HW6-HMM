import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        This function runs the forward algorithm on an input sequence of observation states.
        
        The forward algorithm computes the probability of an observation sequence given the HMM model.
        It uses dynamic programming to efficiently compute P(observations | model).

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        n_observations = len(input_observation_states)
        n_hidden_states = len(self.hidden_states)
        
        # Forward probability table: forward_table[t][i] = probability of being in state i at time t
        forward_table = np.zeros((n_observations, n_hidden_states))
        
        # Get the index of the first observation
        first_obs_idx = self.observation_states_dict[input_observation_states[0]]
        
        # Step 2. Calculate probabilities
        # Initialize: forward_table[0][i] = prior_p[i] * emission_p[i][first_obs]
        forward_table[0] = self.prior_p * self.emission_p[:, first_obs_idx]
        
        # Forward pass: compute forward probabilities for each time step
        for t in range(1, n_observations):
            obs_idx = self.observation_states_dict[input_observation_states[t]]
            
            for j in range(n_hidden_states):
                # Sum over all previous states i: forward_table[t-1][i] * transition_p[i][j]
                forward_table[t][j] = np.sum(forward_table[t-1] * self.transition_p[:, j]) * self.emission_p[j, obs_idx]
        
        # Step 3. Return final probability (sum of all states at final time step)
        return np.sum(forward_table[-1]) 
        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        This function runs the viterbi algorithm on an input sequence of observation states.
        
        The Viterbi algorithm finds the most likely sequence of hidden states given the observation sequence.
        It uses dynamic programming with log probabilities to avoid numerical underflow.

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        n_observations = len(decode_observation_states)
        n_hidden_states = len(self.hidden_states)
        
        # Store probabilities of hidden state at each step (log scale to avoid underflow)
        viterbi_table = np.zeros((n_observations, n_hidden_states))
        # Store best previous state for traceback
        backtrace_table = np.zeros((n_observations, n_hidden_states), dtype=int)
        
        # Get the index of the first observation
        first_obs_idx = self.observation_states_dict[decode_observation_states[0]]
        
        # Step 2. Calculate Probabilities
        # Initialize: viterbi_table[0][i] = log(prior_p[i] * emission_p[i][first_obs])
        viterbi_table[0] = np.log(self.prior_p + 1e-10) + np.log(self.emission_p[:, first_obs_idx] + 1e-10)
        
        # Forward pass: compute Viterbi probabilities for each time step
        for t in range(1, n_observations):
            obs_idx = self.observation_states_dict[decode_observation_states[t]]
            
            for j in range(n_hidden_states):
                # Find the best previous state: max_i(viterbi_table[t-1][i] + log(transition_p[i][j]))
                transition_scores = viterbi_table[t-1] + np.log(self.transition_p[:, j] + 1e-10)
                backtrace_table[t][j] = np.argmax(transition_scores)
                
                # Update Viterbi probability
                viterbi_table[t][j] = np.max(transition_scores) + np.log(self.emission_p[j, obs_idx] + 1e-10)
        
        # Step 3. Traceback
        best_hidden_state_sequence = []
        
        # Find the state with highest probability at final time step
        current_state = np.argmax(viterbi_table[-1])
        best_hidden_state_sequence.append(self.hidden_states_dict[current_state])
        
        # Trace back through the sequence
        for t in range(n_observations - 1, 0, -1):
            current_state = backtrace_table[t][current_state]
            best_hidden_state_sequence.append(self.hidden_states_dict[current_state])
        
        # Reverse to get the correct order
        best_hidden_state_sequence.reverse()
        
        # Step 4. Return best hidden state sequence
        return best_hidden_state_sequence 
        