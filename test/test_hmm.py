import pytest
from hmm import HiddenMarkovModel
import numpy as np


def test_mini_weather():
    """
    Create an instance of your HMM class using the "mini_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "mini_weather_sequences.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm is correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    # Load data
    mini_hmm = np.load('./data/mini_weather_hmm.npz')
    mini_input = np.load('./data/mini_weather_sequences.npz')
    
    # Create HMM instance
    hmm = HiddenMarkovModel(
        observation_states=mini_hmm['observation_states'],
        hidden_states=mini_hmm['hidden_states'],
        prior_p=mini_hmm['prior_p'],
        transition_p=mini_hmm['transition_p'],
        emission_p=mini_hmm['emission_p']
    )
    
    # Test forward algorithm
    observation_sequence = mini_input['observation_state_sequence']
    forward_prob = hmm.forward(observation_sequence)
    
    # Forward probability should be positive and less than 1
    assert forward_prob > 0, "Forward probability should be positive"
    assert forward_prob <= 1, "Forward probability should be <= 1"
    print(f"Mini Weather - Forward Probability: {forward_prob}")
    
    # Test Viterbi algorithm
    best_hidden_sequence = hmm.viterbi(observation_sequence)
    expected_sequence = mini_input['best_hidden_state_sequence']
    
    # Check output format and correctness
    assert len(best_hidden_sequence) == len(expected_sequence), "Sequence length mismatch"
    assert all(state in hmm.hidden_states for state in best_hidden_sequence), "Invalid hidden states in output"
    assert list(best_hidden_sequence) == list(expected_sequence), "Viterbi sequence does not match expected"
    print(f"Mini Weather - Viterbi Output: {best_hidden_sequence}")
    print(f"Mini Weather - Expected:       {list(expected_sequence)}")
    
    # EDGE CASE 1: Single observation
    single_obs = np.array(['sunny'])
    single_result = hmm.viterbi(single_obs)
    assert len(single_result) == 1, "Single observation should return single state"
    assert single_result[0] in hmm.hidden_states, "Single observation output should be valid"
    print(f"Edge Case 1 (Single observation) - Result: {single_result}")
    
    # EDGE CASE 2: All same observations (test with rainy sequence)
    same_obs = np.array(['rainy', 'rainy', 'rainy', 'rainy'])
    same_result = hmm.viterbi(same_obs)
    assert len(same_result) == 4, "All same observations should return correct length"
    assert all(state in hmm.hidden_states for state in same_result), "Output should contain valid states"
    print(f"Edge Case 2 (All same observations) - Result: {same_result}")


def test_full_weather():
    """
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_sequences.npz" file
        
    Ensure that the output of your Viterbi algorithm is correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 
    """
    
    # Load data
    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')
    
    # Create HMM instance
    hmm = HiddenMarkovModel(
        observation_states=full_hmm['observation_states'],
        hidden_states=full_hmm['hidden_states'],
        prior_p=full_hmm['prior_p'],
        transition_p=full_hmm['transition_p'],
        emission_p=full_hmm['emission_p']
    )
    
    # Test forward algorithm
    observation_sequence = full_input['observation_state_sequence']
    forward_prob = hmm.forward(observation_sequence)
    
    # Forward probability should be positive and less than 1
    assert forward_prob > 0, "Forward probability should be positive"
    assert forward_prob <= 1, "Forward probability should be <= 1"
    print(f"Full Weather - Forward Probability: {forward_prob}")
    
    # Test Viterbi algorithm
    best_hidden_sequence = hmm.viterbi(observation_sequence)
    expected_sequence = full_input['best_hidden_state_sequence']
    
    # Check output format and correctness
    assert len(best_hidden_sequence) == len(expected_sequence), "Sequence length mismatch"
    assert all(state in hmm.hidden_states for state in best_hidden_sequence), "Invalid hidden states in output"
    assert list(best_hidden_sequence) == list(expected_sequence), "Viterbi sequence does not match expected"
    print(f"Full Weather - Viterbi Output matches expected sequence")
    
    # EDGE CASE 3: Very short sequence for full model
    short_seq = np.array(['cloudy', 'rainy'])
    short_result = hmm.viterbi(short_seq)
    assert len(short_result) == 2, "Short sequence should return correct length"
    assert all(state in hmm.hidden_states for state in short_result), "Output should contain valid states"
    print(f"Edge Case 3 (Short sequence on full model) - Result: {short_result}")
    
    # EDGE CASE 4: Maximum likelihood observation (most common in first state)
    # Create a sequence of observations likely to come from the most probable state
    likely_seq = np.array(['sunny', 'sunny', 'sunny'])
    likely_result = hmm.viterbi(likely_seq)
    assert len(likely_result) == 3, "Sequence should return correct length"
    assert all(state in hmm.hidden_states for state in likely_result), "Output should contain valid states"
    print(f"Edge Case 4 (Likely observations) - Result: {likely_result}")
