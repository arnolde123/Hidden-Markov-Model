import sys

#Loads files and returns all states and probabiltiies
def load_file(filename):
  hidden_states = []
  evidence_states = []
  
  initial_probs = {}
  transition_probs = {}
  observation_probs = {}
  
  with open(filename, 'r') as file:
    for line in file:
      parts = line.strip().split(',')
      if parts[0] == 'HIDDEN-STATES':
        hidden_states = parts[1:]
      elif parts[0] == 'EVIDENCE-STATES':
        evidence_states = parts[1:]
      elif parts[0] == 'TRANSITION-PROBABILITY':
        if parts[1] == 'START':
          initial_probs[parts[2]] = float(parts[3])
        else:
          if parts[1] not in transition_probs:
            transition_probs[parts[1]] = {}
          transition_probs[parts[1]][parts[2]] = float(parts[3])
      elif parts[0] == 'EVIDENCE-PROBABILITY':
        if parts[1] not in observation_probs:
          observation_probs[parts[1]] = {}
        observation_probs[parts[1]][parts[2]] = float(parts[3])
  
  return hidden_states, evidence_states, transition_probs, observation_probs, initial_probs

#Prints out probabilities for transitions and also observations
def print_transition_tables(hidden_states, transition_probs, observation_probs):
  print("Transition Probabilities:")
  for from_state in hidden_states:
    for to_state in hidden_states:
      print(f"P({to_state}|{from_state}) = {transition_probs[from_state][to_state]:.2f}")
  
  print("\nObservation Probabilities:")
  for state in hidden_states:
    for evidence in observation_probs:
      print(f"P({evidence}|{state}) = {observation_probs[evidence][state]:.2f}")

#First calculates and prints the initial distribution,then begins calculating each steps 
#distribution using the last. I use the following formulas:
#
def forward_algorithm(sequence, hidden_states, transition_probs, observation_probs, initial_probs):
  distribution = {}
  alphaValue = 0
  #Calculate dist at step 0 (using steps like days in Question 1)
  for state in hidden_states:
    obs_prob = observation_probs.get(sequence[0]).get(state)
    distribution[state] = initial_probs.get(state) * obs_prob
    alphaValue += distribution[state]
  
  # Normalize initial distribution and print
  for state in hidden_states:
    distribution[state] /= alphaValue

  print("Initial distribution values: " + ", ".join(f"{state}: {value:.4f}" for state, value in distribution.items()))
  print(f"Initial Alpha Value: {1 / alphaValue:.4f}")
  # Stores all distributions so i can access the last steps distribution
  all_dists = [distribution]
  
  #Calculate the dist and alpha values for each step, given that each input in the 
  #sequence string is a new step with a different observation
  for step in range(1, len(sequence)):
    new_dist = {}
    observation = sequence[step]
    alphaValue = 0  # Reset alpha
    
    # Calculate unnormalized probabilities for the dist
    for current_state in hidden_states:
      sum_alpha = 0
      for prev_state in hidden_states:
        transition_prob = transition_probs.get(prev_state).get(current_state)
        sum_alpha += all_dists[step - 1].get(prev_state) * transition_prob
      
      obs_prob = observation_probs.get(observation).get(current_state)
      new_dist[current_state] = sum_alpha * obs_prob
      alphaValue += new_dist[current_state]
    
    # Normalize the probabilities, or make it 0 if all probs are 0 
    for state in hidden_states:
      if alphaValue > 0:  
        new_dist[state] /= alphaValue
      else:
        new_dist[state] = 0
    
    #Add the dist and print
    distribution = new_dist
    all_dists.append(distribution)
    print(f"Distribution values at step {step}: " + ", ".join(f"{state}: {value:.4f}" for state, value in distribution.items()))
    print(f"Alpha Value: at step {step}: {1 / alphaValue:.4f}")

  return all_dists

def main():
  #Stores the filename and sequence of observations
  infile = sys.argv[1]
  sequence = sys.argv[2].split(',')
  
  #Call all methods
  hidden_states, evidence_states, transition_probs, observation_probs, initial_probs = load_file(infile)
  print_transition_tables(hidden_states, transition_probs, observation_probs)
  
  forward_algorithm(sequence, hidden_states, transition_probs, observation_probs, initial_probs)
  
if __name__ == "__main__":
  main()
