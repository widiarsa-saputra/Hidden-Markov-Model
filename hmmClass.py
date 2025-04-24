from collections import defaultdict, Counter
import math, json, time, os, copy

class HmmTagger:
    def __init__(self, observation_data=None, label_data=None):
        """
        observation_data and label_data contains 2D array 

        self.states is all the target label that consist in label_data
        self.emission is all the observe emission that consist in observation_data

        A is the transition states probability, every transition probability from state i and state j is counted by dividing the number of transition 
        from state i to state j by the sum of all the transitions occuring in state i

        formula => A[i][j] = number_of_transitions_ij / sum_all_transitions_i
        example:
        {
            'state1' : {
                'state1' : 1.0,
                'state2' : 0.3
            },
            'state2 : {
                'state1' : 0.0,
                'state2' : 0.7
            }
        }

        B is the emission states probability, every emission probability from emission o that occur given state s is counted by dividing the number of
        emission o that occur in state s by the sum of all emission that occur in state s
        
        formula => B[s][o] = number_of_emission_so / sum_all_emission_s
        example:
        {
            'state1' : {
                'obs1' : 1.0,
                'obs2' : 0.3
            },
            'state2 : {
                'obs1' : 0.0,
                'obs2' : 0.7
            }
        }

        pi is the starting states probability, every starting states probability is counted by dividing the emount of sequence that started from state s 
        by the total amount of sequence that consist in the dataset

        formula => pi[s] = sum_start_s / total_seq
        example:
        {
            'state1' : 0.3,
            'state2' : 0.7        
        }

        T is the total amount of observe emission data
        S is the total amount of observe data 
        N is the total amount of the states
        """
        self.observation_data = observation_data
        self.label_data = label_data
        
        if self.observation_data is not None and self.label_data is not None:
            self.states = list(set([s for seq in self.label_data for s in seq]))
            self.emission = list(set([s.lower() for seq in self.observation_data for s in seq])) 

            transition_matrix = defaultdict(Counter) #capture the amount of (number_of_transitions_ij) from state i to j
            emission_matrix = defaultdict(Counter) #capture the amount of (number_of_emission_so) from observation o given state s
            pi_matrix = Counter() #capture the amount of (sum_start_s) from state s

            for x, y in zip(self.observation_data, self.label_data):
                pi_matrix[y[0]] += 1 #key y[0] in pi_matrix dictionary got value addition by 1 (library: collections(Counter))

                for t in range(len(y)):
                    emission_matrix[y[t]][x[t].lower()] += 1 #key [y[t]][x[t].lower()] in emission matrix dictionary got value addition by 1 (library: collections(Counter))
                    if t > 0:
                        transition_matrix[y[t-1]][y[t]] += 1 #key [y[t-1]][y[t]] in transition matrix dictionary got value addition by 1 (library: collections(Counter))

            self.A = {i: {j: c / sum(transition_matrix[i].values()) for j, c in transition_matrix[i].items()} for i in transition_matrix}
            self.B = {s: {o: c / sum(emission_matrix[s].values()) for o, c in emission_matrix[s].items()} for s in emission_matrix}
            self.pi = {s: c / sum(pi_matrix.values()) for s, c in pi_matrix.items()}

            self.T = sum(len(seq) for seq in self.observation_data) / len(self.observation_data)
            self.S = len(self.observation_data)
            self.N = len(self.states)
    
    def forward_backward(self, observation_sample, A=None, B=None, pi=None):
        """
        Forward and Backward is calculates the total probability of a sequence of observations by summing all possible state trajectories.
        P(O | A, B, pi) => probability of the observation sequence given A, B, and pi

        Forward calculates the probability from previous forward probability given previous state(i) into the time t emission, then immedietely going to 
        current state(i) transition and observation
        
        formula => alpha[t][j] = sum(
            alpha[t-1][i] * A[i][j] for i in N
        ) * B[j][observation[t]]

        Backward calculates the probability from future backward probability given next state(j) back into the time t emission, then immedietely going to 
        current state(i) transition and next observation
        
        formula => beta[t][i] = sum(
            beta[t+1][j] * A[i][j] * B[j][observation[t+1]] for j in N
        )

        N = all unique state available in dataset
        1 <= i, j <= N
        """
        
        #calling the global variabel if param A, B, and pi wasn't given (initial train)
        if A is None:
            A = copy.deepcopy(self.A) #deep copy the global parameter (library: copy)
        if B is None:
            B = copy.deepcopy(self.B) #deep copy the global parameter (library: copy)
        if pi is None:
            pi = copy.deepcopy(self.pi) #deep copy the global parameter (library: copy)

        T = len(observation_sample) #length of the observation sequence
        alpha = [{} for _ in range(T)] #initialization alpha following the length of the sequence
        beta = [{} for _ in range(T)] #initialization beta following the length of the sequence

        # Inisialisasi forward dan backward
        for label in self.states: #1 <= i, j <= N
            obs0 = observation_sample[0].lower()
            alpha[0][label] = self.pi.get(label, 1e-10) * self.B[label].get(obs0, 1e-10)
            beta[T-1][label] = 1.0  # β_T(i) = 1

        # Forward formula
        for t in range(1, T):
            obs_t = observation_sample[t].lower()
            for i in self.states: #1 <= i, j <= N
                alpha[t][i] = sum(
                    alpha[t-1][j] * self.A[j].get(i, 1e-10)
                    for j in self.states
                ) * self.B[i].get(obs_t, 1e-10)

        # Backward formula
        for t in range(T - 2, -1, -1):
            obs_t1 = observation_sample[t + 1].lower()
            for i in self.states:
                beta[t][i] = sum(
                    self.A[i].get(j, 1e-10) *
                    self.B[j].get(obs_t1, 1e-10) *
                    beta[t + 1][j]
                    for j in self.states
                )

        return alpha, beta


    def compute_gamma(self, sampel_observation, A=None, B=None, pi=None):
        """
        gamma calculates the probability of each emission individually by considering the alpha and beta values.

        formula => gamma[t][s] = alpha[t][s] * beta[t][s] / sum(
            alpha[t][j] * beta[t][j] for j in N
        )

        N = all unique state available in dataset
        1 <= i, j <= N
        """

        #calling the global variabel if param A, B, and pi wasn't given (initial train)
        if A is None:
            A = copy.deepcopy(self.A) #deep copy the global parameter (library: copy)
        if B is None:
            B = copy.deepcopy(self.B) #deep copy the global parameter (library: copy)
        if pi is None:
            pi = copy.deepcopy(self.pi) #deep copy the global parameter (library: copy)
            
        alpha, beta = self.forward_backward(sampel_observation, A, B, pi) #call the forward and backward representation from current sequence observation
        T = len(alpha) #length of the observation sequence
        gamma = [{} for _ in range(T)] #initialization gamma following the length of the sequence

        for t in range(T):
            denominator = sum(alpha[t][label] * beta[t][label] for label in alpha[t])
            for s in alpha[t]:
                numerator = alpha[t][s] * beta[t][s]
                count_gamma = numerator / denominator if denominator > 0 else 0.0
                gamma[t][s] = count_gamma
        return gamma, alpha

    def compute_xi(self, sampel_observation, A=None, B=None, pi=None):
        """
        xi calculates the probability of each transition given the observation for each timestep

        formula => xi[t][i][j] = alpha[t][i] * A[i][j] * B[j][observation[t+1]] * beta[t+1][j] / sum(
            sum(
                alpha[t][i] * A[i][j] * B[j][observation[t+1]] * beta[t+1][j] for j in N
            ) for i in N
        )

        N = all unique state available in dataset
        1 <= i, j <= N
        """

        #calling the global variabel if param A, B, and pi wasn't given (initial train)
        if A is None:
            A = copy.deepcopy(self.A) #deep copy the global parameter (library: copy)
        if B is None:
            B = copy.deepcopy(self.B) #deep copy the global parameter (library: copy)
        if pi is None:
            pi = copy.deepcopy(self.pi) #deep copy the global parameter (library: copy)

        alpha, beta = self.forward_backward(sampel_observation, A, B, pi) #call the forward and backward representation from current sequence observation
        T = len(sampel_observation) #length of the observation sequence
        xi = [] #initialization xi
        
        for t in range(T-1):
            obs_t = (sampel_observation[t+1]).lower()
            denominator = sum(
                    alpha[t][i] * 
                    self.A[i].get(j, 1e-10) * 
                    self.B[j].get(obs_t, 1e-10) *
                    beta[t+1][j]
                    for i in self.states for j in self.states)
            
            xi_t = {
                i: {
                    j:alpha[t][i] * self.A[i].get(j, 1e-10) * self.B[j].get(obs_t, 1e-10) * beta[t+1][j] / denominator if denominator > 0 else 0
                    for j in self.states
                } 
                for i in self.states
                }
            xi.append(xi_t)
        return xi
    
    def save_parameters(self, filepath="hmm_params.json", A=None, B=None, pi=None):
        """
        this function is used for saving the parameter model into .json format
        """

        if A is None:
            A = copy.deepcopy(self.A) #deep copy the global parameter (library: copy)
        if B is None:
            B = copy.deepcopy(self.B) #deep copy the global parameter (library: copy)
        if pi is None:
            pi = copy.deepcopy(self.pi) #deep copy the global parameter (library: copy)
        os.makedirs("model", exist_ok=True)
        data = {
            "A" : {k: dict(v) for k, v in self.A.items()},
            "B" : {k: dict(v) for k, v in self.B.items()},
            "pi" : dict(self.pi)
        }
        with open(f"model/{filepath}", "w") as f:
            json.dump(data, f, indent=4)
            print(f"Saved model in model/{filepath}")
    
    def load_parameters(self, filepath="hmm_params.json"):
        """
        this function is used for load the lambda parameter (A, B, pi) that saved from the .json model
        """
        with open(filepath, "r") as f:
            data = json.load(f) # (library: json)
            self.A = {k: defaultdict(float, v) for k, v in data["A"].items()} #(library: collections)
            self.B = {k: defaultdict(float, v) for k, v in data["B"].items()} #(library: collections)
            self.pi = defaultdict(float, data["pi"])
    
    def train(self, tol=1e-4, epoch=10, verbose=True):
        """
        Training:
        
        Hidden Markov Model training is aim to getting the appropriate probability for the A, B, and pi

        update pi formula: pi[i] = gamma[0][i]

        uppdate A formula: A[i][j] = sum(
            xi[t][i][j] for t in T-1
        ) / sum(
            gamma[t][i] for t in t-1
        )

        update B formula : B[s][o] = sum(
            gamma[t][s] only for v_k observation
        ) / sum(
            gamma[t][s] for t in T
        )
        so, the numerator is sum over all the gamma probability for all emission that occur in state s

        T = length of the sequence
        v_k = emission k
        """
        if verbose:
            print(f"\n{'Jml. Data':<20} {self.S:<20}")
            print(f"{'Avg. Seq':<20} {self.T:<20.1f}")
            print(f"{'Jml. State':<20} {self.N:<20}")
            print(f"{'Estimasi Complexity':<20} {self.S*self.T*(self.N**2):<20}")
        
        start_time = time.time() #(library: time)

        # lambda initialization
        A = copy.deepcopy(self.A) 
        B = copy.deepcopy(self.B)
        pi = copy.deepcopy(self.pi)
        
        prev_log_likelihood = 0.0

        if verbose:
            print(f"\n{'epoch':<20} {'log-likelihood':<20} {'previous log-ll':<20} {'loss':<20} {'time'}")

        for l in range(epoch):
            start_epoch = time.time() #(library: time)
            gamma = [] #capture all the gamma from all sequence in dataset (2D arrays)
            xi = [] #capture all the xi from all sequence in dataset (2D arrays)
            log_likelihood = 0.0

            for sample_observation in self.observation_data: #count all gamma and xi
                gamma_n, alpha = self.compute_gamma(sample_observation, A, B, pi)
                gamma.append(gamma_n)

                xi.append(self.compute_xi(sample_observation, A, B, pi))

                # count the log-likelihood
                prob = sum(alpha[-1].values())  #total prob at time T
                if prob > 0:
                    log_likelihood += math.log(prob)
                else:
                    log_likelihood += math.log(1e-10)
            
            # count the loss
            if abs(log_likelihood - prev_log_likelihood) < tol:
                if verbose:
                    print(f"Converged at epoch {l+1}")
                break
            if verbose==True:
                end_epoch = time.time()
                elapsed_time = end_epoch - start_epoch
                print(f"{l+1:<20} {log_likelihood:<20.4f} {prev_log_likelihood:<20.4f} {abs(log_likelihood - prev_log_likelihood):<20.4f} {elapsed_time:.2f} detik")
            prev_log_likelihood = log_likelihood


            # update parameters A, B, pi
            emission_counts = defaultdict(lambda: defaultdict(float)) #capturing the total amounts of each emission occur in some state (library: collections)
            state_totals = defaultdict(float) #capturing the total amounts each state occur in dataset

            for n, sample_observation in enumerate(self.observation_data): 
                for t, obs_t in enumerate(sample_observation):
                    obs = obs_t.lower()
                    gamma_nt = gamma[n][t]
                    for state, val in gamma_nt.items():
                        emission_counts[state][obs] += val #total emission counts for v_k observation
                        state_totals[state] += val # state total
                
            for i in self.states: #update pi
                pi[i] = sum(gamma[n][0][i] for n in range(len(gamma))) / len(gamma) #sum over all possible first state in every sequence than divide by the number of the sequence available
            
            for i in self.states: 
                for j in self.states:
                    numerator = sum(
                        xi[n][t][i][j] for n in range(len(xi)) for t in range(len(xi[n]))
                    )
                    denominator = sum(
                        gamma[n][t][i] for n in range(len(xi)) for t in range(len(xi[n]))
                    )
                    A[i][j] = numerator / denominator if denominator > 0 else 0 #update A

                    for obs in emission_counts[j]:
                        B[j][obs] = emission_counts[j][obs] / state_totals[j] if state_totals[j] > 0 else 0.0 #update B

        end_time = time.time() #(library: time)
        elapsed_time = end_time - start_time

        if verbose:
            print(f"\nTraining completed in {elapsed_time:.2f} second")
        return A, B, pi

    def predict(self, observation, A=None, B=None, pi=None):
        """
        Viterbi Algorithm

        initialization:
            delta[1][i] = pi[i]B[i][obs[1]]
            psy[0][i] = 0
        induction:
            delta[t][j] = max(delta[t-1][i] * A[i][j] for i in N) * B[j][obs[t]]
            psy[t][j] = argmax(delta[t-1][i] * A[i][j])
        
        Termination:
            P* = max(delta[T][i] for i in N)
            q*[T] = argmax(detla[T][i] for i in N)
            q*[t] = psy[t+1][q*[t+1]]
        
        N = all unique state available in dataset
        1 <= i <= N
        2 <= t <= T
        """
        #calling the global variabel if param A, B, and pi wasn't given (initial train)
        if A is None:
            A = copy.deepcopy(self.A) #deep copy the global parameter (library: copy)
        if B is None:
            B = copy.deepcopy(self.B) #deep copy the global parameter (library: copy)
        if pi is None:
            pi = copy.deepcopy(self.pi) #deep copy the global parameter (library: copy)

        self.states = [v for v, _ in A.items()] #define the states using the A parameter

        T = len(observation)
        delta = [{} for _ in range(T)] #define delta
        psy = [{} for _ in range(T)] #define psy

        # initialization viterbi step
        for label in self.states:
            obs0 = observation[0].lower()
            if label=="O": #normalization the B probability for "O" state in NER problem
                delta[0][label] = pi.get(label, 1e-10)*B[label].get(obs0, 1)    
            else:
                delta[0][label] = pi.get(label, 1e-10)*B[label].get(obs0, 1e-10)
            psy[0][label] = 0
        
        # Induction viterbi step
        for t in range(1, T):
            obst = observation[t].lower()
            for j in self.states:
                probs = {i : delta[t-1][i] * A[i].get(j, 1e-10) for i in self.states} #(delta[T][i] for i in N)
                if j=="O": #normalization the B probability for "O" state in NER problem
                    delta[t][j] = max(probs.values()) * B[j].get(obst, 1) #find the max probability, then multiply with B 
                else:
                    delta[t][j] = max(probs.values()) * B[j].get(obst, 1e-10)
                psy[t][j] = max(probs, key=probs.get) #save the the state that gives the highest probability (i.e. "I-PER" : "O" -> means in current observation, state I-PER has big chance to correlated with O)
        
        # Termination
        P = max(delta[T-1], key=delta[T-1].get) #get the state that has highest probability of the last sequence(T)
        
        # Backtracking
        best_path = [P] #the last state of sequence is P
        for t in range(T-1, 0, -1):
            best_state = psy[t][best_path[-1]] #q*[t] = psy[t+1][q*[t+1]] it could be same as q*[t-1] = psy[t][q*[t]], q*[t] is same as the last state that is obtained for the current backtrace
            best_path.append(best_state)
        best_path.reverse() #reverse, because current best_path is started from behind
        
        return best_path
    
    def accuracy(self, x, y, verbose=True, A=None, B=None, pi=None):
        """
        Accuracy = true label recognized /  all label sequence
        """
        #calling the global variabel if param A, B, and pi wasn't given (initial train)
        if A is None:
            A = copy.deepcopy(self.A) #deep copy the global parameter (library: copy)
        if B is None:
            B = copy.deepcopy(self.B) #deep copy the global parameter (library: copy)
        if pi is None:
            pi = copy.deepcopy(self.pi) #deep copy the global parameter (library: copy)

        states = list(set([s for seq in y for s in seq]))
        if verbose:
            print(f"\t{'label':<10} | {'accuracy'}\n")
        tp_all = 0
        total_all = sum([len(seq) for seq in y])
        
        for s in states:
            tp = 0
            total = 0
            for i in range(len(x)):
                y_pred = self.predict(x[i], A, B, pi)
                for j in range(len(y_pred)):
                    if y[i][j] == s:
                        if y_pred[j] == y[i][j]:
                            tp += 1
                        total += 1
            if verbose:
                print(f"\t{s:<10} | {tp/total:.2f}")
        
        for i in range(len(x)):
            y_pred = self.predict(x[i])
            for j in range(len(y_pred)):
                if y_pred[j] == y[i][j]:
                    tp_all += 1
        if verbose:
            print(f"\n{'Accuracy total':>10} | {tp_all/total_all:.2f}")
        return tp_all/total_all
        
    
    def confusion_matriks(self, x, y, neg):
        tp, tn, fp, fn = 0, 0, 0, 0

        states = list(set([s for seq in y for s in seq]))
        if neg not in states:
            print(f"{neg} is not label in Dataset")
            return 0
        
        for i in range(len(x)):
            y_pred = self.predict(x[i])
            for j in range(len(y_pred)):
                # positives part
                if y[i][j] != neg:
                    if y_pred[j]==y[i][j]:
                        tp += 1
                    else:
                        fn += 1
                # negatives part
                else:
                    if y_pred[j] == y[i][j]:
                        tn += 1
                    else:
                        fp += 1
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        accuracy = (tn+tp) / (tn+tp+fn+fp)
        f1score = 2*precision*recall/(precision+recall)
        print(f"{'recall':<10} {'precision':<10} {'accuracy':<10} {'f1score':<10}")
        print(f"{recall:<10.4f} {precision:<10.4f} {accuracy:<10.4f} {f1score:<10.4f}")
    
    def grid_search(self, x_test, y_test, epoch=[5], tol=[1e-4], verbose=False, best_param_name="best_model", call_back=100.0):
        print_result = True
        max_accuracy = -float('inf')
        max_pair = None
        eval = defaultdict(lambda: defaultdict(float))
        A_list = defaultdict(lambda: defaultdict(dict))
        B_list = defaultdict(lambda: defaultdict(dict))
        pi_list = defaultdict(lambda: defaultdict(dict))
        for e in epoch:
            for t in tol:
                print(f"Training parameter epoch: {e} with Tolerance: {t}", end=" ")
                A, B, pi = self.train(tol=t, epoch=e, verbose=verbose)
                A_list[e][t] = A
                B_list[e][t] = B
                pi_list[e][t] = pi
                eval[e][t] = self.accuracy(
                    x=x_test,
                    y=y_test,
                    A=A,
                    B=B,
                    pi=pi,
                    verbose=verbose
                )
                if eval[e][t] >= float(call_back):
                    print_result = False
                    print(f"Call back: {call_back}")
                    print(f"Epoch{e} and tolerance {t} gives highest accuracy with {eval[e][t]}")
                    break
                print(f"accuracy: {eval[e][t]}")

                if eval[e][t] > max_accuracy:
                    max_accuracy = eval[e][t]
                    max_pair = [e, t]
        print("\n")
        if print_result:
            print(f"{'tol/epoch':>10}", end=" ")
            for t in tol:
                print(f"{t:>10}", end="")
            print("\n")
            for e in epoch:
                print(f"{e:>10}", end=" ")
                for t in tol:
                    print(f"{eval[e][t]:>10.4f}", end=" ")
                print("\n")
            
            print(f"Epoch {max_pair[0]}, Tolerance {max_pair[1]} gives highest accuracy with {max_accuracy}")
        A = A_list[max_pair[0]][max_pair[1]]
        B = B_list[max_pair[0]][max_pair[1]]
        pi = pi_list[max_pair[0]][max_pair[1]]
        self.save_parameters(f"{best_param_name}.json", A, B, pi)