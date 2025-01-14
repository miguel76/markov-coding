from hmmlearn.hmm import CategoricalHMM
import numpy as np

def new_hmm(
    start_prob: np.ndarray = None,
    trans_prob: np.ndarray = None,
    emission_prob: np.ndarray = None,
    num_symbols : int = None,
    num_states : int = None,
    last_is_termination_state: bool = True,
    random_fill: bool = False,
    random_generator: np.random.Generator = None
) -> CategoricalHMM:
    num_states = (
        num_states
        or (start_prob and start_prob.shape)
        or (trans_prob and (trans_prob.shape[0] + last_is_termination_state))
        or (emission_prob and (emission_prob.shape[0] + last_is_termination_state))
        or 1 + last_is_termination_state
    )
    num_internal_states = num_states - last_is_termination_state
    num_symbols = (
        num_symbols
        or (emission_prob and emission_prob.shape[1])
        or 2
    )
    if random_fill:
        random_generator = random_generator or np.random.default_rng()
        
    def fill(shape: tuple[int,int]) -> np.ndarray:
        if random_fill:
            random_matrix = random_generator.random(shape)
            return random_matrix / np.sum(random_matrix, axis=1)
        else:
            return np.full(shape, 1.0 / shape[1])
        
    start_prob = start_prob or fill((1,num_states))[0]
    trans_prob = trans_prob or fill((num_internal_states, num_states))
    emission_prob = emission_prob or fill((num_internal_states, num_symbols))
    model = CategoricalHMM(
        n_components=num_states,
        n_features=num_symbols + last_is_termination_state
    )
    model.startprob_ = start_prob
    model.transmat_ = (
        np.vstack([trans_prob, np.eye(N=1, M=num_states, k=num_states - 1)])
        if last_is_termination_state
        else trans_prob
    )
    model.emissionprob_ = (
        np.vstack([
            np.hstack([emission_prob, np.zeros((1,num_internal_states))]),
            np.eye(N=1, M=num_symbols + 1, k=num_symbols)
        ])
        if last_is_termination_state
        else emission_prob
    )
    return model

def generate_until_terminates(hmm: CategoricalHMM) -> tuple[np.ndarray, np.ndarray]:
    termination_state = hmm.n_components - 1
    curr_state = None
    outputs = []
    _, new_states = hmm.sample(n_samples=1, currstate=None)
    curr_state = new_states[0]
    states = [curr_state]
    while (curr_state != termination_state):
        new_outputs, new_states = hmm.sample(n_samples=2, currstate=curr_state)
        curr_state = new_states[1]
        states.append(curr_state)
        outputs.append(new_outputs[0][0])
    return np.array(outputs, dtype=int), np.array(states, dtype=int)

def prob_distr_entropy(distr: np.ndarray) -> np.ndarray:
    return - np.sum(np.nan_to_num(distr * np.log(distr)), axis = 1)

def total_entropy(
    hmm: CategoricalHMM,
    last_is_termination_state: bool = True
) -> float:
    if last_is_termination_state:
        inner_states_trans = hmm.transmat_[np.s_[:-1,:-1]]
        a = inner_states_trans - np.eye(N=inner_states_trans.shape[0]),
        b = - prob_distr_entropy(hmm.emissionprob_[np.s_[:-1]]) - prob_distr_entropy(hmm.transmat_[np.s_[:-1]])
    else:
        a = hmm.transmat_ - np.eye(N=hmm.transmat_.shape[0])
        b = - prob_distr_entropy(hmm.emissionprob_) - prob_distr_entropy(hmm.transmat_)
    try :
        state_entropies = np.linalg.solve(a=a, b=b)
    except np.linalg.LinAlgError:
        state_entropies, _, _, _ = np.linalg.lstsq(a=a, b=b)
        state_entropies = state_entropies - min(state_entropies)
    if last_is_termination_state:
        state_entropies = np.append(state_entropies, [0])
    return (hmm.startprob_ * state_entropies + prob_distr_entropy(hmm.startprob_.reshape(1,-1)))[0]

def expected_length(hmm: CategoricalHMM):
    inner_states_trans = hmm.transmat_[np.s_[:-1,:-1]]
    state_expected_lengths = np.linalg.solve(
        a = inner_states_trans - np.eye(N=inner_states_trans.shape[0]),
        b = - hmm.transmat_[np.s_[:-1]].sum(axis=1)
    )
    return (hmm.startprob_[np.s_[:-1]] * state_expected_lengths)[0]

def entropy_by_symbol(hmm: CategoricalHMM):
    return total_entropy(hmm) / expected_length(hmm)