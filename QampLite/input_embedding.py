import utils
from src.consts import *
import pickle
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity
import numpy as np
from scipy.optimize import minimize
import warnings
import os
import time
import cirq
from utils import read_experimental_results

# Use gradient_free = True to use COBYLA gradient-free optimization method. NOT RECOMMENDED -- MUCH SLOWER THAN GRADIENT BASED
gradient_free = False

def get_QampLite_cirq(cirq_params, num_qubits=NUM_QUBITS):
    circuit = cirq.Circuit()
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]

    # add an ry gate on all qubits.
    for i in range(num_qubits):
        circuit.append(cirq.ry(cirq_params[i])(qubits[i]))

    # add a controlled ry gate with control=qubits[0] and target = qubits[1]
    cry = cirq.ry(cirq_params[num_qubits]).controlled()
    circuit.append(cry(qubits[0], qubits[1]))

    return circuit

def subset_circuit(orig_circuit, num_qubits=NUM_QUBITS):
    """
    Given a QuantumCircuit object, return the resulting subcircuit that operates only on the first num_qubits qubits.

    :param orig_circuit: a QuantumCircuit that represents the circuit to be subsetted
    :param num_qubits: an integer representing the number of qubits desired in the subcircuit
    :return: a QuantumCircuit that represents the subset circuit, that only operates on num_qubits qubits.
    """
    # Qubits and classical bits of interest
    qubits_of_interest = range(num_qubits)
    classical_bits_of_interest = range(num_qubits)

    # Initialize new circuit
    new_circuit = QuantumCircuit(num_qubits, num_qubits)

    # Create mappings
    qubit_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(qubits_of_interest)}
    classical_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(classical_bits_of_interest)}

    # Extract and add instructions
    for instr, qargs, cargs in orig_circuit.data:
        # Get indices of qubits involved in the instruction
        qubit_indices = [orig_circuit.qubits.index(qubit) for qubit in qargs]

        # Check if all qubits are among the qubits of interest
        if all(q in qubits_of_interest for q in qubit_indices):
            # Map the qubits to the new circuit's qubits
            new_qargs = [new_circuit.qubits[qubit_mapping[orig_circuit.qubits.index(q)]] for q in qargs]

            # Map classical bits if necessary
            if cargs:
                classical_indices = [orig_circuit.clbits.index(bit) for bit in cargs]
                new_cargs = [new_circuit.clbits[classical_mapping[idx]] for idx in classical_indices]
            else:
                new_cargs = []

            # Add the instruction to the new circuit
            new_circuit.append(instr, new_qargs, new_cargs)
    return new_circuit

def get_QampLite_qiskit(params, num_qubits=NUM_QUBITS, do_transpile=True):
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))

    # add an ry gate on all qubits.
    for i in range(num_qubits):
        qc.ry(params[i], qubits[num_qubits - 1 - i])

    # add a controlled ry gate with control=qubits[-1] and target = qubits[-2]
    ry_gate = RYGate(params[num_qubits]).control(1)
    qc.append(ry_gate, [qubits[num_qubits - 1], qubits[num_qubits - 2]])

    if do_transpile:
        qc = transpile(
            qc,
            backend=ibm_backend,
            optimization_level=0,  # keep it raw, no heavy optimizations
            initial_layout=range(num_qubits),  # map logical qubits 0–8 directly to first 9 physical qubits
            seed_transpiler=42  # reproducibility
        )
        qc = subset_circuit(qc)
    return qc

def perform_simulation(qiskit_circuit, sim_noise_model):
    qiskit_circuit.save_density_matrix()
    if sim_noise_model is None:
        qiskit_simulator = ideal_qiskit_simulator
    else:
        qiskit_simulator = noisy_qiskit_simulator
    result = qiskit_simulator.run(qiskit_circuit).result()
    density_matrix = result.data()['density_matrix']
    return density_matrix

def get_fidelity(qc, image, noise_model):
    density_matrix = perform_simulation(qc, sim_noise_model=noise_model)
    targ_density_mat = DensityMatrix(image)
    return state_fidelity(density_matrix, targ_density_mat)


# --- 2. Analytical Helper Functions ---

def ry_state(theta):
    """Returns the 1-qubit state vector Ry(theta)|0>."""
    return np.array([np.cos(theta / 2),
                     np.sin(theta / 2)])


def ry_deriv_state(theta):
    """Returns the derivative d/d(theta) [Ry(theta)|0>]."""
    return 0.5 * np.array([-np.sin(theta / 2),
                           np.cos(theta / 2)])


# --- 3. Core Logic: Cost and Gradient Function ---

def get_cost_and_gradient(params, image_vec, num_qubits=NUM_QUBITS):
    """
    Calculates the loss and the analytical gradient.
    This is the function we will pass to SciPy's optimizer.
    """

    # --- 3a. Unpack Parameters ---
    # params[0...NUM_QUBITS-1] are the single-qubit Ry gates
    # params[NUM_QUBITS] is the CRy gate

    # State vectors for the first two qubits (special case)
    c0 = np.cos(params[0] / 2)
    s0 = np.sin(params[0] / 2)

    # State |phi_1> = Ry(theta_1)|0>
    phi_1 = ry_state(params[1])

    # State |phi_1+N> = Ry(theta_1 + theta_N)|0>
    # (where theta_N is params[NUM_QUBITS])
    phi_1_plus_N = ry_state(params[1] + params[num_qubits])

    # Derivatives for the first two qubits
    d_c0 = -s0 / 2
    d_s0 = c0 / 2

    # d|phi_1>/d(theta_1)
    phi_1_deriv = ry_deriv_state(params[1])

    # d|phi_1+N>/d(theta_1) and d|phi_1+N>/d(theta_N)
    # (they are the same by chain rule)
    phi_1_plus_N_deriv = ry_deriv_state(params[1] + params[num_qubits])

    # State vectors for the "tail" qubits (qubit 2 to N-1)
    phi_states = [ry_state(params[i]) for i in range(2, num_qubits)]
    phi_deriv_states = [ry_deriv_state(params[i]) for i in range(2, num_qubits)]

    # --- 3b. Calculate Full State Vector |psi> ---

    # |psi_tail> = |phi_2> @ |phi_3> @ ...
    psi_tail = 1.0
    for state in phi_states:
        psi_tail = np.kron(psi_tail, state)

    # |state_0_branch> = |phi_1> @ |psi_tail>
    state_0_branch = np.kron(phi_1, psi_tail)

    # |state_1_branch> = |phi_1+N> @ |psi_tail>
    state_1_branch = np.kron(phi_1_plus_N, psi_tail)

    # |psi> = c0 * |0> @ |state_0_branch> + s0 * |1> @ |state_1_branch>
    psi_vec = (c0 * np.kron([1, 0], state_0_branch) +
               s0 * np.kron([0, 1], state_1_branch))

    # --- 3c. Calculate Loss ---
    # L = - <image | psi>
    # np.vdot is the complex-conjugating inner product <a|b>
    # We use np.real just to be safe, though all values are real.
    loss = -np.real(np.vdot(image_vec, psi_vec))

    # --- 3d. Calculate Gradient ---
    grad = np.zeros_like(params)

    # dL/d(theta_0)
    d_psi_d_theta_0 = (d_c0 * np.kron([1, 0], state_0_branch) +
                       d_s0 * np.kron([0, 1], state_1_branch))
    grad[0] = -np.real(np.vdot(image_vec, d_psi_d_theta_0))

    # dL/d(theta_1)
    d_state_0_branch = np.kron(phi_1_deriv, psi_tail)
    d_state_1_branch = np.kron(phi_1_plus_N_deriv, psi_tail)
    d_psi_d_theta_1 = (c0 * np.kron([1, 0], d_state_0_branch) +
                       s0 * np.kron([0, 1], d_state_1_branch))
    grad[1] = -np.real(np.vdot(image_vec, d_psi_d_theta_1))

    # dL/d(theta_N) (where N is NUM_QUBITS)
    # The |0> branch has no dependency on theta_N
    d_state_1_branch_N = np.kron(phi_1_plus_N_deriv, psi_tail)
    d_psi_d_theta_N = s0 * np.kron([0, 1], d_state_1_branch_N)
    grad[num_qubits] = -np.real(np.vdot(image_vec, d_psi_d_theta_N))

    # dL/d(theta_k) for k in [2, N-1] (the tail qubits)
    for k in range(2, num_qubits):
        # We must re-calculate psi_tail_deriv for each k
        psi_tail_deriv = 1.0
        for i in range(len(phi_states)):
            # (i+2) is the true qubit index (k)
            if (i + 2) == k:
                psi_tail_deriv = np.kron(psi_tail_deriv, phi_deriv_states[i])
            else:
                psi_tail_deriv = np.kron(psi_tail_deriv, phi_states[i])

        d_state_0_k = np.kron(phi_1, psi_tail_deriv)
        d_state_1_k = np.kron(phi_1_plus_N, psi_tail_deriv)

        d_psi_d_theta_k = (c0 * np.kron([1, 0], d_state_0_k) +
                           s0 * np.kron([0, 1], d_state_1_k))

        grad[k] = -np.real(np.vdot(image_vec, d_psi_d_theta_k))

    return loss, grad


# --- 4. The Main Optimizer Function ---

def optimize(image, initial_params, num_qubits=NUM_QUBITS):
    """
    Finds the optimal parameters using analytical gradients or COBYLA gradient free method based on the boolean variable gradient-free
    """

    if gradient_free:
        cirq_simulator = cirq.Simulator()

        # evaluation of a circuit function
        def evaluate_circuit(params):
            circuit = get_QampLite_cirq(params)
            res = cirq_simulator.simulate(circuit)
            state_vector = res.final_state_vector
            return -np.abs(np.vdot(state_vector, image)) ** 2

        # if there is no warm-start provided, then set the initial parameters randomly.
        if initial_params is None:
            initial_params = tuple(
                [np.random.random() * np.pi for _ in range(num_qubits)] + [np.random.random() * np.pi / 2 - 0.25 * np.pi
                                                                           for _ in
                                                                           range(1)])
        # optimize using scipy.minimize()
        result = minimize(evaluate_circuit, initial_params,
                          bounds=[(0, np.pi) for _ in range(num_qubits)] + [(-np.pi / 4, np.pi / 4) for _ in range(1)],
                          method='COBYLA')
        return result.fun, result.x

    else:
        if initial_params is None:
            # Create a random starting point if none is provided
            # There are NUM_QUBITS + 1 parameters
            initial_params = np.random.rand(num_qubits + 1) * 2 * np.pi
        # --- This is the core optimization step ---
        # We pass our cost-and-gradient function to SciPy
        # 'jac=True' tells minimize that our function returns (loss, gradient)
        result = minimize(
            get_cost_and_gradient,
            initial_params,
            args=(image, num_qubits),
            method='L-BFGS-B',
            jac=True
        )

        # 'result.fun' is the final loss (minimized negative inner product)
        # We return the positive inner product
        opt_value = -result.fun

        # 'result.x' is the array of optimal parameters
        opt_params = result.x

    return opt_value ** 2, opt_params


def _run_single_dataset(processed_images, noise_model):
    """Helper function to run the experiment loop on one dataset."""

    # Pre-allocate arrays for speed (better than appending)
    num_images = len(processed_images)
    opt_values = np.zeros(num_images)
    opt_times = np.zeros(num_images)
    ideal_fidelities = np.zeros(num_images)
    noisy_fidelities = np.zeros(num_images)

    print(f"Starting experiment for {num_images} images...")

    opt_params = None
    for i in range(num_images):
        image = processed_images[i]

        # 1. Time and run optimization
        start_time = time.perf_counter()
        opt_value, opt_params = optimize(image, opt_params)
        end_time = time.perf_counter()

        # 3. Get Qiskit results
        qc = get_QampLite_qiskit(opt_params, do_transpile=True)
        qc_copy = qc.copy()

        ideal_fidelity = get_fidelity(qc_copy, image, noise_model=None)
        noisy_fidelity = get_fidelity(qc, image, noise_model=noise_model)
        # print("opt value", opt_value)
        # print("ideal fidelity:", ideal_fidelity)
        # print("noisy fidelity:", noisy_fidelity)
        # 4. Record all results
        opt_values[i] = opt_value
        opt_times[i] = end_time - start_time
        ideal_fidelities[i] = ideal_fidelity
        noisy_fidelities[i] = noisy_fidelity

        if (i + 1) % 100 == 0 or i == num_images - 1:
            print(f"  ... completed image {i + 1}/{num_images}")

    return opt_values, opt_times, ideal_fidelities, noisy_fidelities


def run_experiments(noise_model):
    """
    Main function to load, process, and run experiments on all datasets.
    """
    # 1. Load data using the utils function
    print("--- Loading Subsampled Data ---")
    (sub_c_img, _,
     sub_m_img, _,
     sub_f_img, _) = utils.load_subsampled_data()

    # 2. Process all three datasets
    print("\n--- Processing Datasets ---")
    processed_cifar = utils.process_images(sub_c_img)
    processed_mnist = utils.process_images(sub_m_img)
    processed_fashion = utils.process_images(sub_f_img)

    # 3. Run experiments for each dataset
    print("\n--- 1. Running CIFAR-10 Experiment ---")
    c_vals, c_times, c_ideal, c_noisy = _run_single_dataset(
        processed_cifar, noise_model
    )

    print("\n--- 2. Running MNIST-784 Experiment ---")
    m_vals, m_times, m_ideal, m_noisy = _run_single_dataset(
        processed_mnist, noise_model
    )

    print("\n--- 3. Running Fashion-MNIST Experiment ---")
    f_vals, f_times, f_ideal, f_noisy = _run_single_dataset(
        processed_fashion, noise_model
    )

    # 4. Store all results in one file
    results_path = os.path.join("../input_embedding_res", 'gradient-free=' + str(gradient_free) + '.npz')
    print(f"\n--- Saving All Results ---")
    np.savez_compressed(
        results_path,
        # CIFAR results
        cifar_opt_values=c_vals,
        cifar_opt_times=c_times,
        cifar_ideal_fidelities=c_ideal,
        cifar_noisy_fidelities=c_noisy,
        # MNIST results
        mnist_opt_values=m_vals,
        mnist_opt_times=m_times,
        mnist_ideal_fidelities=m_ideal,
        mnist_noisy_fidelities=m_noisy,
        # Fashion-MNIST results
        fashion_opt_values=f_vals,
        fashion_opt_times=f_times,
        fashion_ideal_fidelities=f_ideal,
        fashion_noisy_fidelities=f_noisy
    )
    print(f"All experiment results saved to '{results_path}'")

if __name__ == '__main__':
    with open(NOISE_MODEL_PATH, 'rb') as file:
        noise_model = pickle.load(file)
    ibm_backend = FakeBrisbane()
    ideal_qiskit_simulator = AerSimulator(method='density_matrix', noise_model=None)
    noisy_qiskit_simulator = AerSimulator(method='density_matrix', noise_model=noise_model)
    warnings.filterwarnings('ignore')


    # check QampLite circuit
    # print(get_QampLite_qiskit([0] * (NUM_QUBITS + 1), do_transpile=False))

    # Check subset_circ
    # qc = QuantumCircuit(3)
    # qubits = list(range(3))
    #
    # qc.x(qubits[0])
    # qc.x(qubits[1])
    # qc.x(qubits[2])
    # print("The entire circuit")
    # print(qc)
    # print("The subset circuit restricted to first 2 qubits")
    # print(subset_circuit(qc, 2))

    # Run the experiments and store them in either
    # experiment_results-gradient-free=True.npz or experiment_results-gradient-free=False.npz
    # based on the gradient_free boolean value
    run_experiments(noise_model=noise_model)

    # Read the experiments results
    read_experimental_results('gradient-free=' + str(gradient_free)+'.npz')