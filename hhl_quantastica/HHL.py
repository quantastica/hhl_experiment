from quantastica.qps_api import QPS

from circuit_lite import CircuitLite, SimulatorLite

from copy import deepcopy
import numpy as np
from scipy.linalg import expm

#
# Create QFT/IQFT subroutine
#
def get_qft_circuit(num_qubits, inverse=False):
    circuit = CircuitLite()
    
    if(not inverse):
        for q in range(num_qubits):
            circuit.add("h", q)

            counter = 1
            for i in range(q + 1, num_qubits):
                angle = "pi/" + str(2**counter)
                circuit.add("cu1", [q, i], angle)
                counter += 1
    else:
        for q in range(num_qubits - 1, -1, -1):
            counter = num_qubits - q - 1
            for i in range(num_qubits - 1, q, -1):
                angle = "-pi/" + str(2**counter)
                circuit.add("cu1", [q, i], angle)
                counter -= 1

            circuit.add("h", [q])
                
    return circuit

#
# Eigenvalue rotation subroutine
#
def get_x_1_circuit(clock_register_size):
    num_qubits = clock_register_size + 1
    
    circuit = CircuitLite()

    for q in range(clock_register_size):
        k = q
        if(k == 0):
            k = clock_register_size
        angle = str(k) + "*pi/" + str(clock_register_size)
        
        circuit.add("cry", [clock_register_size - q - 1, num_qubits - 1], angle)
        
    return circuit
 
#
# Make controlled unitary
#
def get_controlled_u(U):
    Z = np.zeros((len(U), len(U)), dtype=complex)
    eye = np.eye(len(U))
    return np.block([[eye, Z], [Z, U]])
    
#
# Make HHL circuit for linear system given as square matrix A and vector b
#
def get_hhl_circuit(A, b, verbose=False):
    vector_register_size = int(np.log2(len(b)))
    clock_register_size = 3 # !!!
    total_qubits = vector_register_size + clock_register_size + 1
    
    min_eigval = np.min(np.linalg.eigvals(A))
        
    circuit = CircuitLite()

    # State preparation using Quantastica tools
    job_name = "b"
    vector = b / np.linalg.norm(b)
    job_id = QPS.generator.state_preparation(vector.tolist(), endianness="little", job_name=job_name, settings = { "instruction_set": ["u3", "cx"] }, start_job=True)
    job = QPS.generator.get_job(job_id, wait=True)
    subroutine = CircuitLite()
    subroutine.from_toaster(job["output"]["circuits"][0])
    circuit.register_subroutine(job_name, subroutine)

    target_wires = list(range(vector_register_size))
    circuit.add(job_name, target_wires)

    # Hadamard block on clock register
    for q in range(vector_register_size, vector_register_size + clock_register_size):
        circuit.add("h", q)
        
    # Time evolution
    control_wire = 0
    for t in range(clock_register_size, 0, -1):
        # Construct matrix
        U = expm(1j * A * (np.pi / min_eigval / (2**(t - 1))))

        #if(verbose):
        #    display(np.round(U, 5))

        # Make controlled matrix
        CU = get_controlled_u(U)

        # Decompose matrix to u3,cx with Quantastica tools
        job_name = "u_" + str(2**t)
        job_id = QPS.generator.decompose_unitary(CU.tolist(), endianness="big", job_name=job_name, settings = { "pre_processing": "experimental5" }, start_job=True)
        job = QPS.generator.get_job(job_id, wait=True)

        # Add to circuit
        subroutine = CircuitLite()
        subroutine.from_toaster(job["output"]["circuits"][0])
        circuit.register_subroutine(job_name, subroutine)

        target_wires = []
        target_wires.append(vector_register_size + control_wire)
        target_wires = target_wires + list(range(vector_register_size))
        
        circuit.add(job_name, target_wires)
        
        control_wire += 1
    
    # IQFT - Inverse Quantum Fourier Transform (without SWAPs) 
    subroutine = get_qft_circuit(clock_register_size, inverse=True)
    circuit.register_subroutine("iqft", subroutine)
    target_wires = [i + vector_register_size for i in range(clock_register_size)]
    circuit.add("iqft", target_wires)

    # 1/X - Eigenvalue inverter
    subroutine = get_x_1_circuit(clock_register_size)
    circuit.register_subroutine("x_1", subroutine)
    target_wires = list(range(vector_register_size, vector_register_size + clock_register_size + 1))
    circuit.add("x_1", target_wires)

    # IQFT dagger - inverse of IQFT which is fancy name for QFT :), again without SWAPs
    subroutine = get_qft_circuit(clock_register_size, inverse=False)
    circuit.register_subroutine("iqft_dg", subroutine)
    target_wires = [i + vector_register_size for i in range(clock_register_size)]
    circuit.add("iqft_dg", target_wires)

    # Time evolution dagger - inverse time evoulution steps
    # This can be done by simple inverting time evolution steps, but CircuitLite doesn't have invert() method [TODO],
    # So we start over - construct inverse matrices and decompose
    control_wire = clock_register_size - 1
    for t in range(1, clock_register_size + 1):
        # Construct matrix
        U = expm(-1j * A * (np.pi / min_eigval / (2**(t - 1))))

        #if(verbose):
        #    display(np.round(U, 5))

        CU = get_controlled_u(U)
        
        # Decompose matrix to u3,cx with Quantastica tools
        job_name = "u_" + str(2**t) + "_dg"
        job_id = QPS.generator.decompose_unitary(CU.tolist(), endianness="big", job_name=job_name, settings = { "pre_processing": "experimental5" }, start_job=True)
        job = QPS.generator.get_job(job_id, wait=True)

        # Add to circuit
        subroutine = CircuitLite()
        subroutine.from_toaster(job["output"]["circuits"][0])
        circuit.register_subroutine(job_name, subroutine)

        target_wires = []
        target_wires.append(vector_register_size + control_wire)
        target_wires = target_wires + list(range(vector_register_size))
        
        circuit.add(job_name, target_wires)

        control_wire -= 1

    # Hadamard block on clock register
    for q in range(vector_register_size, vector_register_size + clock_register_size):
        circuit.add("h", q)

    return circuit
        
#
# Function takes linear system problem, constructs HHL circuit, executes it and
# extracts solution from statevector.
#
# For running on real QPU, we will need to perform tomography (otherwise it is
# impossible to read negative output values)
#
def linalg_qsolve(A, b, verbose=False, print_qasm=False):

    # Get classical solution (to compare with quantum solution)
    xc = np.linalg.solve(A, b)
    
    # We need this to normalize output.
    # Can be done without classical solution (but it is hard!)
    euclidean_norm = np.linalg.norm(xc)
    
    # Build circuit
    circuit = get_hhl_circuit(A, b, verbose=verbose)

    # Execute circuit
    simulator = SimulatorLite()
    simulator.execute(circuit, reverse_bits=True)    
    state_vector = simulator.state
    
    # Read raw result from state vector
    num_qubits = circuit.num_qubits()
    offset = 2**(num_qubits - 1)
    x = []
    for index in range(len(b)):
        x.append(state_vector[offset + index])
    x = np.real(x)

    # Normalize
    x = euclidean_norm * (x / np.linalg.norm(x))
    
    if(verbose):
        print("Classical solution:", xc)
        print("Quantum solution:  ", x)
        print("")
        print(circuit.count_ops())

    if(print_qasm):
        print("Program:\n")
        print(circuit.qasm())

    return x

