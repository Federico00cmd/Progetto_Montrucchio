import qiskit
import networkx as nx
from scipy.linalg import eig
import numpy as np
import random
import math
import copy
import random
import sympy as sp

def matrices():

    I = np.eye(2, dtype=complex)

    # Pauli-X (sigma_x)
    sigma_x = np.array([[0, 1],
                        [1, 0]], dtype=complex)

    # Pauli-Z (sigma_z)
    sigma_z = np.array([[1, 0],
                        [0, -1]], dtype=complex)

    # T gate: diagonal matrix with 1 and exp(i*pi/4)
    T = np.array([[1, 0],
                [0, np.exp(1j * np.pi/4)]], dtype=complex)

    # S gate: diagonal matrix with 1 and i
    S = np.array([[1, 0],
                [0, 1j]], dtype=complex)

    # Tdagger: conjugate transpose of T (diagonal with 1 and exp(-i*pi/4))
    Tdagger = np.array([[1, 0],
                        [0, np.exp(-1j * np.pi/4)]], dtype=complex)

    # Sdagger: conjugate transpose of S (diagonal with 1 and -i)
    Sdagger = np.array([[1, 0],
                        [0, -1j]], dtype=complex)

    # Rotation about the x-axis by angle theta:
    def Rx(theta):
        return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                        [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

    # Rotation about the z-axis by angle theta:
    def Rz(theta):
        return np.array([[np.exp(-1j * theta/2), 0],
                        [0, np.exp(1j * theta/2)]], dtype=complex)

    # Hadamard gate
    Hadamard = (1/np.sqrt(2)) * np.array([[1, 1],
                                            [1, -1]], dtype=complex)

    # Controlled-Z (Cz) gate: a 4x4 matrix.
    cz = np.array([[1, 0, 0,  0],
                [0, 1, 0,  0],
                [0, 0, 1,  0],
                [0, 0, 0, -1]], dtype=complex)

    # Controlled-X (Cx) gate, often known as CNOT: a 4x4 matrix.
    cx = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]], dtype=complex)

    # Create a dictionary of the matrices (and rotation functions)
    matrices = {
        'x': sigma_x,
        'sigma_z': sigma_z,
        'T': T,
        'S': S,
        'Tdagger': Tdagger,
        'Sdagger': Sdagger,
        'Rx': Rx,     # call Rx(theta) to get a rotation matrix in x
        'Rz': Rz,     # call Rz(theta) to get a rotation matrix in z
        'h': Hadamard,
        'cz': cz,
        'cx': cx
    }
    return matrices

def separate_qc(complete_qc, verbose = False):

    #create a list of lists where in each row are listened all the operations for a given qubit
    list_qc = [[[i]] for i in range(complete_qc.num_qubits)]
    discarded_operation_list = [0 for i in range(complete_qc.num_qubits)]

    for i in range(complete_qc.num_qubits):
        #explore for all the qubits
        for instr, qargs, cargs in complete_qc.data:
        # Check if qubit i is among the qubits the gate is applied to
            if complete_qc.qubits[i] in qargs:
                '''
                if verbose:
                    if len(qargs)>1:
                        print(f"Gate: {instr.name}, Parameters: {instr.params}, control: {qargs[0]._index}, Target: {qargs[1]._index}")
                    elif len(qargs) ==1:
                        print(f"Gate: {instr.name}, Parameters: {instr.params}")
                    else:
                        print("Operation {instr.name} not recognized")
                '''
                gates = matrices()
                if instr.name in gates:
                    if len(qargs) == 1:
                        list_qc[i].append([instr.name, qargs[0]._index, instr.params])
                    elif len(qargs) == 2:
                        list_qc[i].append([instr.name, [qargs[0]._index, qargs[1]._index], instr.params])
                    else:
                        print("Operations with more than one target or control are not managed here, the operation will be ignored")
                        discarded_operation_list[i] += 1
    return list_qc, discarded_operation_list

def create_connectivity_graph(list_qc):

    '''
    list_qc: previously defined list for the quantum circuit in function 'separate_qc'

    this function aims to create a graph that display the connections of the qubits. Each edge have 
    an associated weight that is the number of controlled operations from qubit i to qubit j
    '''

    connectivity_graph = nx.Graph()

    for qubit_list in list_qc:
        for operation in qubit_list:
            #control if the list is a gate and it is a controlled operation
            if len(operation)> 1 and isinstance(operation[1], list) and len(operation[1]) == 2:
                    #add by 1 if an operation is found
                    if connectivity_graph.has_edge(operation[1][0], operation[1][1]):
                        connectivity_graph[operation[1][0]][ operation[1][1]]['weight'] += 1
                    #create a new edge if an operation is discoveded
                    else:
                        connectivity_graph.add_edge(operation[1][0], operation[1][1], weight = 1)

    #since we count the connections twice, then we have to divide by two
    for u, v, data in connectivity_graph.edges(data=True):
        if 'weight' in data:
            data['weight'] *= 0.5  
        
    return connectivity_graph
        

def objective_state(state, graph, max_group_size, full_bonus = 1):
    """
    Computes the total connectivity score for the state.
    For each group (subset), it sums the weights of edges connecting every pair of nodes.
    """
    score = 0
    for group in state:
        group_nodes = list(group)
        for i in range(len(group_nodes)):
            for j in range(i+1, len(group_nodes)):
                u, v = group_nodes[i], group_nodes[j]
                if graph.has_edge(u, v):
                    score += 10*graph[u][v]['weight']

                if len(group) == max_group_size:
                    score += full_bonus
    return score

def ensure_all_assigned(state, all_nodes, max_group_size):
    """
    Ensures that every node in all_nodes is present in the state.
    If a node is missing, it is added to a group with available capacity.
    If no group has capacity for a missing node, a ValueError is raised.
    """
    assigned = set()
    for group in state:
        assigned |= group
    missing = all_nodes - assigned
    for node in missing:
        # Look for a group with available capacity
        for group in state:
            if len(group) < max_group_size:
                group.add(node)
                break
        else:
            raise ValueError(f"Not enough capacity to assign node {node}.")
    return state

def random_neighbor_new(state, all_nodes, max_group_size):
    """
    state: current state (a list of sets, each set is a group of nodes)
    all_nodes: a set of all nodes that must be assigned
    max_group_size: maximum number of nodes allowed in each group

    This function randomly chooses one of three operations:
      - "add": add a node from the unassigned set into a non-full group.
      - "remove": remove a node from a group (only if that group has > 1 node)
      - "move": move a node from one group to another non-full group.

    After performing the operation, it calls ensure_all_assigned to guarantee that every
    node in all_nodes is in the state. If it's not possible (because all groups are full), 
    an error is raised.
    """
    new_state = copy.deepcopy(state)
    move_type = random.choice(["add", "remove", "move"])

    # Compute the set of already assigned nodes
    assigned_nodes = set()
    for group in new_state:
        assigned_nodes |= group
    unassigned = list(all_nodes - assigned_nodes)

    if move_type == "add" and unassigned:
        # Choose a group that is not full
        non_full_groups = [i for i, group in enumerate(new_state) if len(group) < max_group_size]
        if non_full_groups:
            group_idx = random.choice(non_full_groups)
            node = random.choice(unassigned)
            new_state[group_idx].add(node)
        else:
            # All groups are full, fall back to move
            move_type = "move"

    if move_type == "remove":
        # To ensure that we don't leave a node unassigned,
        # only remove a node from groups with at least 2 nodes.
        non_singleton = [i for i, group in enumerate(new_state) if len(group) > 1]
        if non_singleton:
            group_idx = random.choice(non_singleton)
            node = random.choice(list(new_state[group_idx]))
            new_state[group_idx].remove(node)
        else:
            # If no group has more than one node, try adding instead.
            if unassigned:
                non_full_groups = [i for i, group in enumerate(new_state) if len(group) < max_group_size]
                if non_full_groups:
                    group_idx = random.choice(non_full_groups)
                    node = random.choice(unassigned)
                    new_state[group_idx].add(node)

    if move_type == "move":
        # Choose a non-empty source group.
        non_empty = [i for i, group in enumerate(new_state) if len(group) > 0]
        if non_empty:
            source_idx = random.choice(non_empty)
            node = random.choice(list(new_state[source_idx]))
            # Choose a target group (different from source) that is not full.
            target_indices = [i for i in range(len(new_state)) if i != source_idx and len(new_state[i]) < max_group_size]
            if target_indices:
                target_idx = random.choice(target_indices)
                new_state[source_idx].remove(node)
                new_state[target_idx].add(node)
            else:
                # If no target group has capacity, do nothing (or try a different move type).
                pass
        else:
            # If all groups are empty, try adding a node.
            if unassigned:
                group_idx = random.randint(0, len(new_state)-1)
                node = random.choice(unassigned)
                new_state[group_idx].add(node)
    
    # Finally, ensure that every node is assigned.
    new_state = ensure_all_assigned(new_state, all_nodes, max_group_size)
    return new_state

def random_neighbor(state, all_nodes, max_group_size):
    '''
    state: current state
    all_nodes: all the states that are present
    max_group_size: maximum number of nodes for each QPU (group)

    This function aims to create a random state using 3 attributes:
    -add: add a new node to a group
    -remove: remove a node from a group
    -move: move a node from one goup to another
    '''
    new_state = copy.deepcopy(state)
    move_type = random.choice(["add", "remove", "move"])
    
    # Compute the set of nodes that are already assigned to any group.
    assigned_nodes = set()
    #the already assigned nodes are united, then the unassigned are basically the negation of the union
    for group in new_state:
        assigned_nodes |= group
    unassigned = list(all_nodes - assigned_nodes)
    
    if move_type == "add" and unassigned:
        #choose a non full group
        non_full_groups = [i for i, group in enumerate(new_state) if len(group) < max_group_size]
        if non_full_groups:
            #pick at random an element from the non full groups
            group_idx = random.choice(non_full_groups)
            node = random.choice(unassigned)
            new_state[group_idx].add(node)
            return new_state
        else:
            #if all groups are full try with the move attribute
            move_type = "move"
    
    if move_type == "remove":
        #Choose a non-empty group and remove one node.
        non_empty = [i for i, group in enumerate(new_state) if len(group) > 0]
        if non_empty:
            group_idx = random.choice(non_empty)
            node = random.choice(list(new_state[group_idx]))
            new_state[group_idx].remove(node)
            return new_state
        else:
            #if no groups have a node, then try with add
            if unassigned:
                group_idx = random.randint(0, len(new_state) - 1)
                node = random.choice(unassigned)
                new_state[group_idx].add(node)
                return new_state
            else:
                return new_state
    
    if move_type == "move":
        #Choose a non-empty source group.
        non_empty = [i for i, group in enumerate(new_state) if len(group) > 0]
        if non_empty:
            source_idx = random.choice(non_empty)
            node = random.choice(list(new_state[source_idx]))
            # Choose a different target group that is not full.
            target_indices = [i for i in range(len(new_state)) if i != source_idx and len(new_state[i]) < max_group_size]
            if target_indices:
                target_idx = random.choice(target_indices)
                new_state[source_idx].remove(node)
                new_state[target_idx].add(node)
                return new_state
            else:
                #if all groups are full go in remove
                new_state[source_idx].remove(node)
                return new_state
        else:
            #if all groups are empty go in add
            if unassigned:
                group_idx = random.randint(0, len(new_state) - 1)
                node = random.choice(unassigned)
                new_state[group_idx].add(node)
                return new_state
            else:
                return new_state
    return new_state

def simulated_annealing_grouping(graph, max_groups, max_group_size, 
                                 simulated_annealing_parameters):
    '''
    graph: connectivity graph provided by the function create_connectivity_graph
    max_groups: total number of QPUs that we use in our core
    max_group_size: maximum number of nodes that are allowed in the QPU

    This function aims to use the simulated annealing in order to find the better configuration 
    for the connectivity graphs.
    '''

    initial_temp = simulated_annealing_parameters[0]
    cooling_rate = simulated_annealing_parameters[1]
    iterations = simulated_annealing_parameters[2]

    #set is a built in function in python
    all_nodes = set(graph.nodes())
    
    # Start with an initial state: a list of empty groups.
    current_state = [set() for _ in range(max_groups)]
    
    #Assign at random the nodes (qubits) if there is enough space
    for node in all_nodes:
        group_idx = random.randint(0, max_groups - 1)
        if len(current_state[group_idx]) < max_group_size:
            current_state[group_idx].add(node)
    
    #evaluate the score by the function objective_state
    current_score = objective_state(current_state, graph, max_group_size)
    #in order to copy use deepcopy so the original state is not modified
    best_state = copy.deepcopy(current_state)
    #at the beginning I suppose to stay in the best situation
    best_score = current_score
    temp = initial_temp
    
    for i in range(iterations):
        new_state = random_neighbor(current_state, all_nodes, max_group_size)
        new_score = objective_state(new_state, graph, max_group_size)
        delta = new_score - current_score
        
        #Accept the solution if it is a good solution or, with a small prob if it is a worse one in order to not be trapped in local minima
        if delta > 0 or random.random() < math.exp(delta / temp):
            current_state = new_state
            current_score = new_score
            if new_score > best_score:
                best_state = copy.deepcopy(new_state)
                best_score = new_score
        
        temp *= cooling_rate
    
    return best_state, best_score


'''

# NON FUNZIONA
def evolve_internal(group, list_qc, dt, matrices):
    """
    Constructs the internal evolution operator for a given group over time dt.
    
    It scans the operations for qubits in 'group' (from list_qc) and multiplies the corresponding
    unitary operators, stopping when an operation that involves a qubit outside the group is found.
    
    Parameters:
      group    : A set of qubit indices (one group from best_state).
      list_qc  : The list of operations per qubit.
      dt       : The time step (here used to set the duration of evolution).
      matrices : Dictionary of gate matrices.
      
    Returns:
      U_internal : The net evolution operator for the group over time dt.
    """
    # For simplicity, we assume that the gate matrices approximate the evolution over dt.
    dim = 2 ** len(group)
    U_internal = np.eye(dim, dtype=complex)
    matrices_dict = matrices()

    # Create a deep copy of list_qc to track remaining operations
    to_do_list_qc = copy.deepcopy(list_qc)

    # For each qubit in the group, process its operations sequentially.
    for qubit in group:
        # Iterate through operations in the copied list
        for operations in to_do_list_qc:
            if operations[0] == qubit:
                ops_to_remove = []
                # Skip the first element (qubit index), process the operations
                for op in operations[1:]:
                    # Check if it is a controlled operation
                    if len(op) > 1 and isinstance(op[1], list):
                        targets = op[1]
                        # If any target is outside the group, stop processing this operation
                        if any(t not in group for t in targets):
                            break  # Skip to the next qubit if external target is found
                        else:
                            # Internal controlled operation: multiply its matrix.
                            U_gate = matrices_dict.get(op[0])
                            U_gate = expand_gate(U_gate= U_gate, n = len(group), targets= targets)
                            if U_gate is not None:
                                U_internal = U_gate @ U_internal
                                ops_to_remove.append(op)  # Mark the operation to be removed
                    else:
                        # For a single-qubit gate, simply multiply the operator.
                        U_gate = expand_gate(U_gate= U_gate, n = len(group), targets= targets)
                        U_gate = matrices_dict.get(op[0])
                        if U_gate is not None:
                            U_internal = U_gate @ U_internal
                            ops_to_remove.append(op)  # Mark the operation to be removed

                # After processing, remove the operations from the list
                for op in ops_to_remove:
                    operations.remove(op)

    return U_internal, to_do_list_qc
'''
#####################################################

def separate_qc_order(complete_qc, verbose=False):
    """
    Extract an ordered list of operations from the quantum circuit complete_qc.
    Each operation is stored in the order they appear in the circuit (using qc.data).
    
    For a single-qubit operation, the format is:
      [instr.name, qargs[0]._index, instr.params, order]
    
    For a two-qubit operation, the format is:
      [instr.name, [qargs[0]._index, qargs[1]._index], instr.params, order]
    
    Returns:
      list_qc_order: A list of operations (in circuit order).
      discarded_ops: The count of operations that were ignored.
    """
    list_qc_order = []
    discarded_ops = 0

    for idx, (instr, qargs, cargs) in enumerate(complete_qc.data):
        # If you want verbose printing of gate details, you can uncomment below:
        # if verbose:
        #     print(f"Gate: {instr.name}, qargs: {[q._index for q in qargs]}, params: {instr.params}")
        gate_dict = matrices()
        if instr.name in gate_dict:
            if len(qargs) == 1:
                op = [instr.name, qargs[0]._index, instr.params, idx]
                list_qc_order.append(op)
                if verbose:
                    print(f"Added single-qubit op: {op}")
            elif len(qargs) == 2:
                op = [instr.name, [qargs[0]._index, qargs[1]._index], instr.params, idx]
                list_qc_order.append(op)
                if verbose:
                    print(f"Added two-qubit op: {op}")
            else:
                if verbose:
                    print(f"Discarded op {instr.name}: more than two qubits.")
                discarded_ops += 1
        else:
            if verbose:
                print(f"Discarded op {instr.name}: gate not recognized.")
            discarded_ops += 1

    return list_qc_order, discarded_ops

def expand_gate(U_gate, n, targets):
    """
    Embed U_gate (acting on some k qubits) into the full Hilbert space of n qubits.
    
    Parameters:
      U_gate  : The gate matrix (of dimension 2^k x 2^k) that acts on the target qubits.
      n       : Total number of qubits in the system.
      targets : A list of qubit indices where the gate is applied.
    
    Returns:
      U_full  : The full 2^n x 2^n matrix representing the gate acting on the specified targets.
    """
    if not isinstance(targets, list):
        targets = [targets]
    print("Targets:" ,targets)
    I = np.eye(2, dtype=complex)  # The single-qubit identity matrix.
    U_full = 1  # Start with a scalar 1 for the tensor product.
    if len(targets) == 1:
        #we have no a control operation
        for i in range(n ):
            if i in targets:
                U_full = np.kron(U_full, U_gate)  # Place the gate at this qubit.
            else:
                U_full = np.kron(U_full, I)  # Place the identity on qubits that are not targeted.
    else:
        control_qubit = targets[0]
        target_qubit = targets[1]
        control_matrix = U_gate[:2, :2]
        target_matrix = U_gate[2:, 2:]
        print("\n\n", target_matrix)
        #we have a control operation
        for i in range(n ):
            if i in targets:
                if i == control_qubit:
                    U_full = np.kron(U_full, control_matrix)  # Place the gate at this qubit.
                else:
                    U_full = np.kron(U_full, target_matrix)
            else:
                U_full = np.kron(U_full, I)  # Place the identity on qubits that are not targeted.

    return U_full

def evolve_internal_new(group, list_qc_ordered, tot_qubits, T_decoherence, decoherence = True):

    """
    Evolves a quantum system by applying a sequence of quantum gates to the qubits in the specified group.

    The function processes the gates in the `list_qc_ordered`, which are applied in sequence to the qubits 
    belonging to the `group`. Each gate is expanded to the full Hilbert space of the system, and decoherence 
    can be optionally applied to each gate using the provided `T_decoherence` parameter.

    Parameters:
    - group (list): A list of qubits that the function can operate on.
    - list_qc_ordered (list): A list of quantum circuit operations, where each operation is represented 
      as [gate_name, target_qubits, parameters, gate_qubit_index].
    - tot_qubits (int): The total number of qubits in the quantum system.
    - T_decoherence (float): A parameter representing the decoherence time.
    - decoherence (bool, optional): If True, applies decoherence to the quantum operations. Defaults to True.

    Returns:
    - np.ndarray: The total unitary evolution matrix representing the system's evolution after applying 
      the sequence of gates.

    Note:
    - The function assumes that `gates` is a dictionary containing matrix representations for each gate 
      (e.g., `rx`, `rz`, `cx`, `h`, etc.).
    - Decoherence is applied by modifying the gate matrices before expanding them to the full system size.
    - The evolution is accumulated in a matrix `U_internal`, which is multiplied by each gate's expanded form.
    """
    # list_qc_ordered example: [['cx', [0, 1], [], 0], ['h', 2, [], 1], ['x', 4, [], 2], ['x', 5, [], 3], ['x', 4, [], 4]], 0
    operation = list_qc_ordered[0]

    operation_name = operation[0]
    operation_targets = operation[1]
    operation_params = operation[2]
    operation_qubit = operation[3]
    # All the involved qubits are inside the group

    operation_targets_list = [operation_targets]
    print("\n\n The group is:", group, "\n\n")
    gates = matrices()
    U_internal = np.eye(2 ** tot_qubits)
    while all(t in group for t in operation_targets_list):

        print(list_qc_ordered)
        print("operation:", operation)
        print("operation targets:", operation_targets)
        # rx, rz have also theta as parameter
        if operation_name in ['rx', 'rz']:
            U_gate = gates[operation_name](operation_params)  # call the function to get the evaluated matrix
        else:
            U_gate = gates[operation_name]
        
        if decoherence:
            U_gate = apply_decoherence(U = U_gate, T_decoherence= T_decoherence)
        U_tot = expand_gate(U_gate = U_gate, n = tot_qubits, targets= operation_targets)
        
        print("\n\n The dimension of U_internal is:", U_internal.shape, "while the dimension of the U_tot is:", U_tot.shape)
        U_internal = U_internal @ U_tot
        #remove the done operation
        list_qc_ordered.pop(0)

        #Fetch a new operation
        if list_qc_ordered:
            operation = list_qc_ordered[0]
            operation_name = operation[0]
            operation_targets = operation[1]
            operation_params = operation[2]
            operation_targets_list = [operation_targets]
        else:
            break

        print("First loop ok")

    return U_internal

def apply_decoherence(U, T_decoherence):
    """
    Applies a simple decoherence model where the unitary matrix U decays over time
    with a characteristic time T_decoherence.
    
    Keeps t as a symbolic parameter.
    """
    t = sp.Symbol('t')  # Define t as a symbolic variable
    decay_factor = sp.exp(-t / T_decoherence)  # Use SymPy's exp function
    return decay_factor * U  # Keep U symbolic

import numpy as np
import sympy as sp

def is_t_present_in_matrix(H, t):
    # If the matrix is a SymPy matrix
    if isinstance(H, sp.Matrix):
        # Check if the symbol 't' is present in the matrix
        if t in H.free_symbols:
            return True
    # If the matrix is a NumPy array, check if it contains any SymPy expressions
    elif isinstance(H, np.ndarray):
        # Convert the NumPy array to a SymPy matrix to check for symbols
        H_sympy = sp.Matrix(H)
        if t in H_sympy.free_symbols:
            return True
    
    return False

def evaluate_matrix_for_t(H, t_value):
    """Evaluate symbolic matrix with 't' substituted by its numerical value."""
    if isinstance(H, sp.Matrix):  # Check if it's a SymPy matrix
        return H.subs('t', t_value).evalf()  # Substitute and evaluate symbolically
    elif isinstance(H, np.ndarray):  # If it's a NumPy array, convert to SymPy matrix first
        H_sympy = sp.Matrix(H)
        return H_sympy.subs('t', t_value).evalf()  # Substitute and evaluate symbolically
    return H  # In case H is already numeric

def taylor_expansion_exponent(exponent, n):
    """
    Expands e^(exponent) using the Taylor series up to degree n.
    :param exponent: The matrix exponent (complex matrix)
    :param n: The degree of the expansion
    :return: The matrix exponential as a Taylor series expansion
    """
    # Initialize result as the identity matrix (I)
    result = 1
    
    # Initialize term (first term is exponent^0 / 0! = I)
    term = 1
    
    # Iterate over degrees 1 to n
    for i in range(1, n+1):
        term = np.dot(term, exponent) / i  # term = (exponent^i) / i!
        print("Therm:", term)
        result += term  # Add the term to the result
    
    return result

# Example usage
H_numeric = np.eye(16)  # Example: identity matrix as the exponent (numeric)
n = 10  # Degree of expansion

# Compute the Taylor expansion of e^exponent
expanded_matrix = taylor_expansion_exponent(H_numeric, n)

def trotter_suzuki(hamiltonians, t, k=10, eval_t=0):
    from scipy.linalg import expm
    """
    Applies the first-order Trotter-Suzuki decomposition to a list of Hamiltonian matrices.
    
    Parameters:
        hamiltonians : list of sympy matrices (H_i), each of size (n x n)
        t        : symbolic time variable (sympy.Symbol or numeric)
        k        : number of Trotter steps (higher k improves accuracy)
        eval_t   : Optional numeric value for t to evaluate the matrix at (default is None, for symbolic expression)
    
    Returns:
        U_trotter : Approximated evolution operator as a symbolic or numeric matrix.
    """
    # Ensure t is a symbolic variable if it's not numeric
    if not isinstance(t, sp.Basic):
        t = sp.Symbol('t')

    #dimension of the Hamiltonians (all the same)
    n = hamiltonians[0].shape[0] 

    U_trotter = sp.eye(n)  # Start with identity matrix

    # Compute the first-order Trotter expansion
    for _ in range(k):
        U_step = sp.eye(n)  # Start each step with identity
        for H in hamiltonians:
            print("\n\nThe Hamiltonian is:", H)
            if is_t_present_in_matrix(H = H, t = t):
                H_numeric = evaluate_matrix_for_t(H = H, t_value = eval_t)
            else:
                H_numeric = H
            print("The evaluated hamiltonian is:", H_numeric)
            #evolutions throught Trotter Suzuki
            exponent = -1j * (t / k) * H_numeric
            exponential = taylor_expansion_exponent(exponent= exponent, n = 10)
            print(exponential)
            #U_step = U_step @ expm(-1j * (t / k) * H_numeric)
        #multiply k times
        U_trotter = U_step @ U_trotter  
    
    # Simplify the final expression
    U_trotter = sp.simplify(U_trotter)
    
    # If a numeric value for t is provided, evaluate the matrix at that t
    if eval_t is not None:
        U_trotter = U_trotter.subs(t, eval_t)  # Substitute the symbolic t with the numeric value
    
    return U_trotter

def simulated_annealing_parameters(initial_temp=100, cooling_rate=0.95, iterations=10000):
    """
    Separated function that simply load the values for the simulated annealing
    """
    return initial_temp, cooling_rate, iterations

def circuit_solver(qc, max_groups, max_groups_size, tot_qubits, T_decoherence,t,k,  fixed = True):
    """
    Tryes to solve the circuuit by alternating the single node execution and the multiple node execution
    
    fixed: True if the groups remains always the same for all the steps
    """

    simulated_annealing_params = simulated_annealing_parameters()

    list_qc_ordered = separate_qc_order(complete_qc= qc)
    list_qc_unordered = separate_qc(complete_qc= qc)

    # Obtain the connectivity graph
    connectivity_graph = create_connectivity_graph(list_qc= list_qc_unordered)
    best_group_set = simulated_annealing_grouping(graph= connectivity_graph, max_groups= max_groups, max_group_size= max_groups_size, simulated_annealing_parameters= simulated_annealing_parameters())

    if fixed:
        # Immutable group
        best_group_set = frozenset(best_group_set)
        list_qc_ordered_cpy = copy.deepcopy(list_qc_ordered)
        while list_qc_ordered_cpy:
            hamiltonians_list = []
            for groups in best_group_set:
                # Calculate the evolution of each group i an independent way
                U_internal = evolve_internal_new(group= best_group_set, list_qc_ordered= list_qc_ordered_cpy, tot_qubits= tot_qubits, T_decoherence= T_decoherence)
                hamiltonians_list.append(U_internal)

            U_trotter = trotter_suzuki(hamiltonians= hamiltonians_list, eval_t = t)
    
    else:
        list_qc_ordered_cpy = copy.deepcopy(list_qc_ordered)
        while list_qc_ordered_cpy:
            hamiltonians_list = []
            for groups in best_group_set:
                # Calculate the evolution of each group i an independent way
                U_internal = evolve_internal_new(group= best_group_set, list_qc_ordered= list_qc_ordered_cpy, tot_qubits= tot_qubits, T_decoherence= T_decoherence)
                hamiltonians_list.append(U_internal)

            U_trotter = trotter_suzuki(hamiltonians= hamiltonians_list, eval_t = t)

            # Obtain the connectivity graph and update the best_grpup
            connectivity_graph = create_connectivity_graph(list_qc= list_qc_ordered_cpy)
            best_group_set = simulated_annealing_grouping(graph= connectivity_graph, max_groups= max_groups, max_group_size= max_groups_size, simulated_annealing_parameters= simulated_annealing_parameters())

def circuit_solver_final(qc, max_groups, max_group_size, T_decoherence, t, k, fixed=True):
    """
    Solves the quantum circuit by progressively applying the evolution operations extracted 
    from the circuit. First, it separates the circuit into an ordered list of operations and 
    builds a connectivity graph to pre-allocate qubits into groups. Then, it repeatedly evolves 
    each group and combines the internal evolutions via a Trotter–Suzuki decomposition until all 
    operations have been processed.
    
    Parameters:
      qc             : The quantum circuit (an instance of qiskit.QuantumCircuit).
      max_groups     : Total number of groups (QPUs) to use.
      max_group_size : Maximum number of qubits per group.
      T_decoherence  : The decoherence time parameter.
      t              : The time parameter used in the Trotter–Suzuki decomposition.
      k              : Number of Trotter steps (improves accuracy if increased).
      fixed          : If True, the qubit grouping remains constant during the evolution.
    
    Returns:
      U_total        : The overall unitary evolution operator for the circuit.
    """
    # Separate the circuit into an ordered list of operations.
    list_qc_ordered, _ = separate_qc_order(qc, verbose=False)
    # Also separate by qubit (used for connectivity)
    list_qc_unordered, _ = separate_qc(qc, verbose=False)
    
    # Build the connectivity graph from the unordered operations.
    connectivity_graph = create_connectivity_graph(list_qc_unordered)
    
    # Use simulated annealing to group the qubits.
    best_group_set, _ = simulated_annealing_grouping(
        graph=connectivity_graph,
        max_groups=max_groups,
        max_group_size=max_group_size,
        simulated_annealing_parameters=simulated_annealing_parameters()
    )
    
    # The total number of qubits comes from the circuit.
    tot_qubits = qc.num_qubits
    # Initialize overall evolution operator to the identity.
    U_total = np.eye(2 ** tot_qubits, dtype=complex)
    
    # Create a deep copy of the ordered list so that we can remove operations as they are processed.
    remaining_ops = copy.deepcopy(list_qc_ordered)
    
    # Main loop: process until no operations remain.
    while remaining_ops:
        hamiltonians_list = []
        # For each group in our best grouping, compute the evolution operator.
        for group in best_group_set:
            # evolve_internal_new expects a group (list of qubit indices) and the list of operations.
            # It processes (and pops) the operations that act exclusively on qubits in the group.
            U_internal = evolve_internal_new(
                group=list(group),       # Convert set to list if needed
                list_qc_ordered=remaining_ops,
                tot_qubits=tot_qubits,
                T_decoherence=T_decoherence,
                decoherence=True
            )
            hamiltonians_list.append(U_internal)
        
        # Combine the evolution operators from each group using a Trotter–Suzuki decomposition.
        U_trotter = trotter_suzuki(hamiltonians=hamiltonians_list, t=t, k=k, eval_t=t)
        # Update the total evolution operator.
        U_total = U_trotter @ U_total
        
        # (Optional) If groups are not fixed, update the connectivity and re-calculate the grouping.
        if not fixed:
            connectivity_graph = create_connectivity_graph(remaining_ops)
            best_group_set, _ = simulated_annealing_grouping(
                graph=connectivity_graph,
                max_groups=max_groups,
                max_group_size=max_group_size,
                simulated_annealing_parameters=simulated_annealing_parameters()
            )
    
    return U_total


def statevector_evaluator(qc):
    backend = Aer.get_backend('statevector_simulator')
    result = execute(qc, backend=backend).result()
    psi_qiskit = result.get_statevector(qc)

    return psi_qiskit

def get_probabilities(state_vector, tot_qubits, qubit_index):
    """
    Compute the probabilities that a given qubit is measured in |0> and |1>.
    Assumes that the state vector is ordered with the rightmost bit as qubit 0.
    
    Parameters:
      state_vector: A numpy array representing the state vector.
      tot_qubits: Total number of qubits in the state.
      qubit_index: The qubit index for which to compute probabilities.
    
    Returns:
      A tuple (prob0, prob1) corresponding to the probability of measuring |0> and |1>.
    """
    prob0 = 0.0
    prob1 = 0.0
    # Loop over each basis state index and its amplitude
    for i, amplitude in enumerate(state_vector):
        # Convert the index to a binary string with tot_qubits bits.
        state_str = format(i, '0' + str(tot_qubits) + 'b')
        # The bit corresponding to qubit_index (with rightmost bit = qubit 0)
        if state_str[-(qubit_index + 1)] == '0':
            prob0 += np.abs(amplitude)**2
        else:
            prob1 += np.abs(amplitude)**2
    return prob0, prob1

def compare_results(U_total, statevector):
    """
    Compare the output probabilities obtained from Qiskit's statevector simulation
    with the probabilities computed by applying the full unitary operator U_total.
    
    Parameters:
      U_total: The total unitary matrix (numpy array) representing the evolution.
      statevector: The final statevector from Qiskit's simulation.
      
    Prints the probability for each qubit being in |0> and |1> for both methods.
    """
    # Determine the total number of qubits (assuming statevector length is 2^n)
    tot_qubits = int(np.log2(len(statevector)))
    
    # Compute the final state using the eigenvalue approach:
    # Define the initial state |0...0>
    psi_init = np.zeros(2**tot_qubits, dtype=complex)
    psi_init[0] = 1.0
    psi_eigen = U_total @ psi_init
    
    # Compare probabilities for each qubit:
    for qubit in range(tot_qubits):
        prob0_qiskit, prob1_qiskit = get_probabilities(statevector, tot_qubits, qubit)
        prob0_eigen, prob1_eigen = get_probabilities(psi_eigen, tot_qubits, qubit)
        
        print(f"\nFor qubit {qubit}:")
        print("  Qiskit simulation probabilities:")
        print("    |0>: {:.4f}, |1>: {:.4f}".format(prob0_qiskit, prob1_qiskit))
        print("  Eigenvalue method probabilities:")
        print("    |0>: {:.4f}, |1>: {:.4f}".format(prob0_eigen, prob1_eigen))