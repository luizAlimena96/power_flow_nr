import numpy as np

# Base of the system
S_base = 100  # MVA

# Data for buses
buses = [
    {"num": 1, "type": "slack", "V_spec": 1.0, "delta_spec": 0.0, "PD": 0, "QD": 0, "PG": 0, "QG": 0},
    {"num": 2, "type": "PQ", "PD": 200, "QD": 100, "PG": 0.0, "QG": 0.0},
    {"num": 3, "type": "PV", "V_spec": 1.05, "PD": 0, "QD": 0, "PG": 200, "QG": 0},
    {"num": 4, "type": "PQ", "PD": 300, "QD": 150, "PG": 0.0, "QG": 0.0},
]

# Data for transmission lines
lines = [
    {"from": 1, "to": 2, "R": 0.004, "X": 0.04, "MVAR_max": 12, "MVA_max": 300},
    {"from": 1, "to": 4, "R": 0.004, "X": 0.04, "MVAR_max": 12, "MVA_max": 300},
    {"from": 2, "to": 3, "R": 0.002, "X": 0.02, "MVAR_max": 5, "MVA_max": 300},
    {"from": 3, "to": 4, "R": 0.007, "X": 0.07, "MVAR_max": 20, "MVA_max": 300},
]

def create_admittance_matrix(lines, buses):
    n = len(buses)
    Y = np.zeros((n, n), dtype=complex)
    for line in lines:
        i = line["from"] - 1
        j = line["to"] - 1
        Z = line["R"] + 1j * line["X"]
        Y_line = 1 / Z
        Y[i, i] += Y_line
        Y[j, j] += Y_line
        Y[i, j] -= Y_line
        Y[j, i] -= Y_line
    return Y

def newton_raphson(buses, lines, max_iterations=10, tolerance=1e-6):
    Y = create_admittance_matrix(lines, buses)
    num_buses = len(buses)

    # Initialize voltage magnitudes and angles
    V = np.array([bus.get('V_spec', 1.0) for bus in buses])
    delta = np.zeros(num_buses)

    # Set slack bus voltage and angle
    for i, bus in enumerate(buses):
        if bus['type'] == 'slack':
            V[i] = bus['V_spec']
            delta[i] = np.radians(bus['delta_spec'])
            slack_bus_index = i
            break

    # Identify PQ and PV buses
    pq_indices = []
    pv_indices = []
    for i, bus in enumerate(buses):
        if bus['type'] == 'PQ':
            pq_indices.append(i)
        elif bus['type'] == 'PV':
            pv_indices.append(i)

    # Start iterations
    for iteration in range(max_iterations):
        P_mismatch = np.zeros(num_buses)
        Q_mismatch = np.zeros(num_buses)

        # Calculate power injections and mismatches
        for i in range(num_buses):
            P_calc = Q_calc = 0.0
            for j in range(num_buses):
                Y_ij = Y[i, j]
                theta_ij = np.angle(Y_ij)
                G_ij = Y_ij.real
                B_ij = Y_ij.imag
                P_calc += V[i] * V[j] * (G_ij * np.cos(delta[i] - delta[j]) + B_ij * np.sin(delta[i] - delta[j]))
                Q_calc += V[i] * V[j] * (G_ij * np.sin(delta[i] - delta[j]) - B_ij * np.cos(delta[i] - delta[j]))

            # Specified power
            PD_i = buses[i].get('PD', 0.0)
            QD_i = buses[i].get('QD', 0.0)
            PG_i = buses[i].get('PG', 0.0)
            QG_i = buses[i].get('QG', 0.0)
            P_spec = (PG_i - PD_i) / S_base
            Q_spec = (QG_i - QD_i) / S_base if QG_i is not None else -QD_i / S_base

            P_mismatch[i] = P_spec - P_calc
            Q_mismatch[i] = Q_spec - Q_calc

            # For slack bus
            if buses[i]['type'] == 'slack':
                P_mismatch[i] = 0.0
                Q_mismatch[i] = 0.0
            elif buses[i]['type'] == 'PV':
                Q_mismatch[i] = 0.0  # Do not include Q mismatch for PV buses

        # Form the mismatch vector
        mismatches = np.hstack((P_mismatch[1:], Q_mismatch[pq_indices]))

        # Check convergence
        if np.max(np.abs(mismatches)) < tolerance:
            print(f"Convergence achieved after {iteration+1} iterations.")
            break

        # Build Jacobian matrix
        num_unknowns = len(delta) - 1 + len(pq_indices)
        J = np.zeros((num_unknowns, num_unknowns))

        # H matrix (∂P/∂δ)
        for i in range(1, num_buses):
            for j in range(1, num_buses):
                if i == j:
                    H = 0.0
                    for k in range(num_buses):
                        if k != i:
                            G_ik = Y[i, k].real
                            B_ik = Y[i, k].imag
                            H += V[i] * V[k] * (-G_ik * np.sin(delta[i] - delta[k]) + B_ik * np.cos(delta[i] - delta[k]))
                    J[i - 1, j - 1] = H
                else:
                    G_ij = Y[i, j].real
                    B_ij = Y[i, j].imag
                    J[i - 1, j - 1] = V[i] * V[j] * (G_ij * np.sin(delta[i] - delta[j]) - B_ij * np.cos(delta[i] - delta[j]))

        # N matrix (∂P/∂V)
        for idx_i, i in enumerate(range(1, num_buses)):
            if i in pq_indices:
                row = idx_i
                for idx_j, j in enumerate(pq_indices):
                    col = len(delta) - 1 + idx_j
                    if i == j:
                        N = 0.0
                        for k in range(num_buses):
                            G_ik = Y[i, k].real
                            B_ik = Y[i, k].imag
                            N += V[k] * (G_ik * np.cos(delta[i] - delta[k]) + B_ik * np.sin(delta[i] - delta[k]))
                        N *= 1
                        J[row, col] = N
                    else:
                        G_ij = Y[i, j].real
                        B_ij = Y[i, j].imag
                        J[row, col] = V[i] * (G_ij * np.cos(delta[i] - delta[j]) + B_ij * np.sin(delta[i] - delta[j]))

        # M matrix (∂Q/∂δ)
        for idx_i, i in enumerate(pq_indices):
            row = len(delta) - 1 + idx_i
            for j in range(1, num_buses):
                if i == j:
                    M = 0.0
                    for k in range(num_buses):
                        if k != i:
                            G_ik = Y[i, k].real
                            B_ik = Y[i, k].imag
                            M += V[i] * V[k] * (G_ik * np.cos(delta[i] - delta[k]) + B_ik * np.sin(delta[i] - delta[k]))
                    M *= 1
                    J[row, j - 1] = M
                else:
                    G_ij = Y[i, j].real
                    B_ij = Y[i, j].imag
                    J[row, j - 1] = V[i] * V[j] * (-G_ij * np.cos(delta[i] - delta[j]) - B_ij * np.sin(delta[i] - delta[j]))

        # L matrix (∂Q/∂V)
        for idx_i, i in enumerate(pq_indices):
            row = len(delta) - 1 + idx_i
            for idx_j, j in enumerate(pq_indices):
                col = len(delta) - 1 + idx_j
                if i == j:
                    L = 0.0
                    for k in range(num_buses):
                        G_ik = Y[i, k].real
                        B_ik = Y[i, k].imag
                        L += V[k] * (G_ik * np.sin(delta[i] - delta[k]) - B_ik * np.cos(delta[i] - delta[k]))
                    L *= 1
                    L -= 2 * V[i] * Y[i, i].imag
                    J[row, col] = L
                else:
                    G_ij = Y[i, j].real
                    B_ij = Y[i, j].imag
                    J[row, col] = V[i] * (G_ij * np.sin(delta[i] - delta[j]) - B_ij * np.cos(delta[i] - delta[j]))

        # Solve for corrections
        corrections = np.linalg.solve(J, mismatches)

        # Update angles and voltages
        delta[1:] += corrections[0:len(delta) - 1]
        for idx, i in enumerate(pq_indices):
            V[i] += corrections[len(delta) - 1 + idx]

    return V, np.degrees(delta)

def fast_decoupled_method(buses, lines, max_iterations=10, tolerance=1e-6):
    Y = create_admittance_matrix(lines, buses)
    num_buses = len(buses)

    # Initialize voltage magnitudes and angles
    V = np.array([bus.get('V_spec', 1.0) for bus in buses])
    delta = np.zeros(num_buses)

    # Set slack bus voltage and angle
    slack_bus_index = None
    for i, bus in enumerate(buses):
        if bus['type'] == 'slack':
            V[i] = bus['V_spec']
            delta[i] = np.radians(bus['delta_spec'])
            slack_bus_index = i
            break
    if slack_bus_index is None:
        raise ValueError("Slack bus not defined in buses data.")

    # Identify PQ and PV buses
    pq_indices = []
    pv_indices = []
    for i, bus in enumerate(buses):
        if bus['type'] == 'PQ':
            pq_indices.append(i)
        elif bus['type'] == 'PV':
            pv_indices.append(i)

    # Build B' and B" matrices
    B_prime = -np.imag(Y)
    B_double_prime = -np.imag(Y)

    # Reduce matrices by removing slack bus
    B_prime_reduced = np.delete(np.delete(B_prime, slack_bus_index, axis=0), slack_bus_index, axis=1)
    B_double_prime_reduced = np.delete(np.delete(B_double_prime, slack_bus_index, axis=0), slack_bus_index, axis=1)

    # Map original indices to reduced indices (after removing slack bus)
    reduced_bus_indices = [i for i in range(num_buses) if i != slack_bus_index]
    original_to_reduced_index = {}
    reduced_index = 0
    for i in range(num_buses):
        if i != slack_bus_index:
            original_to_reduced_index[i] = reduced_index
            reduced_index += 1

    # Adjust PQ indices to match reduced matrix
    pq_indices_reduced = [original_to_reduced_index[i] for i in pq_indices if i != slack_bus_index]

    # Start iterations
    for iteration in range(max_iterations):
        P_mismatch = np.zeros(num_buses)
        Q_mismatch = np.zeros(num_buses)

        # Calculate power injections and mismatches
        for i in range(num_buses):
            P_calc = Q_calc = 0.0
            for j in range(num_buses):
                Y_ij = Y[i, j]
                G_ij = Y_ij.real
                B_ij = Y_ij.imag
                P_calc += V[i] * V[j] * (G_ij * np.cos(delta[i] - delta[j]) + B_ij * np.sin(delta[i] - delta[j]))
                Q_calc += V[i] * V[j] * (G_ij * np.sin(delta[i] - delta[j]) - B_ij * np.cos(delta[i] - delta[j]))

            # Specified power
            PD_i = buses[i].get('PD', 0.0)
            QD_i = buses[i].get('QD', 0.0)
            PG_i = buses[i]['PG'] if buses[i].get('PG') is not None else 0.0
            QG_i = buses[i]['QG'] if buses[i].get('QG') is not None else 0.0
            P_spec = (PG_i - PD_i) / S_base
            Q_spec = (QG_i - QD_i) / S_base

            P_mismatch[i] = P_spec - P_calc
            Q_mismatch[i] = Q_spec - Q_calc

            # For slack bus and PV buses
            if buses[i]['type'] == 'slack':
                P_mismatch[i] = 0.0
                Q_mismatch[i] = 0.0
            elif buses[i]['type'] == 'PV':
                Q_mismatch[i] = 0.0  # Do not include Q mismatch for PV buses

        # Form mismatch vectors
        P_mismatch_reduced = P_mismatch[reduced_bus_indices]
        Q_mismatch_pq = Q_mismatch[pq_indices]

        # Check convergence
        if np.max(np.abs(P_mismatch_reduced)) < tolerance and np.max(np.abs(Q_mismatch_pq)) < tolerance:
            print(f"Convergence achieved after {iteration+1} iterations.")
            break

        # Solve for angle corrections
        delta_correction = np.linalg.solve(B_prime_reduced, P_mismatch_reduced)
        # Update delta for buses excluding slack bus
        for idx, i in enumerate(reduced_bus_indices):
            delta[i] += delta_correction[idx]

        # Solve for voltage corrections
        B_double_prime_pq = B_double_prime_reduced[np.ix_(pq_indices_reduced, pq_indices_reduced)]
        Q_mismatch_pq_array = Q_mismatch[pq_indices]
        V_correction = np.linalg.solve(B_double_prime_pq, Q_mismatch_pq_array)
        # Update V for PQ buses
        for idx, i in enumerate(pq_indices):
            V[i] += V_correction[pq_indices_reduced.index(original_to_reduced_index[i])]

    return V, np.degrees(delta)


# Run Fast Decoupled Method
V_fd, delta_fd = fast_decoupled_method(buses, lines)
print("\nFAST DECOUPLED METHOD")
print("Voltages (magnitude and angle) after convergence:")
for i, bus in enumerate(buses):
    print(f"Bus {bus['num']}: |V| = {V_fd[i]:.4f} p.u., θ = {delta_fd[i]:.4f}°")

# Run Newton-Raphson Method
V_nr, delta_nr = newton_raphson(buses, lines)
print("\nNEWTON-RAPHSON METHOD")
print("Voltages (magnitude and angle) after convergence:")
for i, bus in enumerate(buses):
    print(f"Bus {bus['num']}: |V| = {V_nr[i]:.4f} p.u., θ = {delta_nr[i]:.4f}°")