"""
Backend STIM para Rotated Surface Code.
Gera circuitos de altíssima performance para simulação de corretores de erro.
"""
import stim
import numpy as np
from typing import List, Dict, Tuple
from ..core.rotated_surface import RotatedSurfaceCode

class RotatedStimBackend:
    """
    Simulador de Circuito usando Stim.
    Suporta:
    - Geração de circuito com coordenadas.
    - Modelo de ruído (depolarizing noise).
    - Detectores e Observáveis lógicos automáticos.
    """
    
    def __init__(self, code: RotatedSurfaceCode):
        self.code = code
        self.d = code.d
        self.circuit = stim.Circuit()
        
        # Mapeamento de Qubits para Inteiros (Stim usa índices 0..N)
        # Data Qubits: 0 .. num_qubits-1
        # X-Ancillas: num_qubits .. +num_x_ancillas
        # Z-Ancillas: ... .. +num_z_ancillas
        
        self.data_start = 0
        self.x_anc_start = code.num_qubits
        self.z_anc_start = self.x_anc_start + len(code.x_ancillas)
        
        self.coord_to_idx = {}
        
        # Data Qubits
        for idx, coord in enumerate(code.data_qubits):
            self.coord_to_idx[coord] = self.data_start + idx
            
        # X Ancillas
        for i, coord in enumerate(code.x_ancillas.keys()):
            self.coord_to_idx[coord] = self.x_anc_start + i
            
        # Z Ancillas
        for i, coord in enumerate(code.z_ancillas.keys()):
            self.coord_to_idx[coord] = self.z_anc_start + i
            
    def _add_coords(self):
        """Adiciona anotações de coordenadas para visualização."""
        for coord, idx in self.coord_to_idx.items():
            r, c = coord
            self.circuit.append("QUBIT_COORDS", [idx], [r, c])
            
    def generate_circuit(self, rounds: int, noise: float = 0.001, crosstalk_strength: float = 0.0, bad_qubits: Dict[int, float] = None):
        """
        Gera o circuito completo com N rounds e ruído.
        Args:
            bad_qubits: Dicionário mapeando índice do qubit -> taxa de erro aumentada.
        """
        if bad_qubits is None:
            bad_qubits = {}
            
        self.circuit = stim.Circuit()
        self._add_coords()
        
        # Índices úteis
        data_indices = list(range(self.data_start, self.x_anc_start))
        x_anc_indices = list(range(self.x_anc_start, self.z_anc_start))
        z_anc_indices = list(range(self.z_anc_start, self.z_anc_start + len(self.code.z_ancillas)))
        all_anc_indices = x_anc_indices + z_anc_indices
        
        # Helper para aplicar ruído heterogêneo
        def append_noise_1q(instruction, targets, p_base):
            # Separar alvos normais e ruins
            good_targets = [t for t in targets if t not in bad_qubits]
            
            # Aplicar ruído base nos bons
            if good_targets and p_base > 0:
                self.circuit.append(instruction, good_targets, p_base)
                
            # Aplicar ruído específico nos ruins que estão nesta lista de alvos
            for t in targets:
                if t in bad_qubits:
                    p_bad = bad_qubits[t]
                    if p_bad > 0:
                        self.circuit.append(instruction, [t], p_bad)

        def append_noise_2q(instruction, t1, t2, p_base):
            # Se qualquer um for ruim, o par é ruim? 
            # Modelo simples: usa o MAX do erro dos dois ou p_base se ambos bons.
            # Se um é ruim, a porta gate que toca nele falha mais.
            
            p_eff = p_base
            is_bad = False
            
            if t1 in bad_qubits: 
                p_eff = max(p_eff, bad_qubits[t1])
                is_bad = True
            if t2 in bad_qubits:
                p_eff = max(p_eff, bad_qubits[t2])
                is_bad = True
                
            if p_eff > 0:
                self.circuit.append(instruction, [t1, t2], p_eff)
        
        # Helper para Ordenação CNOT (TR, BR, TL, BL)
        def get_neighbor_priority(anc_coord, data_idx):
            data_coord = self.code.data_qubits[data_idx - self.data_start]
            r_anc, c_anc = anc_coord
            r_data, c_data = data_coord
            dr = r_data - r_anc
            dc = c_data - c_anc
            if dr == 0 and dc == 1: return 0 
            if dr == 1 and dc == 1: return 1 
            if dr == 0 and dc == 0: return 2 
            if dr == 1 and dc == 0: return 3 
            return 99

        # Reset inicial de dados
        self.circuit.append("R", data_indices)
        
        # Loop de rounds
        for r in range(rounds):
            self.circuit.append("TICK")
            
            # 1. Reset Ancillas
            self.circuit.append("R", all_anc_indices)
            append_noise_1q("DEPOLARIZE1", all_anc_indices, noise)
            
            # 2. H nos X-Ancillas
            self.circuit.append("H", x_anc_indices)
            append_noise_1q("DEPOLARIZE1", x_anc_indices, noise)
                
            # 3. CNOTs (Scheduled)
            # X-Ancillas (Control) -> Data (Target)
            for coord, neighbors in self.code.x_ancillas.items():
                anc_idx = self.coord_to_idx[coord]
                sorted_neighs = sorted(neighbors, key=lambda i: get_neighbor_priority(coord, self.data_start + i))
                
                for data_idx_local in sorted_neighs:
                    data_idx = self.data_start + data_idx_local
                    self.circuit.append("CNOT", [anc_idx, data_idx])
                    append_noise_2q("DEPOLARIZE2", anc_idx, data_idx, noise)

            # Z-Ancillas (Target) <- Data (Control)
            for coord, neighbors in self.code.z_ancillas.items():
                anc_idx = self.coord_to_idx[coord]
                sorted_neighs = sorted(neighbors, key=lambda i: get_neighbor_priority(coord, self.data_start + i))
                
                for data_idx_local in sorted_neighs:
                    data_idx = self.data_start + data_idx_local
                    self.circuit.append("CNOT", [data_idx, anc_idx])
                    append_noise_2q("DEPOLARIZE2", data_idx, anc_idx, noise)
            
            # CROSSTALK (ZZ)
            if crosstalk_strength > 0:
                for row in range(self.d):
                    for col in range(self.d):
                        if (row, col) in self.code.grid:
                            q1 = self.code.grid[(row, col)] + self.data_start
                            if (row, col+1) in self.code.grid:
                                q2 = self.code.grid[(row, col+1)] + self.data_start
                                self.circuit.append("CORRELATED_ERROR", [stim.target_z(q1), stim.target_z(q2)], crosstalk_strength)
                            if (row+1, col) in self.code.grid:
                                q2 = self.code.grid[(row+1, col)] + self.data_start
                                self.circuit.append("CORRELATED_ERROR", [stim.target_z(q1), stim.target_z(q2)], crosstalk_strength)

            # 4. H nos X-Ancillas
            self.circuit.append("H", x_anc_indices)
            append_noise_1q("DEPOLARIZE1", x_anc_indices, noise)

            # 5. Measure Ancillas
            self.circuit.append("M", all_anc_indices)
            append_noise_1q("X_ERROR", all_anc_indices, noise) # Readout error
            
            # 6. Definir Detectores
            n_x = len(x_anc_indices)
            n_z = len(z_anc_indices)
            total_anc = n_x + n_z
            
            # X-Ancillas: Comparar Round N com N-1. (Round 0 random -> skip)
            for i in range(n_x):
                # rec index from end: -total_anc + i
                rec_idx = -total_anc + i
                if r > 0:
                    self.circuit.append("DETECTOR", [stim.target_rec(rec_idx), stim.target_rec(rec_idx - total_anc)], 
                                      [self.code.data_qubits[0][0], self.code.data_qubits[0][1], r]) 
                
            # Z-Ancillas: Comparar Round N com N-1. (Round 0 deterministic 0 -> compare with 0)
            for i in range(n_z):
                rec_idx = -total_anc + n_x + i
                if r > 0:
                    self.circuit.append("DETECTOR", [stim.target_rec(rec_idx), stim.target_rec(rec_idx - total_anc)],
                                      [0, 0, r])
                else: 
                     # Round 0: Valid Z-checks should be 0.
                     self.circuit.append("DETECTOR", [stim.target_rec(rec_idx)], [0,0,r])

        # Measure Data Qubits at the end of the experiment (Z-basis)
        self.circuit.append("M", data_indices)
        
        # 7. Final Detectors (Time Boundary for Z-Ancillas)
        # Compare last Z-ancilla measurement with parity of data qubits.
        # This closes the "time loop" for Z-stabilizers.
        # (X-stabilizers are randomized by Z-measures, so no check).
        
        num_data = len(data_indices)
        n_x = len(x_anc_indices)
        n_z = len(z_anc_indices)
        total_anc = n_x + n_z
        
        # Iterate Z ancillas in order
        for i, (coord, neighbors) in enumerate(self.code.z_ancillas.items()):
            # Last measurement of this ancilla:
            # It was in the block of 'total_anc' measurements just before 'num_data' measurements.
            # Its index in that block was n_x + i.
            # So relative to NOW (end): -(num_data) - (total_anc) + (n_x + i)
            # = -(num_data + total_anc - n_x - i)
            # = -(num_data + n_z - i)
            
            anc_rec_idx = -(num_data + n_z - i)
            
            # Data parity:
            data_recs = []
            for data_local_idx in neighbors:
                # data_local_idx is 0..num_data-1
                # Rec index is -(num_data) + data_local_idx
                # Wait, rec[-1] is last. rec[-num_data] is first.
                # So relative index = data_local_idx - num_data
                data_recs.append(stim.target_rec(data_local_idx - num_data))
                
            # Detector: Ancilla XOR DataParity
            self.circuit.append("DETECTOR", [stim.target_rec(anc_rec_idx)] + data_recs, [0,0,999])

        # Definir Observável Lógico (Logical Z)
        num_data_measured = len(data_indices)
        
        logic_recs = []
        if hasattr(self.code, 'z_logicals') and self.code.z_logicals:
            for coord in self.code.z_logicals[0]:
                if coord in self.code.grid:
                    data_q_idx = self.code.grid[coord] 
                    global_idx = self.data_start + data_q_idx
                    local_idx = global_idx - self.data_start
                    lookback = local_idx - num_data_measured
                    logic_recs.append(stim.target_rec(lookback))
        
        self.circuit.append("OBSERVABLE_INCLUDE", logic_recs, 0)

    def get_dem(self):
        return self.circuit.detector_error_model()

    def sample_shots(self, shots: int):
        sampler = self.circuit.compile_sampler()
        return sampler.sample(shots=shots)
        
    def sample_detector_syndrome(self, shots: int):
        """Retorna apenas os eventos de detecção (síndromes) e sucesso lógico."""
        # Usa compile_detector_sampler
        sampler = self.circuit.compile_detector_sampler()
        return sampler.sample(shots=shots, append_observables=True)
