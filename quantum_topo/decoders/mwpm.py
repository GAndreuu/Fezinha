"""
quantum_topo/decoders/mwpm.py
==============================
Decoder baseado em Minimum Weight Perfect Matching.
"""

import numpy as np
from scipy.sparse import lil_matrix
from typing import Dict, List, Tuple, Optional

from ..core.structures import SyndromePattern, CorrectionResult

# Tentar importar PyMatching
try:
    import pymatching
    PYMATCHING_AVAILABLE = True
except ImportError:
    PYMATCHING_AVAILABLE = False


class MWPMDecoder:
    """
    Decoder usando Minimum Weight Perfect Matching.
    
    MWPM encontra o emparelhamento de peso mínimo entre síndromes,
    o que corresponde à correção mais provável dado o modelo de erro.
    
    A beleza topológica: não precisamos saber ONDE estão os erros,
    apenas como conectar as síndromes de forma consistente.
    """
    
    def __init__(self, lattice_size: int):
        """
        Args:
            lattice_size: Tamanho do lattice (2*d - 1)
        """
        self.lattice_size = lattice_size
        self._build_index_maps()
        self._build_parity_matrices()
        self._initialize_matchers()
    
    def _build_index_maps(self):
        """Constrói mapeamentos de índices."""
        # Qubits
        self.qubit_map = {}
        idx = 0
        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                self.qubit_map[(i, j)] = idx
                idx += 1
        self.n_qubits = idx
        
        # Estabilizadores X
        self.x_stab_map = {}
        idx = 0
        for i in range(1, self.lattice_size, 2):
            for j in range(0, self.lattice_size, 2):
                self.x_stab_map[(i, j)] = idx
                idx += 1
        self.n_x_stabs = idx
        
        # Estabilizadores Z
        idx = 0
        for i in range(1, self.lattice_size, 2):
            for j in range(0, self.lattice_size, 2):
                self.x_stab_map[(i, j)] = idx
                idx += 1
        self.n_x_stabs = idx
        
        # Estabilizadores Z
        self.z_stab_map = {}
        idx = 0
        for i in range(0, self.lattice_size, 2):
            for j in range(1, self.lattice_size, 2):
                self.z_stab_map[(i, j)] = idx
                idx += 1
        self.n_z_stabs = idx

    def set_weights(self, weights: Dict[str, float] = None):
        """
        Define pesos personalizados para o matching.
        """
        self.custom_weights = weights
        self._initialize_matchers()
    
    def _build_parity_matrices(self):
        """Constrói matrizes de paridade para o matching."""
        # H_x: relaciona estabilizadores X com qubits
        H_x = lil_matrix((self.n_x_stabs, self.n_qubits), dtype=int)
        for (si, sj), stab_idx in self.x_stab_map.items():
            for ni, nj in self._neighbors(si, sj):
                H_x[stab_idx, self.qubit_map[(ni, nj)]] = 1
        
        # H_z: relaciona estabilizadores Z com qubits
        H_z = lil_matrix((self.n_z_stabs, self.n_qubits), dtype=int)
        for (si, sj), stab_idx in self.z_stab_map.items():
            for ni, nj in self._neighbors(si, sj):
                H_z[stab_idx, self.qubit_map[(ni, nj)]] = 1
        
        self.H_x = H_x.tocsr()
        self.H_z = H_z.tocsr()
    
    def _initialize_matchers(self):
        """Inicializa matchers do PyMatching."""
        self.matching_available = False
        
        if PYMATCHING_AVAILABLE:
            try:
                # Se tiver pesos customizados, usar
                # Nota: PyMatching aceita weights array correspondente às colunas de H (qubits)
                # ou edges. Aqui H relaciona estabilizadores (row) e qubits (col).
                # O peso deve ser associado ao ERRO (qubit).
                
                weights_x = None
                weights_z = None
                
                if hasattr(self, 'custom_weights') and self.custom_weights:
                    # Construir array de pesos para cada qubit
                    # Default weight = 1.0
                    w_arr = np.ones(self.n_qubits, dtype=float)
                    
                    # Aplicar pesos diferenciados
                    # self.custom_weights pode ter 'bulk' e 'boundary'
                    if 'boundary' in self.custom_weights:
                        w_bound = self.custom_weights['boundary']
                        # Identificar qubits de borda?
                        # Simplificação: Qubits nas bordas do lattice (i=0, i=max, etc)
                        for (i, j), idx in self.qubit_map.items():
                            if i == 0 or i == self.lattice_size-1 or j == 0 or j == self.lattice_size-1:
                                w_arr[idx] = w_bound
                    
                    if 'bulk' in self.custom_weights:
                        w_bulk = self.custom_weights['bulk']
                        for (i, j), idx in self.qubit_map.items():
                             if not (i == 0 or i == self.lattice_size-1 or j == 0 or j == self.lattice_size-1):
                                w_arr[idx] = w_bulk

                    weights_x = w_arr
                    weights_z = w_arr # Assumindo simetria X/Z nos erros por enquanto

                self.matcher_x = pymatching.Matching(self.H_x, weights=weights_x)
                self.matcher_z = pymatching.Matching(self.H_z, weights=weights_z)
                self.matching_available = True
            except Exception as e:
                print(f"⚠️ Erro inicializando PyMatching: {e}")
    
    def _neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Vizinhos válidos."""
        return [(i + di, j + dj) 
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= i + di < self.lattice_size and 0 <= j + dj < self.lattice_size]
    
    def decode(self, syndrome: SyndromePattern) -> CorrectionResult:
        """
        Decodifica síndromes e retorna correções.
        
        Args:
            syndrome: Padrão de síndromes medido
            
        Returns:
            Resultado da correção
        """
        if self.matching_available:
            return self._decode_mwpm(syndrome)
        else:
            return self._decode_fallback(syndrome)
    
    def _decode_mwpm(self, syndrome: SyndromePattern) -> CorrectionResult:
        """Decodificação via MWPM."""
        correction = np.zeros((self.lattice_size, self.lattice_size), dtype=int)
        
        try:
            # Síndromes X → Correções Z
            if syndrome.x_syndromes:
                syn_vec_x = self._syndrome_to_vector(syndrome.x_syndromes, 'X')
                if syn_vec_x is not None and np.any(syn_vec_x):
                    corr_vec = self.matcher_x.decode(syn_vec_x)
                    self._apply_correction(correction, corr_vec, 'Z')
            
            # Síndromes Z → Correções X
            if syndrome.z_syndromes:
                syn_vec_z = self._syndrome_to_vector(syndrome.z_syndromes, 'Z')
                if syn_vec_z is not None and np.any(syn_vec_z):
                    corr_vec = self.matcher_z.decode(syn_vec_z)
                    self._apply_correction(correction, corr_vec, 'X')
        
        except Exception as e:
            print(f"⚠️ MWPM falhou, usando fallback: {e}")
            return self._decode_fallback(syndrome)
        
        return self._build_result(correction)
    
    def _decode_fallback(self, syndrome: SyndromePattern) -> CorrectionResult:
        """Decoder fallback simples."""
        correction = np.zeros((self.lattice_size, self.lattice_size), dtype=int)
        
        # Estratégia: para cada síndrome, corrigir um vizinho
        for pos in syndrome.x_syndromes:
            neighbors = self._neighbors(*pos)
            if neighbors:
                ni, nj = neighbors[0]
                if correction[ni, nj] == 0:
                    correction[ni, nj] = 2  # Correção Z
                elif correction[ni, nj] == 1:
                    correction[ni, nj] = 3  # X + Z = Y
        
        for pos in syndrome.z_syndromes:
            neighbors = self._neighbors(*pos)
            if neighbors:
                ni, nj = neighbors[0]
                if correction[ni, nj] == 0:
                    correction[ni, nj] = 1  # Correção X
                elif correction[ni, nj] == 2:
                    correction[ni, nj] = 3  # Z + X = Y
        
        return self._build_result(correction)
    
    def _syndrome_to_vector(self, syndromes: List[Tuple[int, int]], 
                            stab_type: str) -> Optional[np.ndarray]:
        """Converte lista de síndromes para vetor."""
        if stab_type == 'X':
            n_stabs = self.n_x_stabs
            stab_map = self.x_stab_map
        else:
            n_stabs = self.n_z_stabs
            stab_map = self.z_stab_map
        
        if n_stabs == 0:
            return None
        
        vec = np.zeros(n_stabs, dtype=int)
        for pos in syndromes:
            if pos in stab_map:
                vec[stab_map[pos]] = 1
        
        return vec
    
    def _apply_correction(self, correction: np.ndarray, 
                         corr_vector: np.ndarray, corr_type: str):
        """Aplica vetor de correção ao lattice."""
        pauli_combine = {
            'X': {0: 1, 1: 0, 2: 3, 3: 2},  # Aplica X
            'Z': {0: 2, 1: 3, 2: 0, 3: 1},  # Aplica Z
        }
        
        ops = pauli_combine[corr_type]
        for (i, j), qubit_idx in self.qubit_map.items():
            if qubit_idx < len(corr_vector) and corr_vector[qubit_idx] == 1:
                correction[i, j] = ops[correction[i, j]]
    
    def _build_result(self, correction: np.ndarray) -> CorrectionResult:
        """Constrói resultado da correção."""
        x_corr = np.sum((correction == 1) | (correction == 3))
        z_corr = np.sum((correction == 2) | (correction == 3))
        
        return CorrectionResult(
            correction_map=correction,
            residual_errors=np.zeros_like(correction),  # Calculado depois
            success=True,  # Determinado depois
            efficiency=0.0,  # Calculado depois
            x_corrections=int(x_corr),
            z_corrections=int(z_corr)
        )


class AdaptiveMWPMDecoder(MWPMDecoder):
    """
    Decoder com Estratégia de Platô Estendido (Absorbing Boundary).
    
    PLATÔ (p < 0.09): w_boundary = 0.3 (Borda aliada/reflexiva)
    ABSORÇÃO (p >= 0.09): w_boundary = 1.0 (Borda neutra/absorvente)
    
    Em vez de lutar contra a onda (w=1.5), permitimos que a borda
    absorva o excesso de entropia sem viés (w=1.0).
    """
    
    def __init__(self, lattice_size: int, 
                 transition_threshold: float = 0.12,
                 confirm_count: int = 3):
        super().__init__(lattice_size)
        self.transition_threshold = transition_threshold
        self.confirm_count = confirm_count
        
        self.state = 'plateau'
        self.pending_state = None
        self.pending_counter = 0
        self.state_history = []
        
        # Pesos
        self.weights_plateau = {'bulk': 1.0, 'boundary': 0.3}     # Trust
        self.weights_absorb = {'bulk': 1.0, 'boundary': 1.0}      # Neutral/Absorb
        
        self.set_weights(self.weights_plateau)
    
    def decode(self, syndrome: SyndromePattern) -> CorrectionResult:
        n_syndromes = syndrome.total
        n_stabilizers = self.n_x_stabs + self.n_z_stabs
        density = n_syndromes / max(1, n_stabilizers)
        
        # Lógica binária simples: Platô ou Absorção
        suggested = 'absorb' if density > self.transition_threshold else 'plateau'
        
        old_state = self.state
        transition = False
        
        if suggested != self.state:
            if suggested == self.pending_state:
                self.pending_counter += 1
                if self.pending_counter >= self.confirm_count:
                    self.state = suggested
                    self.pending_state = None
                    self.pending_counter = 0
                    transition = True
                    
                    if self.state == 'plateau':
                        self.set_weights(self.weights_plateau)
                    else:
                        self.set_weights(self.weights_absorb)
            else:
                self.pending_state = suggested
                self.pending_counter = 1
        else:
            self.pending_state = None
            self.pending_counter = 0
        
        self.state_history.append({
            'density': density,
            'state': self.state,
            'transition': transition
        })
        
        return super().decode(syndrome)
    
    def get_regime_stats(self) -> Dict:
        if not self.state_history:
            return {}
        
        plateau = sum(1 for s in self.state_history if s['state'] == 'plateau')
        absorb = sum(1 for s in self.state_history if s['state'] == 'absorb')
        transitions = sum(1 for s in self.state_history if s['transition'])
        total = len(self.state_history)
        
        return {
            'total_decodes': total,
            'plateau_pct': plateau / total if total else 0,
            'absorb_pct': absorb / total if total else 0,
            'inversions': transitions,
            'avg_density': np.mean([s['density'] for s in self.state_history]),
            'low_regime_pct': plateau / total if total else 0,
            'high_regime_pct': absorb / total if total else 0
        }


