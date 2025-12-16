"""
quantum_topo/core/surface_code.py
==================================
Implementação do Surface Code em múltiplas escalas.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .structures import SyndromePattern, ErrorModel


class SurfaceCode:
    """
    Implementação do Rotated Surface Code.
    
    O Surface Code é um código topológico onde:
    - Qubits de dados ficam nas faces
    - Estabilizadores X medem produtos de X nos vértices
    - Estabilizadores Z medem produtos de Z nas plaquetas
    
    A propriedade topológica chave: erros só são detectáveis pelas
    bordas das cadeias de erro, não pelos erros individuais.
    """
    
    def __init__(self, distance: int, error_model: Optional[ErrorModel] = None):
        """
        Args:
            distance: Distância do código (d). Corrige até (d-1)/2 erros.
            error_model: Modelo de erros a usar. Default: despolarizante.
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance deve ser ímpar >= 3")
        
        self.d = distance
        self.lattice_size = 2 * distance - 1
        self.n_data_qubits = distance ** 2
        self.n_total_positions = self.lattice_size ** 2
        self.error_model = error_model or ErrorModel.depolarizing()
        
        # Construir estruturas
        self._build_lattice()
        self._build_stabilizers()
    
    def _build_lattice(self):
        """Constrói mapeamento de posições do lattice."""
        self.qubit_positions = {}
        self.position_to_index = {}
        
        idx = 0
        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                self.qubit_positions[idx] = (i, j)
                self.position_to_index[(i, j)] = idx
                idx += 1
    
    def _build_stabilizers(self):
        """Constrói estabilizadores X e Z."""
        self.x_stabilizers = {}  # posição -> lista de qubits vizinhos
        self.z_stabilizers = {}
        
        # Categorização Bulk vs Boundary
        # Bulk: 4 vizinhos
        # Boundary: < 4 vizinhos
        self.x_bulk_positions = set()
        self.z_bulk_positions = set()
        self.x_boundary_positions = set()
        self.z_boundary_positions = set()
        
        # Estabilizadores X: linhas ímpares, colunas pares
        x_idx = 0
        for i in range(1, self.lattice_size, 2):
            for j in range(0, self.lattice_size, 2):
                neighbors = self._get_neighbors(i, j)
                if neighbors:
                    self.x_stabilizers[(i, j)] = neighbors
                    if len(neighbors) == 4:
                        self.x_bulk_positions.add((i, j))
                    else:
                        self.x_boundary_positions.add((i, j))
                    x_idx += 1
        
        # Estabilizadores Z: linhas pares, colunas ímpares
        z_idx = 0
        for i in range(0, self.lattice_size, 2):
            for j in range(1, self.lattice_size, 2):
                neighbors = self._get_neighbors(i, j)
                if neighbors:
                    self.z_stabilizers[(i, j)] = neighbors
                    if len(neighbors) == 4:
                        self.z_bulk_positions.add((i, j))
                    else:
                        self.z_boundary_positions.add((i, j))
                    z_idx += 1
        
        self.n_x_stabilizers = len(self.x_stabilizers)
        self.n_z_stabilizers = len(self.z_stabilizers)
    
    def _get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Retorna vizinhos válidos de uma posição."""
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.lattice_size and 0 <= nj < self.lattice_size:
                neighbors.append((ni, nj))
        return neighbors
    
    def apply_errors(self, error_rate: float) -> np.ndarray:
        """
        Aplica erros aleatórios ao lattice.
        
        Args:
            error_rate: Probabilidade de erro por qubit
            
        Returns:
            Array 2D com erros: 0=nenhum, 1=X, 2=Z, 3=Y
        """
        errors = np.zeros((self.lattice_size, self.lattice_size), dtype=int)
        
        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                if np.random.random() < error_rate:
                    errors[i, j] = self.error_model.sample_error_type()
        
        return errors
    
    def measure_syndrome(self, errors: np.ndarray) -> SyndromePattern:
        """
        Mede síndromes dado um padrão de erros.
        
        Síndromes são as "bordas" das cadeias de erro:
        - Estabilizador X detecta erros Z e Y (têm componente Z)
        - Estabilizador Z detecta erros X e Y (têm componente X)
        
        Args:
            errors: Array de erros
            
        Returns:
            Padrão de síndromes medido
        """
        x_syndromes = []
        z_syndromes = []
        
        # Medir estabilizadores X (detectam Z e Y)
        for pos, neighbors in self.x_stabilizers.items():
            parity = 0
            for ni, nj in neighbors:
                if errors[ni, nj] in [2, 3]:  # Z ou Y
                    parity ^= 1
            if parity:
                x_syndromes.append(pos)
        
        # Medir estabilizadores Z (detectam X e Y)
        for pos, neighbors in self.z_stabilizers.items():
            parity = 0
            for ni, nj in neighbors:
                if errors[ni, nj] in [1, 3]:  # X ou Y
                    parity ^= 1
            if parity:
                z_syndromes.append(pos)
        
        return SyndromePattern(x_syndromes, z_syndromes)
    
    def get_logical_error(self, errors: np.ndarray, corrections: np.ndarray) -> bool:
        """
        Verifica se há erro lógico após correção.
        
        Erro lógico ocorre quando a cadeia de erro+correção forma
        um caminho não-trivial através do toro (atravessa o código).
        
        Args:
            errors: Erros originais
            corrections: Correções aplicadas
            
        Returns:
            True se há erro lógico
        """
        # Combinar erros e correções (mod 2 na álgebra de Pauli)
        residual = self._combine_pauli(errors, corrections)
        
        # Verificar se há cadeia não-trivial
        # Para Surface Code: verificar paridade ao longo das bordas
        
        # Verificar operador lógico X (cadeia Z horizontal)
        z_parity = 0
        for j in range(self.lattice_size):
            if residual[0, j] in [2, 3]:  # Z ou Y
                z_parity ^= 1
        
        # Verificar operador lógico Z (cadeia X vertical)
        x_parity = 0
        for i in range(self.lattice_size):
            if residual[i, 0] in [1, 3]:  # X ou Y
                x_parity ^= 1
        
        return bool(x_parity or z_parity)
    
    def _combine_pauli(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Combina dois arrays de operadores Pauli."""
        # Tabela de multiplicação Pauli (simplificada, ignorando fases)
        # I=0, X=1, Z=2, Y=3
        # X*X=I, Z*Z=I, Y*Y=I, X*Z=Y, Z*X=Y, etc.
        
        result = np.zeros_like(a)
        
        multiplication = {
            (0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3,
            (1, 0): 1, (1, 1): 0, (1, 2): 3, (1, 3): 2,
            (2, 0): 2, (2, 1): 3, (2, 2): 0, (2, 3): 1,
            (3, 0): 3, (3, 1): 2, (3, 2): 1, (3, 3): 0,
        }
        
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                result[i, j] = multiplication[(a[i, j], b[i, j])]
        
        return result
    
    def get_stabilizer_graph(self) -> Dict:
        """
        Retorna grafo de conectividade dos estabilizadores.
        Útil para análise topológica.
        """
        graph = {
            'x_stabilizers': dict(self.x_stabilizers),
            'z_stabilizers': dict(self.z_stabilizers),
            'n_x': self.n_x_stabilizers,
            'n_z': self.n_z_stabilizers,
            'distance': self.d,
            'lattice_size': self.lattice_size
        }
        return graph
    
    def __repr__(self):
        return f"SurfaceCode(d={self.d}, qubits={self.n_data_qubits}, stabilizers={self.n_x_stabilizers + self.n_z_stabilizers})"
