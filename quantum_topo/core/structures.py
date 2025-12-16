"""
quantum_topo/core/structures.py
================================
Estruturas de dados fundamentais para o sistema de invariantes topológicos.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass
class TopologicalInvariant:
    """
    Representa um invariante topológico identificado.
    
    Invariantes são propriedades que se preservam sob deformações
    contínuas - exatamente o que queremos que sobreviva ao ruído.
    """
    name: str
    value: float
    distance: int  # code distance onde foi medido
    description: str
    preserved: bool = True
    scaling_factor: float = 1.0
    confidence: float = 1.0
    
    def scale_to(self, target_distance: int) -> 'TopologicalInvariant':
        """Projeta o invariante para outra escala."""
        ratio = target_distance / self.distance
        return TopologicalInvariant(
            name=self.name,
            value=self.value * (self.scaling_factor ** ratio),
            distance=target_distance,
            description=self.description,
            preserved=self.preserved,
            scaling_factor=self.scaling_factor,
            confidence=self.confidence * (0.95 ** abs(ratio - 1))
        )


@dataclass
class SyndromePattern:
    """Padrão de síndromes medido."""
    x_syndromes: List[Tuple[int, int]]
    z_syndromes: List[Tuple[int, int]]
    
    @property
    def total(self) -> int:
        return len(self.x_syndromes) + len(self.z_syndromes)
    
    @property
    def x_count(self) -> int:
        return len(self.x_syndromes)
    
    @property
    def z_count(self) -> int:
        return len(self.z_syndromes)
    
    def to_dict(self) -> Dict:
        return {'X': self.x_syndromes, 'Z': self.z_syndromes}


@dataclass
class CorrectionResult:
    """Resultado de uma correção."""
    correction_map: np.ndarray
    residual_errors: np.ndarray
    success: bool
    efficiency: float
    x_corrections: int
    z_corrections: int
    
    @property
    def total_corrections(self) -> int:
        return self.x_corrections + self.z_corrections


@dataclass
class ExperimentResult:
    """Resultado completo de um experimento."""
    distance: int
    error_rate: float
    success: bool
    efficiency: float
    syndrome: SyndromePattern
    correction: CorrectionResult
    invariants: List[TopologicalInvariant]
    timing_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingAnalysis:
    """Análise de como invariantes escalam entre dimensões."""
    source_distance: int
    target_distance: int
    invariants_tested: int
    invariants_preserved: int
    preservation_ratio: float
    scaling_law: str  # "linear", "quadratic", "logarithmic", etc.
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_stable(self) -> bool:
        """Retorna True se maioria dos invariantes se preserva."""
        return self.preservation_ratio > 0.7


@dataclass
class ThresholdResult:
    """Resultado de análise de threshold."""
    distance: int
    threshold_value: float
    confidence_interval: Tuple[float, float]
    samples: int
    method: str


class ErrorModel:
    """Modelo de erros configurável."""
    
    def __init__(self, 
                 p_x: float = 0.33,
                 p_z: float = 0.33, 
                 p_y: float = 0.34,
                 correlated: bool = False):
        """
        Args:
            p_x: Probabilidade relativa de erro X
            p_z: Probabilidade relativa de erro Z
            p_y: Probabilidade relativa de erro Y
            correlated: Se True, erros em qubits vizinhos são correlacionados
        """
        total = p_x + p_z + p_y
        self.p_x = p_x / total
        self.p_z = p_z / total
        self.p_y = p_y / total
        self.correlated = correlated
    
    def sample_error_type(self) -> int:
        """Amostra tipo de erro: 0=nenhum, 1=X, 2=Z, 3=Y"""
        r = np.random.random()
        if r < self.p_x:
            return 1
        elif r < self.p_x + self.p_z:
            return 2
        else:
            return 3
    
    @classmethod
    def depolarizing(cls) -> 'ErrorModel':
        """Modelo de despolarização padrão (X, Y, Z equiprováveis)."""
        return cls(p_x=1/3, p_z=1/3, p_y=1/3)
    
    @classmethod
    def bit_flip(cls) -> 'ErrorModel':
        """Apenas erros X (bit flip)."""
        return cls(p_x=1.0, p_z=0.0, p_y=0.0)
    
    @classmethod
    def phase_flip(cls) -> 'ErrorModel':
        """Apenas erros Z (phase flip)."""
        return cls(p_x=0.0, p_z=1.0, p_y=0.0)
