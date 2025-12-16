"""
quantum_topo/experiments/runner.py
===================================
Executor de experimentos de scaling topológico.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from ..core.structures import (
    ExperimentResult, 
    SyndromePattern,
    ScalingAnalysis,
    ThresholdResult
)
from ..core.surface_code import SurfaceCode
from ..decoders.mwpm import MWPMDecoder
from ..analysis.invariants import InvariantExtractor, ScalingAnalyzer


@dataclass
class ExperimentConfig:
    """Configuração de experimento."""
    distances: List[int] = None
    error_rates: np.ndarray = None
    samples_per_config: int = 100
    verbose: bool = True
    
    def __post_init__(self):
        if self.distances is None:
            self.distances = [3, 5, 7]
        if self.error_rates is None:
            self.error_rates = np.linspace(0.01, 0.15, 10)


class ExperimentRunner:
    """
    Executor de experimentos de Surface Code multi-escala.
    
    Objetivo: testar a hipótese de que invariantes topológicos
    identificados em baixa dimensão se preservam em alta dimensão.
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.extractor = InvariantExtractor()
        self.scaling_analyzer = ScalingAnalyzer()
        self.results = defaultdict(list)
        self.thresholds = {}
    
    def run_single_experiment(self,
                             distance: int,
                             error_rate: float) -> ExperimentResult:
        """
        Executa um único experimento.
        
        Args:
            distance: Distância do código
            error_rate: Taxa de erro
            
        Returns:
            Resultado do experimento
        """
        start_time = time.time()
        
        # Criar código e decoder
        code = SurfaceCode(distance)
        decoder = MWPMDecoder(code.lattice_size)
        
        # Aplicar erros
        errors = code.apply_errors(error_rate)
        
        # Medir síndromes
        syndrome = code.measure_syndrome(errors)
        
        # Decodificar
        correction_result = decoder.decode(syndrome)
        
        # Verificar sucesso
        has_logical_error = code.get_logical_error(errors, correction_result.correction_map)
        correction_result.success = not has_logical_error
        
        # Calcular eficiência
        n_errors = np.sum(errors > 0)
        residual = code._combine_pauli(errors, correction_result.correction_map)
        n_residual = np.sum(residual > 0)
        correction_result.efficiency = 1.0 - (n_residual / max(1, n_errors))
        correction_result.residual_errors = residual
        
        # Extrair invariantes
        invariants = self.extractor.extract_invariants(
            code, errors, syndrome, correction_result.correction_map
        )
        
        timing = (time.time() - start_time) * 1000
        
        result = ExperimentResult(
            distance=distance,
            error_rate=error_rate,
            success=correction_result.success,
            efficiency=correction_result.efficiency,
            syndrome=syndrome,
            correction=correction_result,
            invariants=invariants,
            timing_ms=timing
        )
        
        # Armazenar para análise de scaling
        self.results[distance].append(result)
        self.scaling_analyzer.add_result(result)
        
        return result
    
    def run_threshold_analysis(self, 
                              distance: int,
                              error_rates: Optional[np.ndarray] = None,
                              samples: int = None) -> ThresholdResult:
        """
        Executa análise de threshold para uma distância.
        
        Args:
            distance: Distância do código
            error_rates: Taxas de erro a testar
            samples: Amostras por taxa
            
        Returns:
            Resultado da análise de threshold
        """
        error_rates = error_rates if error_rates is not None else self.config.error_rates
        samples = samples or self.config.samples_per_config
        
        success_rates = []
        
        for rate in error_rates:
            successes = 0
            for _ in range(samples):
                result = self.run_single_experiment(distance, rate)
                if result.success:
                    successes += 1
            
            success_rate = successes / samples
            success_rates.append(success_rate)
            
            if self.config.verbose:
                print(f"  d={distance}, p={rate:.3f}: {success_rate:.1%}")
        
        # Encontrar threshold (onde success = 50%)
        threshold = self._find_threshold(error_rates, np.array(success_rates))
        
        result = ThresholdResult(
            distance=distance,
            threshold_value=threshold,
            confidence_interval=(threshold * 0.9, threshold * 1.1),  # Simplificado
            samples=samples * len(error_rates),
            method="interpolation"
        )
        
        self.thresholds[distance] = result
        return result
    
    def _find_threshold(self, 
                       error_rates: np.ndarray, 
                       success_rates: np.ndarray,
                       target: float = 0.5) -> float:
        """Encontra threshold por interpolação."""
        for i in range(len(success_rates) - 1):
            if success_rates[i] >= target and success_rates[i+1] < target:
                # Interpolação linear
                slope = (success_rates[i+1] - success_rates[i]) / (error_rates[i+1] - error_rates[i])
                threshold = error_rates[i] + (target - success_rates[i]) / slope
                return threshold
        
        # Se não cruzou 50%, retornar estimativa
        if success_rates[-1] >= target:
            return error_rates[-1]
        else:
            return error_rates[0]
    
    def run_scaling_experiment(self) -> Dict[str, ScalingAnalysis]:
        """
        Executa experimento completo de scaling.
        
        Roda experimentos em múltiplas distâncias e analisa
        como os invariantes escalam.
        
        Returns:
            Dict com análises de scaling entre pares de distâncias
        """
        print("="*60)
        print("EXPERIMENTO DE SCALING TOPOLÓGICO")
        print("="*60)
        
        # 1. Rodar experimentos para cada distância
        for distance in self.config.distances:
            print(f"\n📊 Distância d={distance}")
            print("-"*40)
            
            # Threshold analysis
            print("  Analisando threshold...")
            self.run_threshold_analysis(distance)
            
            # Mais amostras na região do threshold
            threshold = self.thresholds[distance].threshold_value
            focused_rates = np.linspace(
                max(0.01, threshold * 0.5),
                min(0.2, threshold * 1.5),
                5
            )
            
            print("  Coletando amostras adicionais...")
            for rate in focused_rates:
                for _ in range(self.config.samples_per_config // 2):
                    self.run_single_experiment(distance, rate)
        
        # 2. Analisar scaling entre pares
        print("\n" + "="*60)
        print("ANÁLISE DE SCALING")
        print("="*60)
        
        analyses = {}
        distances = sorted(self.config.distances)
        
        for i, d1 in enumerate(distances[:-1]):
            for d2 in distances[i+1:]:
                key = f"{d1}_to_{d2}"
                analysis = self.scaling_analyzer.analyze_scaling(d1, d2)
                analyses[key] = analysis
                
                print(f"\n  {d1} → {d2}:")
                print(f"    Testados: {analysis.invariants_tested}")
                print(f"    Preservados: {analysis.invariants_preserved}")
                print(f"    Razão: {analysis.preservation_ratio:.1%}")
                print(f"    Lei: {analysis.scaling_law}")
        
        # 3. Identificar invariantes robustos
        print("\n" + "="*60)
        print("INVARIANTES TOPOLÓGICOS IDENTIFICADOS")
        print("="*60)
        
        preserved = self.scaling_analyzer.get_preserved_invariants(threshold=0.7)
        if preserved:
            print("\n✅ Invariantes que se preservam através das escalas:")
            for name in preserved:
                stats = self.extractor.get_statistics().get(name, {})
                print(f"   • {name}: μ={stats.get('mean', 0):.4f}, σ={stats.get('std', 0):.4f}")
        else:
            print("\n⚠️  Nenhum invariante fortemente preservado encontrado.")
            print("   Isso pode indicar:")
            print("   - Amostras insuficientes")
            print("   - Definições de invariante precisam refinamento")
            print("   - A hipótese precisa ajuste")
        
        return analyses
    
    def get_summary(self) -> Dict:
        """Retorna resumo dos experimentos."""
        summary = {
            'total_experiments': sum(len(r) for r in self.results.values()),
            'distances_tested': list(self.results.keys()),
            'thresholds': {d: t.threshold_value for d, t in self.thresholds.items()},
            'invariant_stats': self.extractor.get_statistics(),
            'preserved_invariants': self.scaling_analyzer.get_preserved_invariants()
        }
        return summary


def run_quick_test():
    """Teste rápido do sistema."""
    print("🚀 TESTE RÁPIDO")
    print("="*40)
    
    config = ExperimentConfig(
        distances=[3, 5],
        error_rates=np.array([0.02, 0.05, 0.08, 0.10]),
        samples_per_config=20,
        verbose=True
    )
    
    runner = ExperimentRunner(config)
    analyses = runner.run_scaling_experiment()
    
    return runner, analyses


def run_full_experiment():
    """Experimento completo."""
    print("🔬 EXPERIMENTO COMPLETO")
    print("="*40)
    
    config = ExperimentConfig(
        distances=[3, 5, 7],
        error_rates=np.linspace(0.01, 0.12, 8),
        samples_per_config=100,
        verbose=True
    )
    
    runner = ExperimentRunner(config)
    analyses = runner.run_scaling_experiment()
    
    return runner, analyses
