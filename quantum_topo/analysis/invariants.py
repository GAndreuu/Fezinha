"""
quantum_topo/analysis/invariants.py
=====================================
Identificação e rastreamento de invariantes topológicos.

Este é o módulo central da hipótese:
Invariantes em baixa dimensão devem se preservar em projeções para alta dimensão.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from ..core.structures import (
    TopologicalInvariant, 
    SyndromePattern, 
    ExperimentResult,
    ScalingAnalysis
)
from ..core.surface_code import SurfaceCode


class InvariantExtractor:
    """
    Extrai invariantes topológicos de experimentos com Surface Code.
    
    Invariantes candidatos:
    1. Razão síndromes/erros (como estrutura local vira sinal global)
    2. Conectividade do grafo de síndromes
    3. Distribuição de distâncias entre síndromes pareadas
    4. Entropia da distribuição de correções
    5. Homologia do padrão de erros (ciclos não-triviais)
    """
    
    def __init__(self):
        self.invariant_history = defaultdict(list)
    
    def extract_invariants(self, 
                          code: SurfaceCode,
                          errors: np.ndarray,
                          syndrome: SyndromePattern,
                          correction: np.ndarray) -> List[TopologicalInvariant]:
        """
        Extrai todos os invariantes de um experimento.
        
        Args:
            code: Instância do Surface Code
            errors: Padrão de erros
            syndrome: Síndromes medidas
            correction: Correções aplicadas
            
        Returns:
            Lista de invariantes identificados
        """
        invariants = []
        
        # 1. Razão síndromes/erros
        invariants.append(self._extract_syndrome_ratio(code, errors, syndrome))
        
        # 2. Densidade de correção
        invariants.append(self._extract_correction_density(code, errors, correction))
        
        # 3. Correlação espacial de síndromes
        invariants.append(self._extract_syndrome_correlation(code, syndrome))
        
        # 4. Eficiência de detecção por tipo
        invariants.extend(self._extract_detection_efficiency(code, errors, syndrome))
        
        # 5. Característica de Euler do grafo de síndromes
        invariants.append(self._extract_euler_characteristic(code, syndrome))
        
        # 6. Invariante de paridade (fundamental para topologia)
        invariants.extend(self._extract_parity_invariants(code, syndrome))
        
        # Armazenar histórico
        for inv in invariants:
            self.invariant_history[inv.name].append(inv.value)
        
        return invariants
    
    def _extract_syndrome_ratio(self, code: SurfaceCode, 
                                errors: np.ndarray,
                                syndrome: SyndromePattern) -> TopologicalInvariant:
        """
        Razão entre síndromes e erros.
        
        Propriedade topológica: síndromes são BORDAS de cadeias de erro.
        Teoricamente, para erros isolados: #síndromes ≈ 2 * #erros
        Para cadeias: #síndromes ≈ 2 (só as pontas)
        """
        n_errors = np.sum(errors > 0)
        n_syndromes = syndrome.total
        
        if n_errors == 0:
            ratio = 0.0
        else:
            ratio = n_syndromes / n_errors
        
        return TopologicalInvariant(
            name="syndrome_error_ratio",
            value=ratio,
            distance=code.d,
            description="Razão síndromes/erros - mede localidade vs cadeias",
            scaling_factor=1.0  # Deve ser invariante de escala
        )
    
    def _extract_correction_density(self, code: SurfaceCode,
                                    errors: np.ndarray,
                                    correction: np.ndarray) -> TopologicalInvariant:
        """
        Densidade de correções relativa ao tamanho do código.
        """
        n_corrections = np.sum(correction > 0)
        density = n_corrections / code.n_total_positions
        
        return TopologicalInvariant(
            name="correction_density",
            value=density,
            distance=code.d,
            description="Fração do lattice que recebeu correção",
            scaling_factor=1.0
        )
    
    def _extract_syndrome_correlation(self, code: SurfaceCode,
                                      syndrome: SyndromePattern) -> TopologicalInvariant:
        """
        Correlação espacial entre síndromes.
        
        Mede o quão "espalhadas" estão as síndromes.
        Cadeias longas: síndromes distantes
        Erros isolados: síndromes próximas
        """
        all_syndromes = syndrome.x_syndromes + syndrome.z_syndromes
        
        if len(all_syndromes) < 2:
            return TopologicalInvariant(
                name="syndrome_correlation",
                value=0.0,
                distance=code.d,
                description="Distância média entre síndromes (normalizada)",
                scaling_factor=1.0
            )
        
        # Calcular distância média entre todas as síndromes
        total_dist = 0
        count = 0
        for i, s1 in enumerate(all_syndromes):
            for s2 in all_syndromes[i+1:]:
                dist = abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])  # Manhattan
                total_dist += dist
                count += 1
        
        avg_dist = total_dist / count if count > 0 else 0
        normalized = avg_dist / code.lattice_size  # Normalizar pela escala
        
        return TopologicalInvariant(
            name="syndrome_correlation",
            value=normalized,
            distance=code.d,
            description="Distância média entre síndromes (normalizada)",
            scaling_factor=1.0  # Normalização deve tornar invariante
        )
    
    def _extract_detection_efficiency(self, code: SurfaceCode,
                                      errors: np.ndarray,
                                      syndrome: SyndromePattern) -> List[TopologicalInvariant]:
        """
        Eficiência de detecção por tipo de erro.
        """
        invariants = []
        
        # Contar erros por tipo
        x_errors = np.sum(errors == 1)
        z_errors = np.sum(errors == 2)
        y_errors = np.sum(errors == 3)
        
        # Eficiência X (detectados por estabilizadores Z)
        if x_errors + y_errors > 0:
            x_eff = len(syndrome.z_syndromes) / (x_errors + y_errors)
        else:
            x_eff = 0.0
        
        invariants.append(TopologicalInvariant(
            name="x_detection_efficiency",
            value=min(x_eff, 2.0),  # Cap em 2 (máximo teórico)
            distance=code.d,
            description="Eficiência de detecção de erros X",
            scaling_factor=1.0
        ))
        
        # Eficiência Z (detectados por estabilizadores X)
        if z_errors + y_errors > 0:
            z_eff = len(syndrome.x_syndromes) / (z_errors + y_errors)
        else:
            z_eff = 0.0
        
        invariants.append(TopologicalInvariant(
            name="z_detection_efficiency",
            value=min(z_eff, 2.0),
            distance=code.d,
            description="Eficiência de detecção de erros Z",
            scaling_factor=1.0
        ))
        
        return invariants
    
    def _extract_euler_characteristic(self, code: SurfaceCode,
                                      syndrome: SyndromePattern) -> TopologicalInvariant:
        """
        Característica de Euler do "grafo" de síndromes.
        
        χ = V - E + F
        
        Para síndromes: V = #síndromes, E = #conexões, F = #faces fechadas
        É um invariante topológico clássico.
        """
        all_syndromes = syndrome.x_syndromes + syndrome.z_syndromes
        n_vertices = len(all_syndromes)
        
        if n_vertices == 0:
            return TopologicalInvariant(
                name="euler_characteristic",
                value=0.0,
                distance=code.d,
                description="Característica de Euler normalizada",
                scaling_factor=1.0
            )
        
        # Contar "arestas" (síndromes vizinhas)
        n_edges = 0
        syndrome_set = set(all_syndromes)
        for s in all_syndromes:
            for di, dj in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                neighbor = (s[0] + di, s[1] + dj)
                if neighbor in syndrome_set:
                    n_edges += 1
        n_edges //= 2  # Cada aresta contada duas vezes
        
        # Característica de Euler simplificada (sem faces)
        euler = n_vertices - n_edges
        
        # Normalizar pelo número de estabilizadores
        normalized = euler / (code.n_x_stabilizers + code.n_z_stabilizers)
        
        return TopologicalInvariant(
            name="euler_characteristic",
            value=normalized,
            distance=code.d,
            description="Característica de Euler normalizada",
            scaling_factor=1.0
        )
    
    def _extract_parity_invariants(self, code: SurfaceCode, 
                                   syndrome: SyndromePattern) -> List[TopologicalInvariant]:
        """
        Extrai invariantes de paridade (Bulk vs Boundary).
        """
        invariants = []
        
        # 1. Bulk Parity (Deve ser ZERO)
        # Verificar paridade apenas de estabilizadores no bulk
        bulk_violation = 0
        
        # X Bulk
        x_bulk_syndromes = [s for s in syndrome.x_syndromes if s in code.x_bulk_positions]
        if len(x_bulk_syndromes) % 2 != 0:
            bulk_violation += 1
            
        # Z Bulk
        z_bulk_syndromes = [s for s in syndrome.z_syndromes if s in code.z_bulk_positions]
        if len(z_bulk_syndromes) % 2 != 0:
            bulk_violation += 1
            
        invariants.append(TopologicalInvariant(
            name="bulk_parity",
            value=float(bulk_violation),
            distance=0,
            description="Violação de paridade no bulk (Deve ser 0)",
            scaling_factor=1.0,
            preserved=True
        ))
        
        # 2. Boundary Flux (Atividade nas bordas)
        x_boundary_count = len([s for s in syndrome.x_syndromes if s in code.x_boundary_positions])
        z_boundary_count = len([s for s in syndrome.z_syndromes if s in code.z_boundary_positions])
        
        # Normalizar pelo perímetro (~ 4*d)
        perimeter = 4 * code.d
        flux = (x_boundary_count + z_boundary_count) / perimeter
        
        invariants.append(TopologicalInvariant(
            name="boundary_flux",
            value=flux,
            distance=code.d,
            description="Fluxo de síndromes nas bordas",
            scaling_factor=1.0,  # Deve ser constante no regime topológico?
            preserved=True
        ))
        
        return invariants
    
    def get_statistics(self) -> Dict[str, Dict]:
        """Retorna estatísticas dos invariantes coletados."""
        stats = {}
        for name, values in self.invariant_history.items():
            if values:
                stats[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'samples': len(values)
                }
        return stats


class ScalingAnalyzer:
    """
    Analisa como invariantes escalam entre diferentes distâncias de código.
    
    Hipótese central: se um invariante é verdadeiramente topológico,
    ele deve se preservar (ou escalar previsivelmente) entre d=3, d=5, d=7.
    """
    
    def __init__(self):
        self.results_by_distance = defaultdict(list)
    
    def add_result(self, result: ExperimentResult):
        """Adiciona resultado de experimento."""
        self.results_by_distance[result.distance].append(result)
    
    def analyze_scaling(self, 
                       source_d: int, 
                       target_d: int,
                       min_samples: int = 10) -> ScalingAnalysis:
        """
        Analisa como invariantes escalam de source_d para target_d.
        
        Args:
            source_d: Distância fonte (ex: 3)
            target_d: Distância alvo (ex: 5)
            min_samples: Mínimo de amostras para análise
            
        Returns:
            Análise de scaling
        """
        source_results = self.results_by_distance.get(source_d, [])
        target_results = self.results_by_distance.get(target_d, [])
        
        if len(source_results) < min_samples or len(target_results) < min_samples:
            return ScalingAnalysis(
                source_distance=source_d,
                target_distance=target_d,
                invariants_tested=0,
                invariants_preserved=0,
                preservation_ratio=0.0,
                scaling_law="insufficient_data",
                details={"error": "Amostras insuficientes"}
            )
        
        # Coletar invariantes por nome
        source_invariants = self._collect_invariants(source_results)
        target_invariants = self._collect_invariants(target_results)
        
        # Comparar cada invariante
        preserved = 0
        tested = 0
        details = {}
        
        for name in source_invariants.keys():
            if name not in target_invariants:
                continue
            
            tested += 1
            source_vals = source_invariants[name]
            target_vals = target_invariants[name]
            
            # Calcular estatísticas
            source_mean = np.mean(source_vals)
            source_std = np.std(source_vals)
            target_mean = np.mean(target_vals)
            target_std = np.std(target_vals)
            
            # Verificar preservação (dentro de 2 desvios padrão)
            tolerance = 2 * max(source_std, target_std, 0.01)
            is_preserved = abs(source_mean - target_mean) < tolerance
            
            if is_preserved:
                preserved += 1
            
            details[name] = {
                'source_mean': source_mean,
                'source_std': source_std,
                'target_mean': target_mean,
                'target_std': target_std,
                'preserved': is_preserved,
                'ratio': target_mean / source_mean if source_mean != 0 else 0
            }
        
        # Determinar lei de scaling
        scaling_law = self._infer_scaling_law(details, source_d, target_d)
        
        return ScalingAnalysis(
            source_distance=source_d,
            target_distance=target_d,
            invariants_tested=tested,
            invariants_preserved=preserved,
            preservation_ratio=preserved / tested if tested > 0 else 0,
            scaling_law=scaling_law,
            details=details
        )
    
    def _collect_invariants(self, 
                           results: List[ExperimentResult]) -> Dict[str, List[float]]:
        """Coleta valores de invariantes dos resultados."""
        collected = defaultdict(list)
        for result in results:
            for inv in result.invariants:
                collected[inv.name].append(inv.value)
        return collected
    
    def _infer_scaling_law(self, details: Dict, source_d: int, target_d: int) -> str:
        """Infere a lei de scaling dos invariantes."""
        ratios = []
        for name, data in details.items():
            if data.get('ratio', 0) > 0:
                ratios.append(data['ratio'])
        
        if not ratios:
            return "unknown"
        
        avg_ratio = np.mean(ratios)
        expected_linear = target_d / source_d
        expected_quadratic = (target_d / source_d) ** 2
        
        # Verificar qual lei se ajusta melhor
        if abs(avg_ratio - 1.0) < 0.2:
            return "constant"  # Invariante verdadeiro!
        elif abs(avg_ratio - expected_linear) < 0.2:
            return "linear"
        elif abs(avg_ratio - expected_quadratic) < 0.2:
            return "quadratic"
        else:
            return f"power_{np.log(avg_ratio) / np.log(target_d / source_d):.2f}"
    
    def get_preserved_invariants(self, threshold: float = 0.8) -> List[str]:
        """
        Retorna nomes dos invariantes que se preservam bem.
        
        Estes são os candidatos a "invariantes topológicos verdadeiros".
        """
        if len(self.results_by_distance) < 2:
            return []
        
        distances = sorted(self.results_by_distance.keys())
        preserved_counts = defaultdict(int)
        total_comparisons = 0
        
        for i, d1 in enumerate(distances[:-1]):
            for d2 in distances[i+1:]:
                analysis = self.analyze_scaling(d1, d2)
                total_comparisons += 1
                
                for name, data in analysis.details.items():
                    if data.get('preserved', False):
                        preserved_counts[name] += 1
        
        # Retornar invariantes preservados em pelo menos threshold das comparações
        return [
            name for name, count in preserved_counts.items()
            if count / total_comparisons >= threshold
        ]

    def find_lock_dimension(self) -> Optional[int]:
        """
        Identifica a 'Minimal Topological Dimension' (d_lock).
        
        Procura o ponto onde a variação dos invariantes cai abaixo de um limiar.
        Calcula a 'derivada' da preservação: (Inv(d) - Inv(d-step)).
        """
        distances = sorted(self.results_by_distance.keys())
        if len(distances) < 3:
            return None
            
        # Analisar estabilidade passo a passo
        stability_scores = {}
        
        for i in range(len(distances) - 1):
            d1 = distances[i]
            d2 = distances[i+1]
            analysis = self.analyze_scaling(d1, d2)
            
            # Score baseada na razão de invariantes preservados
            # E na 'força' da lei constante
            score = analysis.preservation_ratio
            if analysis.scaling_law == "constant":
                score += 0.5 # Bonus para lei constante
                
            stability_scores[d2] = score
            
        # Encontrar onde a estabilidade 'trava' (score alto e constante)
        # Ex: 3->5 (0.8), 5->7 (1.5), 7->9 (1.5) => Lock em 7?
        # Ou Lock em 5, pois a transição 5->7 já foi estável.
        
        for i in range(1, len(distances) - 1):
            d_curr = distances[i] # Ex: 5
            d_next = distances[i+1] # Ex: 7
            
            # Se a transição PARTINDO de d_curr é estável
            if stability_scores.get(d_next, 0) > 1.4: # > 90% preserved + constant law
                return d_curr
                
        return None
