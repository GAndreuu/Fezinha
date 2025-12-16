"""
quantum_topo/visualization/plots.py
=====================================
Visualização de resultados e análises.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Dict, List, Optional, Tuple

from ..core.structures import (
    SyndromePattern, 
    ExperimentResult, 
    ScalingAnalysis,
    TopologicalInvariant
)


class QuantumVisualizer:
    """Visualizador para experimentos de Surface Code."""
    
    # Cores padronizadas
    COLORS = {
        'error_x': '#FF4444',
        'error_z': '#4444FF',
        'error_y': '#FF8800',
        'syndrome_x': '#CC0000',
        'syndrome_z': '#0000CC',
        'correction': '#00CC00',
        'success': '#00AA00',
        'failure': '#CC0000',
        'neutral': '#888888'
    }
    
    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        self.figsize = figsize
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 10
    
    def plot_experiment(self, 
                       errors: np.ndarray,
                       syndrome: SyndromePattern,
                       correction: np.ndarray,
                       title: str = "Surface Code Experiment"):
        """
        Visualiza um experimento completo.
        
        Args:
            errors: Matriz de erros
            syndrome: Síndromes medidas
            correction: Correções aplicadas
            title: Título do plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Erros
        self._plot_pauli_matrix(axes[0], errors, "Erros Aplicados")
        
        # 2. Síndromes
        self._plot_syndromes(axes[1], errors.shape[0], syndrome)
        
        # 3. Correções
        self._plot_pauli_matrix(axes[2], correction, "Correções MWPM")
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _plot_pauli_matrix(self, ax, matrix: np.ndarray, title: str):
        """Plota matriz de operadores Pauli."""
        cmap = ListedColormap([
            'white',
            self.COLORS['error_x'],
            self.COLORS['error_z'],
            self.COLORS['error_y']
        ])
        
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=3)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Legenda
        counts = {
            'I': np.sum(matrix == 0),
            'X': np.sum(matrix == 1),
            'Z': np.sum(matrix == 2),
            'Y': np.sum(matrix == 3)
        }
        stats = f"X:{counts['X']} Z:{counts['Z']} Y:{counts['Y']}"
        ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                fontweight='bold', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_syndromes(self, ax, lattice_size: int, syndrome: SyndromePattern):
        """Plota síndromes."""
        syn_map = np.zeros((lattice_size, lattice_size))
        
        for pos in syndrome.x_syndromes:
            if pos[0] < lattice_size and pos[1] < lattice_size:
                syn_map[pos] = 1
        
        for pos in syndrome.z_syndromes:
            if pos[0] < lattice_size and pos[1] < lattice_size:
                syn_map[pos] = 2
        
        cmap = ListedColormap([
            'white',
            self.COLORS['syndrome_x'],
            self.COLORS['syndrome_z']
        ])
        
        ax.imshow(syn_map, cmap=cmap, vmin=0, vmax=2)
        ax.set_title("Síndromes Detectadas", fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        stats = f"X:{syndrome.x_count} Z:{syndrome.z_count}"
        ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                fontweight='bold', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def plot_threshold_analysis(self,
                               error_rates: np.ndarray,
                               success_rates: Dict[int, np.ndarray],
                               title: str = "Análise de Threshold"):
        """
        Plota análise de threshold para múltiplas distâncias.
        
        Args:
            error_rates: Array de taxas de erro testadas
            success_rates: Dict {distance: success_rate_array}
            title: Título do plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(success_rates)))
        
        for (distance, rates), color in zip(sorted(success_rates.items()), colors):
            ax.plot(error_rates, rates, 'o-', 
                   label=f'd={distance}', 
                   color=color,
                   linewidth=2,
                   markersize=6)
        
        # Linha de 50%
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50%')
        
        ax.set_xlabel('Taxa de Erro Físico', fontsize=12)
        ax.set_ylabel('Taxa de Sucesso', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.show()
    
    def plot_scaling_analysis(self, 
                             analyses: List[ScalingAnalysis],
                             title: str = "Análise de Scaling de Invariantes"):
        """
        Visualiza análise de scaling entre dimensões.
        
        Args:
            analyses: Lista de análises de scaling
            title: Título do plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Preservação por comparação
        ax1 = axes[0]
        labels = [f"d={a.source_distance}→{a.target_distance}" for a in analyses]
        ratios = [a.preservation_ratio for a in analyses]
        colors = [self.COLORS['success'] if r > 0.7 else self.COLORS['failure'] 
                 for r in ratios]
        
        bars = ax1.bar(labels, ratios, color=colors, alpha=0.8)
        ax1.axhline(y=0.7, color='orange', linestyle='--', label='Threshold 70%')
        ax1.set_ylabel('Taxa de Preservação')
        ax1.set_title('Preservação de Invariantes por Escala')
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        for bar, ratio in zip(bars, ratios):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{ratio:.1%}', ha='center', fontweight='bold')
        
        # 2. Detalhe por invariante
        ax2 = axes[1]
        if analyses and analyses[0].details:
            inv_names = list(analyses[0].details.keys())
            x = np.arange(len(inv_names))
            width = 0.8 / len(analyses)
            
            for i, analysis in enumerate(analyses):
                preserved = [1 if analysis.details.get(name, {}).get('preserved', False) else 0
                           for name in inv_names]
                ax2.bar(x + i * width, preserved, width, 
                       label=f"d={analysis.source_distance}→{analysis.target_distance}",
                       alpha=0.8)
            
            ax2.set_xticks(x + width * len(analyses) / 2)
            ax2.set_xticklabels(inv_names, rotation=45, ha='right')
            ax2.set_ylabel('Preservado (1) / Não (0)')
            ax2.set_title('Preservação por Invariante')
            ax2.legend()
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_invariant_distribution(self,
                                   invariants: Dict[str, List[float]],
                                   title: str = "Distribuição de Invariantes"):
        """
        Plota distribuição de valores de invariantes.
        
        Args:
            invariants: Dict {nome: [valores]}
            title: Título
        """
        n_invariants = len(invariants)
        if n_invariants == 0:
            print("Sem invariantes para plotar")
            return
        
        cols = min(3, n_invariants)
        rows = (n_invariants + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_invariants == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for ax, (name, values) in zip(axes, invariants.items()):
            ax.hist(values, bins=20, alpha=0.7, color=self.COLORS['neutral'])
            ax.axvline(np.mean(values), color='red', linestyle='--', 
                      label=f'μ={np.mean(values):.3f}')
            ax.set_title(name, fontweight='bold')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Frequência')
            ax.legend()
        
        # Esconder axes extras
        for ax in axes[n_invariants:]:
            ax.set_visible(False)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_multi_scale_comparison(self,
                                   results_by_distance: Dict[int, List[ExperimentResult]],
                                   invariant_name: str,
                                   title: str = None):
        """
        Compara um invariante específico entre múltiplas escalas.
        
        Args:
            results_by_distance: Resultados por distância
            invariant_name: Nome do invariante a comparar
            title: Título opcional
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        distances = sorted(results_by_distance.keys())
        positions = []
        data = []
        labels = []
        
        for d in distances:
            values = []
            for result in results_by_distance[d]:
                for inv in result.invariants:
                    if inv.name == invariant_name:
                        values.append(inv.value)
            
            if values:
                data.append(values)
                positions.append(d)
                labels.append(f'd={d}')
        
        if not data:
            print(f"Sem dados para invariante '{invariant_name}'")
            return
        
        # Box plot
        bp = ax.boxplot(data, positions=positions, widths=0.6)
        
        # Adicionar pontos individuais
        for i, (pos, vals) in enumerate(zip(positions, data)):
            jitter = np.random.normal(0, 0.1, len(vals))
            ax.scatter(pos + jitter, vals, alpha=0.4, s=20)
        
        # Adicionar linha conectando médias
        means = [np.mean(d) for d in data]
        ax.plot(positions, means, 'r-o', linewidth=2, markersize=8, label='Média')
        
        ax.set_xlabel('Distância do Código', fontsize=12)
        ax.set_ylabel(invariant_name, fontsize=12)
        ax.set_title(title or f'Comparação Multi-Escala: {invariant_name}',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
