"""
Benchmark Definitivo: Rotated Surface Code com Circuit-Level Noise via Stim.
Mede o threshold real do código sob ruído de porta (depolarizing).
"""
import stim
import pymatching
import numpy as np
import time
from quantum_topo.core.rotated_surface import RotatedSurfaceCode
from quantum_topo.backends.stim_backend import RotatedStimBackend

def run_stim_benchmark():
    print("="*70)
    print("⚡ STIM BENCHMARK: CIRCUIT-LEVEL NOISE")
    print("="*70)
    
    # Circuit Noise Threshold é tipicamente ~0.5% - 1%.
    # Vamos varrer uma faixa sensível.
    distances = [3, 5, 7] # d=7 pode ser pesado para laptop se samples alto, mas Stim voa.
    noise_rates = [0.001, 0.003, 0.005, 0.007, 0.01] # 0.1% a 1.0%
    samples = 5000 # Stim é muito rápido!
    rounds = None # Se None, usa d rounds.
    
    print(f"Config: d={distances}")
    print(f"Noise (p): {noise_rates}")
    print(f"Samples: {samples}")
    
    results = {}
    
    for p in noise_rates:
        print(f"\n📊 Noise p={p:.4f} (Circuit Delta Depolarizing)")
        print("-" * 50)
        
        for d in distances:
            t0 = time.time()
            num_rounds = d if rounds is None else rounds
            
            # 1. Gerar Circuito e DEM
            code = RotatedSurfaceCode(d)
            backend = RotatedStimBackend(code)
            backend.generate_circuit(rounds=num_rounds, noise=p)
            
            dem = backend.get_dem()
            
            # 2. Configurar Decoder (PyMatching automático via DEM!)
            # Isso é mágico: PyMatching infere o grafo de correção do modelo de erro do Stim.
            matcher = pymatching.Matching.from_detector_error_model(dem)
            
            # 3. Amostrar (Shots)
            # Retorna matriz booleana: [shots, num_detectors + num_observables]
            # Última coluna é o observável lógico real (ground truth se flipou ou não).
            sample_data = backend.sample_detector_syndrome(samples)
            
            # Separar síndromes (detectores) e observáveis reais
            num_detectors = dem.num_detectors
            syndromes = sample_data[:, :num_detectors]
            actual_observables = sample_data[:, num_detectors:]
            
            # 4. Decodificar
            # O decoder prevê qual observável lógico flipou baseado na síndrome.
            predicted_observables = matcher.decode_batch(syndromes)
            
            # 5. Verificar Erro Lógico
            # Erro se predição != realidade
            # (no contexto de QEC, "realidade" é o erro acumulado. Se decoder prevê igual, corrige).
            
            # Importante: decode_batch retorna array de uint8. Sample retorna bool.
            # Comparação:
            num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))
            
            success_rate = 1.0 - (num_errors / samples)
            
            dt = time.time() - t0
            results[(p, d)] = success_rate
            
            print(f"   d={d} | Success: {success_rate:.1%} | Time: {dt:.2f}s | Errors: {num_errors}")
            
    # Tabela Final
    print("\n" + "="*70)
    print("🏆 RESULTADO FINAL: CIRCUIT-LEVEL THRESHOLD")
    print("="*70)
    header = f"  p      |   " + "   |   ".join([f"d={d}" for d in distances]) + "   | Status"
    print(header)
    print("-" * len(header))
    
    for p in noise_rates:
        row_str = f" {p:<7.4f} |"
        rates = []
        for d in distances:
            r = results.get((p, d), 0)
            rates.append(r)
            row_str += f"  {r:.1%}   |"
            
        # Check scaling
        is_suppressed = True
        for i in range(len(rates)-1):
            if rates[i+1] < rates[i]: is_suppressed = False
            
        status = "✅ SUPRESSED" if is_suppressed else "❌ FAILED"
        if rates[0] < 0.6: status = "💀 NOISE"
        
        print(f"{row_str} {status}")
        
if __name__ == "__main__":
    run_stim_benchmark()
