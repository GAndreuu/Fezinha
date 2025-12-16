#!/usr/bin/env python3
"""
🔬 ANÁLISE DE QUALIDADE DE HARDWARE IBM
Identifica qubits problemáticos e estima impacto.
"""

# Dados de calibração extraídos do IBM
# Formato: qubit_id: (T1_us, T2_us, readout_error, CZ_error_avg)
calibration = {
    0: (59.44, 45.43, 0.01294, 0.00642),
    1: (221.87, 244.66, 0.02368, 0.00192),
    2: (290.1, 197.31, 0.00769, 0.00173),
    3: (224.26, 232.98, 0.02405, 0.00172),
    4: (215.31, 80.53, 0.00439, 0.00194),
    5: (139.04, 130.99, 0.00842, 0.00210),
    7: (105.8, 38.65, 0.02612, 0.00295),
    11: (127.16, 80.24, 0.04578, 0.01154),
    12: (76.43, 73.44, 0.02502, 0.00485),
    27: (76.97, 84.82, 0.00830, 0.02642),
    41: (176.04, 61.31, 0.06030, 0.00282),
    43: (158.45, 91.07, 0.09644, 0.01348),
    45: (119.44, 114.15, 0.05090, 0.00260),
    51: (55.89, 48.2, 0.04846, 0.00246),
    53: (140.08, 15.1, 0.04871, 0.00334),
    55: (144.07, 146.93, 0.05969, 0.00245),
    63: (133.44, 17.31, 0.03467, 0.02271),
    65: (59.5, 44.87, 0.09131, 0.00392),
    67: (91.79, 59.92, 0.05078, 0.00619),
    72: (12.14, 99.16, 0.35830, 0.1589),  # MUITO RUIM!
    83: (147.34, 166.45, 0.08362, 0.00357),
    87: (170.81, 291.44, 0.04749, 0.00260),
    101: (164.36, 109.43, 0.06116, 0.00279),
    113: (107.74, 28.27, 0.07458, 0.00293),
    131: (164.73, 225.83, 0.08130, 0.00201),
    135: (217.63, 197.8, 0.05688, 0.00401),
    149: (47.71, 5.63, 0.04504, 0.01025),  # T2 muito baixo!
}

def analyze_hardware():
    print("🔬 ANÁLISE DE QUALIDADE DO HARDWARE IBM")
    print("=" * 60)
    
    # 1. Identificar qubits críticos
    print("\n🚨 QUBITS CRÍTICOS (erro de leitura > 5%):")
    print("-" * 60)
    critical = [(q, d) for q, d in calibration.items() if d[2] > 0.05]
    critical.sort(key=lambda x: -x[1][2])
    
    for q, (t1, t2, re, cz) in critical:
        print(f"   Qubit {q:3d}: Erro leitura={100*re:5.1f}%, T1={t1:6.1f}us, T2={t2:6.1f}us")
    
    # 2. Qubits com T2 muito baixo
    print("\n⏱️ QUBITS COM T2 BAIXO (<50us):")
    print("-" * 60)
    low_t2 = [(q, d) for q, d in calibration.items() if d[1] < 50]
    low_t2.sort(key=lambda x: x[1][1])
    
    for q, (t1, t2, re, cz) in low_t2:
        print(f"   Qubit {q:3d}: T2={t2:5.1f}us, T1={t1:6.1f}us")
    
    # 3. Estatísticas gerais
    print("\n📊 ESTATÍSTICAS GERAIS:")
    print("-" * 60)
    
    all_re = [d[2] for d in calibration.values()]
    all_t2 = [d[1] for d in calibration.values()]
    all_cz = [d[3] for d in calibration.values()]
    
    print(f"   Erro de leitura médio: {100*sum(all_re)/len(all_re):.2f}%")
    print(f"   Erro de leitura máximo: {100*max(all_re):.2f}%")
    print(f"   T2 médio: {sum(all_t2)/len(all_t2):.1f}us")
    print(f"   T2 mínimo: {min(all_t2):.1f}us")
    print(f"   Erro CZ médio: {100*sum(all_cz)/len(all_cz):.3f}%")
    
    # 4. Impacto no Surface Code
    print("\n🎯 IMPACTO NO SURFACE CODE d=5:")
    print("-" * 60)
    
    # Surface code d=5 precisa de 49 qubits (25 data + 24 ancilla)
    # Com 5 rounds de síndrome, temos ~5*24 = 120 medições de síndrome
    # Cada round tem ~100 CNOTs
    
    n_qubits = 49
    n_rounds = 5
    n_cnots_per_round = 100
    total_cnots = n_rounds * n_cnots_per_round
    
    avg_cz_error = sum(all_cz)/len(all_cz)
    avg_readout_error = sum(all_re)/len(all_re)
    
    # Probabilidade de pelo menos um erro
    p_no_gate_error = (1 - avg_cz_error) ** total_cnots
    p_no_readout_error = (1 - avg_readout_error) ** n_qubits
    
    p_success = p_no_gate_error * p_no_readout_error
    
    print(f"   Total de CNOTs: {total_cnots}")
    print(f"   Prob. de ZERO erros de gate: {100*p_no_gate_error:.2f}%")
    print(f"   Prob. de ZERO erros de leitura: {100*p_no_readout_error:.2f}%")
    print(f"   Prob. de circuito perfeito: {100*p_success:.4f}%")
    
    # 5. O problema do qubit 72
    print("\n⚠️ ALERTA CRÍTICO:")
    print("-" * 60)
    print(f"   Qubit 72 tem ERRO DE LEITURA DE 35.8%!")
    print(f"   Se este qubit foi usado, os resultados serão ALEATÓRIOS.")
    print(f"   Recomendação: Evitar qubits 72, 65, 43, 55, 83, 131")
    
    print("\n" + "=" * 60)
    print("💡 CONCLUSÃO")
    print("=" * 60)
    print("""
   O hardware atual NÃO É ADEQUADO para Surface Code d=5 porque:
   
   1. Vários qubits têm erro de leitura > 5% (acima do threshold)
   2. Qubit 72 está praticamente quebrado (35.8% de erro)
   3. Alguns qubits têm T2 < 20us (decoerência muito rápida)
   4. Com ~500 CNOTs, erros acumulam exponencialmente
   
   Soluções possíveis:
   - Usar d=3 (menos qubits, circuito mais curto)
   - Escolher manualmente qubits bons para o mapeamento
   - Esperar calibração melhor do backend
   - Usar backend com melhor qualidade
    """)

if __name__ == "__main__":
    analyze_hardware()
