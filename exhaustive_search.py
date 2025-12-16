#!/usr/bin/env python3
"""
🔬 BUSCA EXAUSTIVA DO Z LÓGICO
Testa TODAS as combinações de 3 bits para ver se alguma funciona.
"""
import json
import itertools

def hex_to_bin(hex_str, num_bits):
    if hex_str.startswith('0x'):
        hex_str = hex_str[2:]
    val = int(hex_str, 16)
    return bin(val)[2:].zfill(num_bits)

def exhaustive_search(data_file="results.json"):
    print("🔬 BUSCA EXAUSTIVA DO Z LÓGICO")
    print("=" * 60)
    
    # Carregar dados
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    num_bits = 33
    
    samples = []
    for hex_key, count in data.items():
        binary = hex_to_bin(hex_key, num_bits)
        for _ in range(count if isinstance(count, int) else 1):
            samples.append(binary)
    
    total = len(samples)
    print(f"   Total shots: {total}")
    print(f"   Bits por sample: {num_bits}")
    
    # Testar todas as combinações de 3 bits
    print("\n🔍 Testando todas combinações de 3 bits...")
    print("-" * 60)
    
    results = []
    
    # Data qubits estão nos últimos 9 bits (c[24] até c[32])
    # Mas vamos testar TODOS os bits
    
    for bits in itertools.combinations(range(num_bits), 3):
        p0, p1 = 0, 0
        for s in samples:
            z_bits = [s[-(b + 1)] for b in bits if b + 1 <= len(s)]
            parity = sum(1 for b in z_bits if b == '1') % 2
            if parity == 0:
                p0 += 1
            else:
                p1 += 1
        
        success = max(p0, p1) / total
        results.append((bits, success, p0, p1))
    
    # Ordenar por sucesso
    results.sort(key=lambda x: -x[1])
    
    print("   Top 20 combinações:")
    for i, (bits, success, p0, p1) in enumerate(results[:20], 1):
        print(f"   {i:2d}. c{list(bits)}: {100*success:.2f}% ({p0} vs {p1})")
    
    # 3 bits entre os últimos 9 (data qubits)
    print("\n🎯 Top 10 combinações apenas dos DATA QUBITS (c[24]-c[32]):")
    print("-" * 60)
    
    data_results = [r for r in results if all(24 <= b <= 32 for b in r[0])]
    for i, (bits, success, p0, p1) in enumerate(data_results[:10], 1):
        print(f"   {i:2d}. c{list(bits)}: {100*success:.2f}%")
    
    # Melhor resultado
    print("\n" + "=" * 60)
    print("🏆 MELHOR RESULTADO GERAL")
    print("=" * 60)
    best = results[0]
    print(f"   Bits: c{list(best[0])}")
    print(f"   Taxa de sucesso: {100*best[1]:.2f}%")
    print(f"   Paridade 0: {best[2]}, Paridade 1: {best[3]}")
    
    if best[1] > 0.6:
        print("\n✅ ENCONTRADO! Taxa > 60%")
    elif best[1] > 0.55:
        print("\n⚠️ SINAL DETECTADO! Taxa > 55%")
    else:
        print("\n❌ Nenhuma combinação funciona bem.")
        print("\n   Isso sugere que:")
        print("   1. O ruído do hardware é MUITO alto (>50% erro)")
        print("   2. Ou o circuito não foi executado corretamente")
        print("   3. Ou há problema na compilação do transpiler")

if __name__ == "__main__":
    exhaustive_search()
