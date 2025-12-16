# Surface Code QEC no IBM Heron R2

Este projeto implementa e valida um código de correção de erro quântico (*Rotated Surface Code*) em hardware real da IBM (processador Heron R2).

## 🚀 Resultados

Conseguimos demonstrar a proteção de um qubit lógico com **91.80% de fidelidade** experimental.

| Métrica | Valor |
|---------|-------|
| Hardware | IBM Quantum (Heron R2) |
| Código | Surface Code d=3 |
| Qubits Físicos | 17 |
| Shots | 1024 |
| **Sucesso Lógico** | **91.80%** |

## 🛠️ Tecnologias

- **Python**: Linguagem principal.
- **Stim**: Simulação de alta performance para validação teórica.
- **Qiskit**: Integração com hardware IBM e exportação de circuitos (QASM).
- **PyMatching**: Decodificação de síndromes de erro.

## 📂 Estrutura do Projeto

- `lab.py`: CLI principal para gerenciar experimentos.
- `surface_d3.qasm`: Circuito quântico gerado.
- `decode_ibm_final.py`: Decodificador otimizado para o mapeamento de hardware.
- `fetch_results.py`: Script de integração com IBM Quantum Cloud.

## 👨‍💻 Sobre

Projeto desenvolvido durante estudos de Análise e Desenvolvimento de Sistemas, explorando a interseção entre Computação Quântica e Engenharia de Software assistida por IA.
