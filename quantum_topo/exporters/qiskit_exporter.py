
"""
Exportador de Circuitos Stim para Qiskit (OpenQASM 3.0).
Permite rodar os códigos de superfície validados no hardware da IBM.
"""
import stim
from typing import Dict, List, Tuple
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit import Qubit, Clbit
except ImportError:
    QuantumCircuit = None

class QiskitExporter:
    def __init__(self, code, circuit: stim.Circuit):
        """
        Args:
            code: Objeto RotatedSurfaceCode (contém geometria/mapa).
            circuit: Circuito Stim gerado.
        """
        if QuantumCircuit is None:
            raise ImportError("Qiskit não está instalado. Instale com 'pip install qiskit'.")
            
        self.code = code
        self.stim_circuit = circuit
        
        # Mapeamento do Stim (Inteiro) -> Qiskit (Qubit Object)
        self.stim_to_qiskit = {}
        
        # Estrutura do Circuito Qiskit
        self.q_reg = None
        self.c_reg = None
        self.qc = None
        
    def build_circuit(self) -> "QuantumCircuit":
        """Converte o circuito Stim para Qiskit QuantumCircuit."""
        
        # 1. Contar Qubits necessários
        # O backend Stim usa índices lineares esparsos baseados em coordenadas.
        # Precisamos descobrir o índice máximo usado ou mapear para um registro denso.
        # Vamos usar um registro denso para os qubits físicos.
        
        num_qubits = self.code.num_qubits + len(self.code.x_ancillas) + len(self.code.z_ancillas)
        
        # Vamos criar um mapa reverso de todos os qubits no código
        self.q_reg = QuantumRegister(num_qubits, 'q')
        # Precisamos de bits clássicos para as medidas
        # O Stim não declara bits explicitamente, mas M gera records.
        # Vamos contar quantas medidas existem no circuito.
        num_measurements = self.stim_circuit.num_measurements
        self.c_reg = ClassicalRegister(num_measurements, 'meas')
        
        self.qc = QuantumCircuit(self.q_reg, self.c_reg)
        
        # Mapeamento: Coordinate -> Index no QReg
        # Vamos reconstruir a ordem usada no backend
        # Data Qubits primeiro
        current_idx = 0
        
        # Precisamos recriar o mapa exato que o StimBackend usou?
        # Sim, o Stim Circuit usa indices inteiros. Precisamos saber q qual qubit físico
        # o indice "10" se refere.
        # O StimBackend usa:
        # data_start = 0
        # x_anc_start = len(data)
        # z_anc_start = ...
        
        # Vamos assumir que os indices no stim_circuit JÁ SÃO os indices 0..N-1 
        # (se o backend foi construído linearmente).
        # Verificando stim_backend.py:
        # self.data_start = 0
        # self.x_anc_start = 2 * self.d * self.d
        # AVISO: O Backend usa coordenadas "virtuais" convertidas em indices 
        # linearizados com gaps ou compactos?
        # Ele usa: self.coord_to_idx[coord]
        # data_qubits são range(0, N_data)
        # É compacto! Então q[i] no Stim == q[i] no Qiskit.
        
        meas_index = 0
        
        for instruction in self.stim_circuit:
            if instruction.name == "QUBIT_COORDS":
                continue
            
            elif instruction.name == "R": # Reset
                for t in instruction.targets_copy():
                    self.qc.reset(self.q_reg[t.value])
                    
            elif instruction.name == "H":
                for t in instruction.targets_copy():
                    self.qc.h(self.q_reg[t.value])
                    
            elif instruction.name in ["CNOT", "CX"]:
                targets = instruction.targets_copy()
                # CNOT em pares [ctrl, target, ctrl, target...]
                for i in range(0, len(targets), 2):
                    c = targets[i].value
                    t = targets[i+1].value
                    self.qc.cx(self.q_reg[c], self.q_reg[t])
            
            elif instruction.name == "M": # Measure
                for t in instruction.targets_copy():
                    self.qc.measure(self.q_reg[t.value], self.c_reg[meas_index])
                    meas_index += 1
            
            elif instruction.name == "TICK":
                self.qc.barrier()
            
            elif instruction.name == "DETECTOR":
                # Detectores são metadados de software, não instruções de hardware.
                # Não geram QASM instructions executáveis.
                pass
            
            elif instruction.name == "OBSERVABLE_INCLUDE":
                pass
            
            elif instruction.name == "SHIFT_COORDS":
                pass
            
            elif instruction.name in ["DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "CORRELATED_ERROR"]:
                # Ignorar ruído na exportação (o hardware tem seu próprio ruído natural!)
                pass
                
            else:
                print(f"⚠️ Instrução ignorada/desconhecida: {instruction.name}")

        return self.qc

    def to_qasm(self, version: int = 3) -> str:
        if self.qc is None:
            self.build_circuit()
            
        if version == 2:
            # OpenQASM 2.0 (Legacy)
            try:
                from qiskit import qasm2
                return qasm2.dumps(self.qc)
            except ImportError:
                 # Fallback for older Qiskit versions
                return self.qc.qasm()
            
        # OpenQASM 3.0 (Modern)
        try:
            from qiskit import qasm3
            return qasm3.dumps(self.qc)
        except ImportError:
            # Fallback to OQ2 if OQ3 module missing (older qiskit)
            return self.qc.qasm()
