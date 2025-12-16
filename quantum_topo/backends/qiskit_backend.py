"""
Backend Qiskit para Rotated Surface Code.
Gera circuitos quânticos a partir da definição geométrica.
"""
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import Dict, List, Tuple, Optional
import numpy as np

# Tentar importar AerSimulator, mas não falhar se não existir
try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False
    # Fallback para BasicProvider se disponível (Qiskit 1.0+)
    try:
        from qiskit.providers.basic_provider import BasicProvider
        BASIC_PROVIDER_AVAILABLE = True
    except ImportError:
        BASIC_PROVIDER_AVAILABLE = False

from ..core.rotated_surface import RotatedSurfaceCode

class RotatedQiskitBackend:
    """
    Converte RotatedSurfaceCode em QuantumCircuit.
    """
    
    def __init__(self, code: RotatedSurfaceCode):
        self.code = code
        self.d = code.d
        
        # Registradores
        self.qr_data = QuantumRegister(code.num_qubits, 'data')
        
        # Mapeamento ancillas -> bits
        self.ancilla_x_list = list(code.x_ancillas.keys())
        self.ancilla_z_list = list(code.z_ancillas.keys())
        
        self.num_x_ancillas = len(self.ancilla_x_list)
        self.num_z_ancillas = len(self.ancilla_z_list)
        
        # Ancillas físicas (1 qubit cada)
        self.qr_anc_x = QuantumRegister(self.num_x_ancillas, 'anc_x')
        self.qr_anc_z = QuantumRegister(self.num_z_ancillas, 'anc_z')
        
        # Bits clássicos serão criados dinamicamente em build_syndrome_extraction_circuit
        self.circuit = QuantumCircuit(self.qr_data, self.qr_anc_x, self.qr_anc_z)
        
        # Mapas de coordenadas para índice no registrador
        self.data_map = code.grid # coord -> int index
        self.x_anc_map = {coord: i for i, coord in enumerate(self.ancilla_x_list)}
        self.z_anc_map = {coord: i for i, coord in enumerate(self.ancilla_z_list)}
        
    def build_syndrome_extraction_circuit(self, rounds: int = 1):
        """
        Constrói o circuito de extração de síndrome para N rounds.
        Cria registradores clássicos: 'syn_x_r{i}' e 'syn_z_r{i}'.
        """
        qc = self.circuit
        
        # Armazenar histórico de registradores para acesso externo
        self.history_cr_x = []
        self.history_cr_z = []
        
        for r in range(rounds):
            # Criar registradores para este round
            cr_x = ClassicalRegister(self.num_x_ancillas, f'syn_x_r{r}')
            cr_z = ClassicalRegister(self.num_z_ancillas, f'syn_z_r{r}')
            qc.add_register(cr_x, cr_z)
            
            self.history_cr_x.append(cr_x)
            self.history_cr_z.append(cr_z)
            
            # 1. Preparação dos Ancillas
            # X-Ancillas (medem X): iniciam em |+> (H gate)
            qc.h(self.qr_anc_x)
            
            # 2. CNOTs (Entangle) - Z-order
            
            # --- X-Stabilizers (Medem XXXX) ---
            # H(anc) -> CNOT(anc, data) -> H(anc)
            for coord, neighbors in self.code.x_ancillas.items():
                anc_idx = self.x_anc_map[coord]
                anc_qubit = self.qr_anc_x[anc_idx]
                sorted_neighbors = sorted(neighbors, key=lambda idx: self.code.data_qubits[idx])
                for data_idx in sorted_neighbors:
                    qc.cx(anc_qubit, self.qr_data[data_idx])
                    
            # --- Z-Stabilizers (Medem ZZZZ) ---
            # CNOT(data, anc)
            for coord, neighbors in self.code.z_ancillas.items():
                anc_idx = self.z_anc_map[coord]
                anc_qubit = self.qr_anc_z[anc_idx]
                sorted_neighbors = sorted(neighbors, key=lambda idx: self.code.data_qubits[idx])
                for data_idx in sorted_neighbors:
                    qc.cx(self.qr_data[data_idx], anc_qubit)
            
            # 3. Medida e Reset
            # X-Ancillas: H gate antes da medida
            qc.h(self.qr_anc_x)
            
            qc.measure(self.qr_anc_x, cr_x)
            qc.measure(self.qr_anc_z, cr_z)
            
            # Reset ancillas para o próximo round
            if r < rounds - 1:
                qc.reset(self.qr_anc_x)
                qc.reset(self.qr_anc_z)

    def export_qasm(self, filename: str):
        """
        Exporta o circuito para arquivo OpenQASM (compatível com IBM Quantum Composer).
        """
        try:
            # Tentar usar qiskit.qasm2 (moderno)
            import qiskit.qasm2
            qiskit.qasm2.dump(self.circuit, filename)
            print(f"📁 Circuito exportado (QASM 2.0): {filename}")
        except (ImportError, AttributeError):
            # Fallback para método antigo
            try:
                qasm_str = self.circuit.qasm()
                with open(filename, 'w') as f:
                    f.write(qasm_str)
                print(f"📁 Circuito exportado (Legacy QASM): {filename}")
            except Exception as e:
                print(f"❌ Falha na exportação QASM: {e}")

    def get_circuit(self):
        return self.circuit

    def simulate(self, shots: int = 1000):
        """Simula o circuito."""
        if AER_AVAILABLE:
            backend = AerSimulator()
            print("🚀 Usando Qiskit Aer Simulator")
        elif BASIC_PROVIDER_AVAILABLE:
            # Fallback Qiskit 1.0+
            from qiskit.providers.basic_provider import BasicProvider
            backend = BasicProvider().get_backend('basic_simulator')
            print("⚠️ Usando BasicSimulator (lento, sem noise model)")
        else:
            print("❌ Nenhum simulador disponível (instale qiskit-aer)")
            return None
            
        # Transpilar (necessário para mapear corretamente)
        transpiled_qc = qiskit.transpile(self.circuit, backend)
        job = backend.run(transpiled_qc, shots=shots)
        return job.result()
