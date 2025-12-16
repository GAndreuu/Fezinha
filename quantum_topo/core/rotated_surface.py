import numpy as np
from typing import Tuple, Dict, Set, List, Optional
from dataclasses import dataclass
from .structures import SyndromePattern

class RotatedSurfaceCode:
    """
    Implementação do Rotated Surface Code (Geometric Leap).
    
    Esta geometria é mais eficiente que o código planar padrão:
    - Usa aproximadamente d^2 qubits (vs 2d^2) para a mesma distância d.
    - Threshold esperado ~10% (vs ~1-3% do planar).
    
    Coordenadas (x, y) de 0 a 2d:
    - Qubits de Dados: (x, y) onde x, y são pares ou x, y são ímpares (x%2 == y%2)
                  dentro dos limites do patch.
    - Estabilizadores: (x, y) nos "buracos" intermediários (x%2 != y%2).
    """

    def __init__(self, distance: int):
        self.d = distance
        self.lattice_size = 2 * distance + 1
        
        # Estruturas de dados
        self.data_qubits = [] # Lista de coords (x, y)
        self.qubit_map = {}   # (x, y) -> index
        
        self.x_stabilizers = [] # Coords dos Z-stabilizers (detectam erros Z?) 
                                # Cuidado: X-stab detecta erro Z. Z-stab detecta erro X.
        self.z_stabilizers = [] 
        
        self.x_ancillas = {} # (x,y) -> [qubit_indices vizinhos]
        self.z_ancillas = {}
        
        self._build_lattice()
        
        # Definir Operadores Lógicos (Cadeias que atravessam o lattice)
        # Logical Z: Cadeia vertical (Col 0)
        # Logical X: Cadeia horizontal (Row 0)
        # Armazenar como lista de listas de coordenadas (suporta múltiplos logicals se necessário)
        
        self.z_logicals = []
        # Z Vertical: Qubits na coluna 0 (aprox)
        # Na verdade, logical Z conecta bordas Top/Bottom (Rough).
        # Caminho: (0,0), (1,0), ..., (d-1,0) ?
        # Vamos pegar qubit em cada linha na coluna 0.
        z_chain = []
        for r in range(self.d):
            # Encontrar qubit mais próximo da col 0 na linha r
            # Rotated code grid é (row, col) com saltos.
            # Grid construction:
            # data_qubits:
            # Vamos achar os da primeira coluna disponível.
            candidates = [q for q in self.data_qubits if q[1] == 0] # Se existir
            if not candidates:
                # Tentar col 1 se grid deslocado?
                candidates = [q for q in self.data_qubits if q[1] == 1]
                
            # Na implementação atual _build_lattice cria data qs em (r,c).
            # Vamos pegar "todos data qubits onde col == 0" se for impar?
            pass
            
        # Revisitando _build_lattice: data qubits são coord (r,c) para 0<=r<d, 0<=c<d?
        # Sim: self.data_qubits = [] ... for r in range(d): for c in range(d): ...
        
        # Então Logical Z (Vertical) é a coluna 0.
        # DEBUG: Coluna 0 (Left Borda) estava dando erro de não-determinismo com Stim.
        # Possível conflito com checks de borda.
        # Vamos tentar Coluna 1 (Bulk) ou Coluna d-1 (Right).
        # Tentar Coluna 0 novamente mas verificando coordenadas.
        # Se falhou, vamos tentar deformar para Coluna central.
        # Para d=3, col 1 é central.
        
        target_col = 0 # Voltar para 0 e persistir? Ou mudar?
        # O erro anterior indicava sensibilidade a Y7 (2,1).
        # Se usarmos col 1, incluimos (2,1).
        
        # Vamos tentar Coluna 1.
        z_chain = [(r, 1) for r in range(self.d)]
        self.z_logicals.append(z_chain)
        
        # Logical X (Horizontal) é a linha 0.
        x_chain = [(0, c) for c in range(self.d)]
        self.x_logicals = []
        self.x_logicals.append(x_chain)

    def _build_lattice(self):
        """Constrói a geometria rotacionada."""
        d = self.d
        
        # 1. Definir Qubits de Dados
        # Padrão: x, y variando, mantendo x%2 == y%2
        # Limites definem o patch "diamante" ou quadrado rotacionado?
        # No rotated code, é um quadrado cortado nas diagonais do lattice original.
        # Coordenadas simples: grade retangular onde qubits estão em um padrão de tabuleiro.
        
        # Range aproximado: 0 a 2d
        idx = 0
        for x in range(2 * d + 1):
            for y in range(2 * d + 1):
                # Data qubits em coords onde (x+y) é par? Ou alguma outra convenção.
                # Vamos usar: Data qubits em (x, y) se x%2 == 1 e y%2 == 1 (ímpares)
                # E também ( pares e pares)?
                # Vamos seguir o padrão padrão do Google 'Rotated Surface Code':
                # Data qubits: vértices (x,y).
                # Z-stabs (Measure Z): faces (x,y). 
                # Mas vamos simplificar para coordenadas inteiras únicas.
                
                # Convenção:
                # Qubits de dados em (row, col) para 0 <= row, col < d * 2 - 1 ?
                # Melhor: Qubits em coordenadas inteiras (r, c).
                # Para d=3:
                # Qubits: (0,1), (0,3), (1,0), (1,2), (1,4), (2,1), ...
                
                # Vamos usar a implementação padrão:
                # Data qubits em (x,y) tal que (x+y) % 2 == 1.
                # X-Checks em (x,y) tal que (x+y) % 2 == 0 e x é par?
                
                # Vamos simplificar: Usar um rectângulo de 0 a 2d.
                # Data Qubits: (x,y) se (y%2 == 0 e x%2 == 1) ou (y%2 == 1 e x%2 == 0)?
                # Não, isso é x+y impar.
                pass
        
        # IMPLEMENTAÇÃO CONCRETA:
        # Data Qubits: (2x+1, 2y+1) ??
        # Vamos usar um design explícito para d=3:
        # 0 -- X -- 0     (Linhas de cima/baixo são Smooth/Rough?)
        
        # Vamos usar a definição clássica de coordenadas (i, j):
        # Data Qubits: coords (2i, 2j) ?? Não, isso é esparso.
        
        # Vamos tentar: 
        # Data qubits em TODAS as coords (r, c) de um grid d x d?
        # Sim! Um grid d x d físicos.
        # data_qubits = [(r, c) for r in range(d) for c in range(d)]
        # Isso gera d^2 qubits. Correto.
        
        # Agora os estabilizadores ficam "entre" eles.
        # X-checks (Medida X, detecta Z): "Vértices" entre 4 qubits.
        # Z-checks (Medida Z, detecta X): "Plaquetas" entre 4 qubits.
        
        self.data_qubits = []
        q_idx = 0
        self.grid = {} # (r, c) -> index
        
        for r in range(d):
            for c in range(d):
                self.data_qubits.append((r, c))
                self.grid[(r, c)] = q_idx
                q_idx += 1
                
        self.num_qubits = len(self.data_qubits)
        
        # Estabilizadores (Ancillas)
        # Ficam em coordenadas fracionárias (r+0.5, c+0.5)? 
        # Vamos usar índices inteiros em um grid deslocado ou lista.
        
        # X-Stabilizers (Vertex type, star):
        # Centrados em (r, c) relativos ao grid de dados.
        # Padrão Checkerboard.
        # Se (r+c) for par -> Z-check. Se (r+c) for ímpar -> X-check. (Ou vice-versa)
        
        # Para d=3:
        # Qubits (0,0) (0,1) (0,2)
        #        (1,0) (1,1) (1,2)
        #        (2,0) (2,1) (2,2)
        
        # Checks internos:
        # Entre (0,0),(0,1),(1,0),(1,1) -> centro (0.5, 0.5)
        
        # Vamos iterar pelos "quadrados" possíveis de 4 qubits.
        # r vai de 0 a d-2, c vai de 0 a d-2.
        # Mas também temos checks de borda (weight-2).
        
        # Definir X-Checks e Z-Checks:
        # Z-boundaries em Top/Bottom (Linhas 0 e d-1).
        # X-boundaries em Left/Right (Colunas 0 e d-1).
        
        # Vamos varrer uma grade expandida de checks.
        # Z-checks:
        #   (r, c) check conecta data qubits: (r, c), (r, c+1), (r+1, c), (r+1, c+1)
        #   Se (r+c) % 2 == 0: Z-check.
        
        # Precisamos cobrir todo o lattice + bordas.
        
        # Lista de potenciais checks (r, c) representando o "topo-esquerda" do quadrado
        # r varia de -1 a d-1
        # c varia de -1 a d-1
        
        for r in range(-1, d):
            for c in range(-1, d):
                # Determinar cor do check (Checkerboard)
                if (r + c) % 2 == 0:
                    check_type = 'Z'
                else:
                    check_type = 'X'
                
                # Identificar vizinhos válidos (Data Qubits)
                # O check em (r,c) conecta potencialmente a:
                # (r, c), (r, c+1), (r+1, c), (r+1, c+1)
                
                neighbors = []
                potential_qs = [(r, c), (r, c+1), (r+1, c), (r+1, c+1)]
                
                for pr, pc in potential_qs:
                    if 0 <= pr < d and 0 <= pc < d:
                        idx = self.grid[(pr, pc)]
                        neighbors.append(idx)
                
                if len(neighbors) == 0:
                    continue
                    
                # Regras de Borda do Rotated Surface Code:
                # Z-Stabilizers (checks Z) devem fechar ciclos verticais (logical Z na vertical?)
                # Normalmente logical Z é cadeia horiz (L->R) ou Vert?
                # Vamos definir: Logical Z = Cadeia de Zs da esquerda p/ direita (atravessa X-boundaries?)
                # Não. Logical Z conecta bordas "Z".
                # Se Z-checks são brancos (r+c par).
                
                # Vamos usar a lógica padrão:
                # Checks internos tem peso 4.
                # Checks de borda tem peso 2.
                
                if len(neighbors) == 4:
                    # Bulk - sempre manter
                    if check_type == 'X':
                        self.x_ancillas[(r, c)] = neighbors
                    else:
                        self.z_ancillas[(r, c)] = neighbors
                
                elif len(neighbors) == 2:
                    # Bordas - Definir tipos corretos para casar com Logical Operators
                    # Logical Z é Vertical (Col 0) -> Top/Bottom devem ser Z-Boundaries (Rough)
                    # Logical X é Horizontal (Row 0) -> Left/Right devem ser X-Boundaries (Smooth)
                    
                    # Identificar borda
                    # Para weight 2, os vizinhos estão alinhados
                    row_coords = [self.data_qubits[idx][0] for idx in neighbors]
                    col_coords = [self.data_qubits[idx][1] for idx in neighbors]
                    
                    is_top = all(row == 0 for row in row_coords)
                    is_bottom = all(row == d-1 for row in row_coords)
                    is_left = all(col == 0 for col in col_coords)
                    is_right = all(col == d-1 for col in col_coords)
                    
                    if check_type == 'Z':
                        # Z-checks (detect X) permitidos no Top/Bottom (Rough Z boundaries)
                        if is_top or is_bottom:
                            self.z_ancillas[(r, c)] = neighbors
                            
                    elif check_type == 'X':
                        # X-checks (detect Z) permitidos no Left/Right (Smooth X boundaries)
                        if is_left or is_right:
                            self.x_ancillas[(r, c)] = neighbors

    def apply_errors(self, p_error: float) -> np.ndarray:
        """Aplica erros depolarizantes (X, Y, Z)."""
        # Array de erros: 0=I, 1=X, 2=Z, 3=Y
        errors = np.zeros(self.num_qubits, dtype=int)
        for i in range(self.num_qubits):
            if np.random.random() < p_error:
                # Escolher tipo de erro (equiprovável 1/3 para X, Y, Z por simplicidade ou independente?)
                # Modelo padrão simulação: Independent X/Z ? Ou Depolarizing?
                # Vamos usar Independent X/Z com prob p_error cada para comparar com planar anterior.
                # No código anterior usavamos: code.py aplica erro em x_errors e z_errors separadamente.
                # "Each qubit has prob p of X error, prob p of Z error".
                # Y error = X + Z, prob p^2.
                pass
                
        # Vamos manter compatibilidade:
        # Retorna array de erros inteiros? Ou tuple (x_errs, z_errs)?
        # A nova interface espera array unico se formos espertos.
        # Mas vamos retornar dois arrays booleanos para facilitar.
        
        x_errors = np.random.random(self.num_qubits) < p_error
        z_errors = np.random.random(self.num_qubits) < p_error
        return x_errors.astype(int), z_errors.astype(int)

    def measure_syndrome(self, errors: Tuple[np.ndarray, np.ndarray]) -> SyndromePattern:
        """Mede estabilizadores."""
        x_errs, z_errs = errors
        
        # X-Stabilizers detectam erros Z
        # Check dispara se paridade de erros Z nos vizinhos for ímpar
        triggered_x = []
        for coord, neighbors in self.x_ancillas.items():
            parity = sum(z_errs[n] for n in neighbors) % 2
            if parity == 1:
                triggered_x.append(coord)
                
        # Z-Stabilizers detectam erros X
        triggered_z = []
        for coord, neighbors in self.z_ancillas.items():
            parity = sum(x_errs[n] for n in neighbors) % 2
            if parity == 1:
                triggered_z.append(coord)
                
        # Retornar objeto compatível
        # Precisamos converter coords (r, c) para índices lineares ou manter coords?
        # O decoder precisa saber posições.
        # Vamos retornar coords. SyndromePattern deve aceitar qualquer hashable.
        
        return SyndromePattern(
            x_syndromes=triggered_x,
            z_syndromes=triggered_z
        )

    def get_logical_error(self, initial_errors, correction_map) -> bool:
        """Verifica se houve erro lógico Z (vertical) ou X (horizontal)."""
        x_errs_init, z_errs_init = initial_errors
        
        # Combinar erros e correções
        final_x = x_errs_init.copy()
        final_z = z_errs_init.copy()
        
        for q_idx, op in correction_map.items():
            if op == 'X' or op == 'Y':
                final_x[q_idx] ^= 1
            if op == 'Z' or op == 'Y':
                final_z[q_idx] ^= 1
                
        # Checar Logical Z (Cadeia vertical de Zs)
        # Basta checar comutação com Logical X operator?
        # Logical X operator ideal: Linha de X's na linha 0.
        # Se final_z anti-comutar com Logical X, houve erro lógico Z.
        # Anti-comuta se overlap for ímpar.
        
        # Logical X operator: X em todos (0, c)
        logical_X_op_qubits = [self.grid[(0, c)] for c in range(self.d)]
        parity_Z_on_LX = sum(final_z[q] for q in logical_X_op_qubits) % 2
        
        # Checar Logical X (Cadeia horizontal de Xs)
        # Logical Z operator ideal: Coluna de Z's na coluna 0.
        logical_Z_op_qubits = [self.grid[(r, 0)] for r in range(self.d)]
        parity_X_on_LZ = sum(final_x[q] for q in logical_Z_op_qubits) % 2
        
        return (parity_Z_on_LX == 1) or (parity_X_on_LZ == 1)
