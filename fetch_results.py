
import json
from qiskit_ibm_runtime import QiskitRuntimeService

# Configuração fornecida pelo usuário
try:
    print("🔄 Conectando ao IBM Quantum service...")
    # Tenta usar conta salva ou a configuração específica
    # ATENÇÃO: Se não tiver conta salva, precisa do TOKEN.
    # Cole seu API TOKEN abaixo se der erro de "Unable to find account".
    MY_TOKEN = 'No .env lol xD' 
    
    if MY_TOKEN == 'COLE_SEU_IBM_TOKEN_AQUI':
        print("⚠️ Token nao definido no script. Tentando carregar do disco/env...")
        service = QiskitRuntimeService(
            channel='ibm_cloud',
            instance='No .env lol xD'
        )
    else:
        print(f"🔑 Usando Token fornecido (IBM Cloud Mode)...")
        service = QiskitRuntimeService(
            channel='ibm_cloud',
            instance='No .env lol xD',
            token=MY_TOKEN
        )
    
    JOB_ID = 'd50c91kgk3fc73avs1s0'
    print(f"📥 Baixando Job ID: {JOB_ID}")
    
    job = service.job(JOB_ID)
    result = job.result()
    
    # Debug: Imprimir tipo do resultado
    print(f"DEBUG: Tipo do resultado: {type(result)}")
    
    # Extraindo counts - PRIORIDADE: Sampler V2 com BitArray
    counts = None
    
    # MÉTODO 1: Sampler V2 (PrimitiveResult -> SamplerPubResult -> DataBin -> BitArray)
    try:
        # result[0] retorna SamplerPubResult
        pub_result = result[0]
        print(f"DEBUG: Tipo de result[0]: {type(pub_result)}")
        
        if hasattr(pub_result, 'data'):
            data_bin = pub_result.data
            print(f"DEBUG: Tipo de data: {type(data_bin)}")
            
            # Tenta acessar o BitArray (nome padrão é 'meas' ou 'c')
            bit_array = None
            if hasattr(data_bin, 'meas'):
                bit_array = data_bin.meas
                print(f"DEBUG: Encontrei data.meas: {type(bit_array)}")
            elif hasattr(data_bin, 'c'):
                bit_array = data_bin.c
                print(f"DEBUG: Encontrei data.c: {type(bit_array)}")
            else:
                # Tenta pegar o primeiro atributo que seja BitArray
                for attr in dir(data_bin):
                    if not attr.startswith('_'):
                        obj = getattr(data_bin, attr)
                        if hasattr(obj, 'get_counts'):
                            bit_array = obj
                            print(f"DEBUG: Encontrei BitArray em data.{attr}")
                            break
            
            if bit_array is not None and hasattr(bit_array, 'get_counts'):
                counts = bit_array.get_counts()
                print(f"✅ Extraído via BitArray.get_counts()")
    except Exception as e:
        print(f"⚠️ Método V2 falhou: {e}")
    
    # MÉTODO 2: Fallback para dict estruturado
    if counts is None and isinstance(result, dict):
        print(f"DEBUG: Chaves do dicionário: {list(result.keys())}")
        
        if 'counts' in result:
            counts = result['counts']
        elif 'results' in result and len(result['results']) > 0:
            res0 = result['results'][0]
            print(f"DEBUG: Tipo de results[0]: {type(res0)}")
            print(f"DEBUG: Chaves de results[0]: {list(res0.keys()) if isinstance(res0, dict) else 'N/A'}")
            
            if isinstance(res0, dict):
                if 'data' in res0:
                    data_obj = res0['data']
                    print(f"DEBUG: Tipo de data: {type(data_obj)}")
                    print(f"DEBUG: Chaves de data: {list(data_obj.keys()) if isinstance(data_obj, dict) else 'N/A'}")
                    
                    # Explorar cada chave de data
                    for key in data_obj.keys():
                        val = data_obj[key]
                        print(f"DEBUG: data['{key}']: tipo={type(val)}, preview={str(val)[:100]}")
                    
                    if 'counts' in data_obj:
                        counts = data_obj['counts']
                    elif 'meas' in data_obj:
                        meas_obj = data_obj['meas']
                        print(f"DEBUG: meas é tipo {type(meas_obj)}")
                        if isinstance(meas_obj, dict):
                            if 'samples' in meas_obj:
                                samples = meas_obj['samples']
                                print(f"DEBUG: meas['samples'] tem {len(samples)} elementos")
                                # Converter lista de samples hex para counts
                                counts = {}
                                for sample in samples:
                                    key = str(sample)
                                    counts[key] = counts.get(key, 0) + 1
                            elif 'counts' in meas_obj:
                                counts = meas_obj['counts']
                        elif hasattr(meas_obj, 'get_counts'):
                            counts = meas_obj.get_counts()
                    elif 'c' in data_obj:
                        c_val = data_obj['c']
                        if isinstance(c_val, dict):
                            counts = c_val
                        elif isinstance(c_val, list):
                            print(f"DEBUG: 'c' é lista com {len(c_val)} elementos")
                            counts = {}
                            for sample in c_val:
                                key = str(sample)
                                counts[key] = counts.get(key, 0) + 1
                    elif 'samples' in data_obj:
                        samples = data_obj['samples']
                        print(f"DEBUG: 'samples' encontrado com {len(samples)} elementos")
                        counts = {}
                        for sample in samples:
                            key = str(sample)
                            counts[key] = counts.get(key, 0) + 1
    
    # MÉTODO 3: get_counts genérico
    if counts is None and hasattr(result, 'get_counts'):
        counts = result.get_counts()
        print(f"✅ Extraído via result.get_counts()")
    
    if counts is None:
        # Dump completo para debug
        print("❌ NÃO CONSEGUI EXTRAIR COUNTS.")
        print(f"Atributos do result: {dir(result)}")
        try:
            print(f"result[0]: {result[0]}")
            print(f"Atributos de result[0]: {dir(result[0])}")
        except:
            pass
        raise ValueError(f"Não encontrei counts! Estrutura: {type(result)}")
        
    print(f"✅ Counts recuperados! Total de bitstrings únicas: {len(counts)}")
    
    output_file = "results.json"
    with open(output_file, "w") as f:
        json.dump(counts, f, indent=2)
        
    print(f"💾 Salvo em: {output_file}")
    print("\n🚀 PRÓXIMO PASSO:")
    print(f"python lab.py decode --distance 5 --data-file {output_file}")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\n❌ Erro fatal: {e}")
