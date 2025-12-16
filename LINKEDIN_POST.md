# 🚀 LinkedIn Post (English Version)

From Zero to Quantum Error Correction in 10 Hours (No Sleep ☕)

Never let anyone tell you it's "too early" to learn cutting-edge technologies.

I'm a 2nd-semester Systems Development student, and during a 10-hour hyperfocus session (deep into the night), I managed to run a quantum error correction algorithm on a real IBM quantum computer (the brand new Heron R2 ibm_fez). With the help of AI for coding and A LOT of debugging, we achieved 91.80% logical fidelity!

The barrier to entry is falling. The future is now. ⚛️💻

But it wasn't magic. It was a fight. 🧠🔥

For those who think quantum computing is just running a script:

1.  The Hardware is Brutal (but fascinating): The ibm_fez processor is an engineering masterpiece, but the daily calibration showed the reality of quantum noise. Qubit 72 had a 35.8% readout error (basically unusable) and the average CNOT gate error was ~1%.
2.  The Failure of d=5: I tried running a more complex code (49 qubits). Result? 50% error. Pure randomness. The ~500 necessary gates accumulated more noise than the system could handle.
3.  The Redemption of d=3: I reduced the complexity to 17 qubits. BINGO! The code worked, protecting the information with 91.8% success.

Fun fact: Initially, I thought the 10 free monthly minutes on IBM Cloud would be useless. To my surprise, this complex experiment consumed only 4 seconds of QPU time! You can play around (and discover) a lot. LOL! ⚡

My creativity ran out along with my sleep, but the satisfaction of seeing that "91.8% Success" in the terminal was worth every minute.

The raw data (JSON) and code are on my GitHub (link in comments) for anyone who wants to try decoding it too.

---
Technical Note / Disclaimer:
It is worth noting that we have not yet reached the "Break-even point" (where the logical qubit is better than the isolated physical one at ~94.3%). HOWEVER, without the correction code, the success probability of a 17-qubit system with this noise would be only ~37%.

The algorithm raised this fidelity from 37% to 91.8%, proving that the error detection and correction logic is indeed working and recovering information from the noisy hardware. 📈 There is still plenty of room to improve these 91.80%, even maintaining d=3.
*(signed: secret llm 🤖)*

Is this good for a first interaction with quantum hardware? Are these real results or just a hallucination by me+LLM?! 💀

#Tech #Quantum #Learning #Coding #Qiskit #IBMQuantum #Estudos #ADS
