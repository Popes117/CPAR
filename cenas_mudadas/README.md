# CPAR
Repositório de Trabalhos Práticos da UC de Computação Paralela

_Notas Pro TP2_

Critic, atomic e assim vão introduzir alguma sequencialização
O tempo de execução vai ser igual aquele da thread que demorou mais 
Fazer a paralelização iterativa, fazendo, a cada tentativa, a verificação e justificação da escalabilidade


Para testar correr versão sequencial runseq:
```
make runseq
```


Para testar correr versão paralela runpar:
```
make runpar -> para 2 threads (default)

make THREADS=x runpar -> outro nº x de threads qualquer
```