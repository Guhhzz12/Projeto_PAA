# Funções tf_idf_similarity.py

```py
def calcular_tf(termo, doc):
    tf = 0
    for t in doc:
        if t == termo:
            tf += 1    
    return tf    
```
- **Descrição:** Calcula a frequência de aparições de um termo em um documento.
- **Variáveis:** `L` = quantidade de termos no documento.
- **Análise:** Contém um loop que itera `L` vezes. Cada operação no loop (comparação, incremento) é O(1). Portanto, a complexidade é **O(L)**.

```py
def calcular_idf(termo, corpus):
    idf = 0
    for doc in corpus:
        if termo in doc:
            idf += 1
    
    n = len(corpus)
    return math.log2(n/(idf + 1))
```
- **Descrição:** Calcula a Frequência Inversa de Documentos (IDF) de um termo no corpus.
- **Variáveis:** `N` = número de documentos no corpus, `L_avg` = comprimento médio dos documentos no corpus.
- **Análise:** Contém um loop que itera `N` vezes. Dentro do loop, a operação termo in doc em uma lista pode levar, no pior caso, O(L_avg) tempo. As demais operações são O(1). Portanto a complexidade é **O(N*L_avg)**
```py
def construir_vocabulario(corpus):
    vocabulario = set()
    for documento in corpus:
        for termo in documento:
            vocabulario.add(termo)
    return sorted(list(vocabulario)) 
```
- **Descrição:** Dado um conjunto de documentos, itera sobre cada elemento e adiciona os seus termos ao conjunto de vocabulário, sem repetições.
- **Variáveis:**  `N` = número de documentos, `L_avg` = comprimento médio dos documentos, `V` = número total de termos únicos no vocabulário.
- **Análise:** Os dois loops aninhados iteram aproximadamente `NL_avg` vezes no total. A operação vocabulario.add(termo) para um set é O(1). A conversão para list é O(V). A operação sorted() para ordenar a lista é O(VlogV). A operação dominante é a iteração sobre todos os termos para adicioná-los ao set e, em seguida, a ordenação. Poratanto a complexidade é **O(NL_avg+VlogV)**

```py
def vetorizar_tf_idf(doc, vocabulario, idf_scores):
    vetor = []
    for termo in vocabulario:
        tf = calcular_tf(termo, doc)
        idf = idf_scores.get(termo, 0) 
        vetor.append(tf * idf)
    return vetor
```
- **Descrição:** Função que cria um vetor de pesos de cada termo do vocabulário aplicado no documento atual
- **Variáveis:** `V` = número de termos no vocabulario, `L_doc` = comprimento do documento que está sendo vetorizado.
- **Análise:** Contém um loop que itera `V` vezes. Dentro do loop, a chamada a calcular_tf(termo, doc) tem complexidade O(L_doc). As demais operações são O(1). Portanto complexidade **O(VL_doc).**

```py
def magnitude_euclidiana(vetor):
    aux = 0
    for v in vetor:
        aux += (v*v)
    return math.sqrt(aux)
```
- **Descrição:** Função que calcula a magnitude euclidiana de um vetor
- **Variáveis:** `D` = quantidade de elementos do vetor
- **Análise:** Contém um loop que itera `D` vezes. Cada operação no loop é O(1), portanto a complexidade é **O(D)**

```py
def calcular_similaridade_cosseno(vetor_a, vetor_b):
    produto_escalar = sum(a * b for a, b in zip(vetor_a, vetor_b))
    magnitude_a = magnitude_euclidiana(vetor_a)
    magnitude_b = magnitude_euclidiana(vetor_b)
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return produto_escalar / (magnitude_a * magnitude_b)
```
- **Descrição:** Função que calcula a similaridade entre dois vetores.
- **Variáveis:** `D` = tamanho do vetor_a e vetor_b
- **Análise:** O cálculo do produto_escalar com zip e sum itera `D` vezes, resultando em O(D). Realiza duas operações de magnitude_euclidiana(), que possui O(D), portanto a complexidade é **O(D)**

```py
class TFIDFVectorizer:
    def __init__(self):
        self.vocabulario = []
        self.idf_scores = {}

    def fit(self, corpus):
        self.vocabulario = construir_vocabulario(corpus)
        for termo in self.vocabulario:
            self.idf_scores[termo] = calcular_idf(termo, corpus)
        
    def transform(self, doc):
        return vetorizar_tf_idf(doc, self.vocabulario, self.idf_scores)
```
- **Descrição:** Classe que encapsula o processo de vetorização TF-IDF, treinando o modelo (calculando vocabulário e IDFs) e transformando documentos em vetores.
- **Variáveis:** `N` = número de documentos no corpus, `L_avg` = comprimento médio dos documentos, `V` = número de termos únicos no vocabulário, `L_doc` = comprimento do documento a ser transformado.
- **Análise:** 
    - fit(corpus): 
        - Chamada a construir_vocabulario(corpus): **O(NL_avg+VlogV).**
        - Loop sobre self.vocabulario V vezes. Dentro do loop, chamada a calcular_idf(termo, corpus) que é O(NL_avg). Portanto, a complexidade deste loop é **O(VNL_avg).** 
        - Portanto a complexidade de fit é: O(NL_avg+VlogV+VNL_avg), que simplifica para **O(VNL_avg)**
    - transform(doc):
        - Chamada a vetorizar_tf_idf() que é O(VL_doc).
        - Portanto a complexidade é **O(VL_doc)**.