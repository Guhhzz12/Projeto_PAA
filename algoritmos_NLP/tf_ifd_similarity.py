import math

def calcular_tf(termo, doc):
    tf = 0
    for t in doc:
        if t == termo:
            tf += 1    
    return tf


def calcular_idf(termo, corpus):
    idf = 0
    for doc in corpus:
        if termo in doc:
            idf += 1
    
    n = len(corpus)
    return math.log2(n/(idf + 1)) # Adiciona 1 para evitar divisão por zero


def construir_vocabulario(corpus):
    vocabulario = set()
    for documento in corpus:
        for termo in documento:
            vocabulario.add(termo)
    return sorted(list(vocabulario)) 

def vetorizar_tf_idf(doc, vocabulario, idf_scores):
    vetor = []
    for termo in vocabulario:
        tf = calcular_tf(termo, doc)
        
        # Garante que o termo existe no idf_scores 
        # Se for um termo de consulta que não está no corpus, seu IDF seria 0 
        idf = idf_scores.get(termo, 0) 
        vetor.append(tf * idf)
    return vetor
        

def magnitude_euclidiana(vetor):
    aux = 0
    for v in vetor:
        aux += (v*v)
    return math.sqrt(aux)
    

def calcular_similaridade_cosseno(vetor_a, vetor_b):
    produto_escalar = sum(a * b for a, b in zip(vetor_a, vetor_b))
    magnitude_a = magnitude_euclidiana(vetor_a)
    magnitude_b = magnitude_euclidiana(vetor_b)
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return produto_escalar / (magnitude_a * magnitude_b)  


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
