import nltk
import glob
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.stem import RSLPStemmer
import math
import pandas as pd
import argparse
import sys
import seaborn as sns; sns.set_theme()
import json
import nltk

class Buscador:

  def __init__(self, methods):
    self.files_name = []
    self.methods_to_treatment = methods
    self.__matriz = {}
    self.__matriz_freq_doc = {}
    # steammer
    self.stemmer = RSLPStemmer()
    # Configurando stop words
    self.stopwords = set(nltk.corpus.stopwords.words('english'))
    self.someError = False
    self.loaded_from_disk = False
    self.my_file = pd.read_csv('/content/sample_data/base_books.csv', sep=",").to_numpy()

  def string_unica(self, document):
    if document is None: return ''
    text = ''
    for l in document:
      text += ' ' + str(l)
    return text

  ## Removendo duplicados
  def removendo_duplicados(self, seq, idfun=None): 
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

  def pre_processamento(self, tokens_to_proccess, methods = []):
    tokens = tokens_to_proccess
    for method in methods:
      # Mantain only alphanum 
      if method == 'alpha_num':
        tokens = [t for t in tokens if t.isalnum()]
      
      # All tokens to lower case
      if method == 'lower_case':
        tokens = [t.lower() for t in tokens]
      
      # removing the stop words
      if method == 'stop_words':
        tokens = [w for w in tokens if w not in self.stopwords]

      # byby words duplicated
      if method == 'remove_duplicated':
        tokens = self.removendo_duplicados(tokens)

      # steaming
      if method == 'steamming':
        tokens = [self.stemmer.stem(t) for t in tokens]

    return tokens

  def processamento_base(self):
    if self.loaded_from_disk:
      return
    matriz = {}
    freq_word_doc = []

    # Calcula a frequência dos termos nos documentos
    for idx, file in enumerate(self.my_file):
      # String única com todo o conteúdo
      text_document = self.string_unica(file)
      # Nome do documento
      name_file = file[2]
      self.files_name.append(name_file)
      # text_document += ' ' + name_file
      # usando sent_tokenize
      sentences_proccessed = sent_tokenize(text_document.replace('\n', ' '))
      # tokenization
      tokens = []
      for s in sentences_proccessed:
          words = word_tokenize(s)    
          tokens.extend(words)
      # normalizacao
      tokens_normalized = self.pre_processamento(tokens, ['alpha_num', 'lower_case', 'stop_words', 'remove_duplicated', 'steamming', 'remove_duplicated'])
      freq_word_doc.extend(tokens_normalized)
    freq_word_doc = FreqDist(freq_word_doc)
    self.__matriz_freq_doc = freq_word_doc
    # matriz de pesos tf-idf
    for idx, file in enumerate(self.my_file):
      text_document = self.string_unica(file)
      name_file = file[2]
      # text_document += ' ' + name_file
      # usando sent_tokenize
      sentences_proccessed = sent_tokenize(text_document.replace('\n', ' '))
      # tokenization
      tokens = []
      for s in sentences_proccessed:
          words = word_tokenize(s)    
          tokens.extend(words)
      # normalizacao
      tokens_normalized = self.pre_processamento(tokens, self.methods_to_treatment)
      # freq dos termos no documento
      freq_tokens = FreqDist(tokens_normalized)
      # qtd de documentos
      qtd_doc = len(self.my_file)
    
      # Acrescentar na matrix essas freq
      for token in freq_tokens:
        try:
          num = (qtd_doc - freq_word_doc[token] * 0.5) 
          idf =  math.log10( num / freq_word_doc[token] )
          if idf <= 0:
            idf = 0.01
        except:
          idf =  0.01

        try:
          tf_idf = { 'value': (1.0 + math.log10( freq_tokens[token] )) * idf, 'doc': name_file }
          matriz[token][idx] = tf_idf
        except KeyError:
          matriz[token] = [{'value': 0, 'doc': ''}] * qtd_doc
          tf_idf = { 'value': (1.0 + math.log10( freq_tokens[token] )) * idf, 'doc': name_file }
          matriz[token][idx] = tf_idf

    self.__matriz = matriz


  def busca_rankeada(self, search):
    search_tokens = search.split(' ')
    search_tokens = self.pre_processamento(search_tokens, ['alpha_num', 'lower_case', 'stop_words', 'steamming'])
    tfidf_search = self.calcular_idf_busca(
      self.pre_processamento(search.split(' '),
      ['alpha_num', 'lower_case', 'stop_words', 'steamming'],
    ))

    files_name = self.files_name
    dict_rank = {}
    common_tokens_mat = []
    # recupero da matriz os termos em comuns com a consulta
    for i in search_tokens:
      try:
        common_tokens_mat.append(self.__matriz[i])
      except:
        pass

    if len(common_tokens_mat) == 0:
      print('Nenhum resultado foi encontrado para a sua busca!\n')
      return

    for i_doc, file_name in enumerate(files_name):
      doc_vector = [doc[i_doc]['value'] for doc in common_tokens_mat ]
      doc_vector_normalized = self.__normalize(doc_vector)
      cos = 0
      for idx, i in enumerate(search_tokens):
        try:
          cos += doc_vector_normalized[idx] * tfidf_search[i]
        except:
          pass
      dict_rank[file_name] = cos

    list_rank = [(k, v) for k, v in dict_rank.items()] 
    results = sorted(list_rank, key = lambda i: i[1], reverse=True)

    print('Mostrando os 10 primeiros do ranking\n')
    #print(len(list_rank))
    #print(len(results))
    print(self.__matriz_freq_doc)
    print(results)
    for i in range(10): 
      print(results[i])

  def __normalize(self, vector):
    acc = 0
    for v in vector:
      acc += v ** 2
    norm = math.sqrt(acc)
    aux_vector = []
    for v in vector:
      if norm:
        aux_vector.append(v / norm)
      else:
        aux_vector.append(0)
    return aux_vector

  def calcular_idf_busca(self, search):
    freq_tokens = FreqDist(search)
    tfidf = {}
    qtd_doc = len(self.files_name)
    for token in freq_tokens:
      try:
        idf =  math.log10( qtd_doc / self.__matriz_freq_doc[token] )
        if idf <= 0:
          idf = 0.01
      except:
        idf =  0.01

      tfidf[token] = (1 + math.log10( freq_tokens[token] )) * idf
    normalized = self.__normalize([ v for k, v in tfidf.items() ])

    for idx, token in enumerate(freq_tokens):
      tfidf[token] = normalized[idx]

    return tfidf













try:
  print('## Baixando pacotes extras...')
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('rslp')
  print('## Pronto...\n')
except:
  print('## Erro ao baixar os pacotes!\n')
  exit(0)

print('## Aplicando configurações e instanciando ferramentas...')

# Tratamentos a serem aplicados no corpus
tratement = ['alpha_num', 'lower_case', 'stop_words', 'steamming']

print('## Instanciando a engine...')
buscador = Buscador(tratement)
print('## Iniciando processamento da base de dados. Isso pode demorar um pouco...')
buscador.processamento_base()
search = input('Busca: ')
while search != 'exit':
  buscador.busca_rankeada(search)
  search = input('Busca: ')