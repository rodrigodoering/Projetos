# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 23:05:38 2019

@author: rodri
"""
# Importa as bibliotecas
import pandas as pd
import nltk
from funcoes_suporte import limpar
from Database import SQLServer

# conecta com o SQL Server 
database = SQLServer(dsn='DSN_TCC')
database.connect()
database.set_database('DB_LIVE')
print('Base de dados %s acessada' % database.current_database)

# Extrai os dados
dataset = database.select(table='tb_dataset_final')

'''
Nesse setor são importados os dados externos de termos que serão considerados ruído durante a etapa de limpeza do texto
Depois é contruída a função que aplicará a normalização e limpeza do texto

'''
# Importa o txt contendo termos específicos relacionados a assinaturas de email entre outros
with open('ruído específico.txt', 'r+', encoding='utf-8') as file:
    ruido_especifico = file.readline()
    ruido_especifico = ruido_especifico.split(',')

# Carrega o dataset contendo todos os nomes existentes no brasil bem como suas frequências, nomes são considerados ruído também
df_nomes = pd.read_csv('nomes.csv')
# Filtramos apenas por nomes frequêntes, caso contrário a etapa de limpeza terá uma validação demasiada grande para por linha
nomes = df_nomes[df_nomes['frequency_total'] >= 1000]['first_name'].values.tolist()
nomes = [nome.lower() for nome in nomes]

# Termos em portugues considerados stopwords do pacote NLTK
stopwords = nltk.corpus.stopwords.words("portuguese")

# Condensa todos os tipos de ruído em uma única lista que será passada na função limpar()
ruido = nomes + stopwords + ruido_especifico

# Separa texto e rótulo em arrays separados
texto = dataset["Conteudo"].values.tolist()
labels = dataset["Email_Persona"].values.tolist()

# Inician pre-processamento do texto
texto_processado = [limpar(sentenca, ruido, stemmer=True) for sentenca in texto]

# Condensa os termos processados e armazena em um novo dataframe
texto_table = list(map(lambda array_termos: ",".join(x for x in array_termos), texto_processado))
df = pd.DataFrame(data={"texto":texto_table, "class":labels})

# Apaga o conteúdo atual da tabela TB_texto_processado
database.query('truncate table TB_texto_processado', commit=True)

# Insere o texto tratado na tabela TB_texto_processado no SQL Server
database.insert(df, 'TB_texto_processado')




