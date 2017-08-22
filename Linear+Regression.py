
# coding: utf-8

# 
# Benvenuti ragazzi a AiforDummies, io sono Robu e vi accompagnerò in questa nuova avventura.
# 
# Immagino che, aprendo questo video, molti di voi abbiano già in menta cosa siano le reti neurali artificiali. Ma se ancora non avete di idea di cosa siano e quali siano le loro potenzialità, mi occuperò io di iniziarvi. 
# Per seguire questo corso non è necessario né essere dei matematici, né essere degli ingegneri informatici. L’unica cosa che vi chiedo è un’approfondita conoscenza della meccanica quantistica. Scherzo. Basta avere un minimo di dimestichezza con la sintassi base di Python.
# 
# Partiamo da una piccola introduzione storica:
# La nascita effettiva dell’Intelligenza Artificiale come disciplina a tutti gli effetti, la possiamo inquadrare a fine anni 50. Ma molte delle sue fondamenta le possiamo rintracciare nel ventennio precedente grazie ad un articolo di Turing che pone le basi per concetti come calcolabilità, computabilità che stanno alla base ancora oggi dei nostri calcolatori.  Senza poi dimenticarci dei matematici McCulloch&Pitts che crearono il primo modello di neurone artificiale. 
# I neural networks infatti, sono una replica in digitale, mettiamola così, del funzionamento delle nostre reti neurali biologiche. 
# Il soma sarà la nostra unità fondamentale di computazione, le linee che partono dagli input saranno i nostri dendriti, mentre il nostro assone sarà rappresentato dal canale output. 
# 
# Ma prima di addentrarci all’interno della costruzione di un neural network mi piacerebbe parlare un attimo del contesto più generale, ovvero quello del Machine Learning. 
# Il Machine Learning si trova agli antipodi della tradizionale programmazione, in cui ogni step ha bisogno di essere definito per avere l’outcome desiderato. Quando parliamo di Machine Learning, stiamo parlando dell’opposto. Parliamo appunto di definire un outcome e poi il programma da sé imparerà gli step necessari per raggiungerlo. 
# Il tutto può essere davvero utile per esempio nel creare un bot capace di riconoscere targhe, se questo è uno shiba o un marshmellow. O un altro esempio pratico potrebbe essere applicato a Super Mario, vai verso il traguardo senza lasciarci le penne. O addirittura creare un algoritmo che mi permetta di montare questa video. 
# Tutto questo rende il Machine Learning il metodo perfetto da utilizzare in situazioni in cui non abbiamo bene idea di cosa dobbiamo cercare come ad esempio nell’ identificare l’attività sospetta di qualche users.
# Il Machine Learning, è un vasto campo che investe direttamente le nostre vite: dagli algoritmi di YouTube con cui viene rimpolpata la categoria “consigliati” a Siri della Apple. 
# 
# Ora, ci ritroviamo con un dataset di misurazioni nel mondo animale in cui abbiamo il peso del cervello e il peso del corpo per ogni specie selezionata. In questo modo, vorremmo riuscire a predire, dato il peso del cervello, il peso dell’animale.
# L’approccio che andremo ad utilizzare sarà il supervised learning, in cui per ogni dato che forniamo abbiamo già un’etichetta che lo identifica. La task che invece andremo a performare sarà quella della Regressione lineare. 
# 
# 20 linee di Python che vi spiegherò durante la loro compilazione.   
# 
# 
# Iniziamo importando le nostre tre dependencies:
# Importiamo in primis Pandas che ci permetterà di leggere i nostri datu
# La seconda è Scikit learn, libreria per la nostra parte di Machine Learning. L'ultima è Matplotlib che ci permetterà di visualizzare il nostro modello e i nostri dati.
# 
# Come vedete io sto utilizzando Jupyter attraverso Anaconda. Lo trovo molto pratico e vi consiglierei di installarlo perchè rende davvero l'esperienza su windows meno traumatica. 
# 
# 
# 

# In[14]:

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


# Ora che abbiamo importato tutto quello che ci serve chiediamo a Pandas di leggere i nostri dati attraverso la read fwf function.

# In[2]:

dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]


# Quello che abbiamo fatto è trasformare il nostro file txt in struttura 2d di colonne e righe. Quello che contengono sono i valore medi di un certo numero di specie animali riguardo peso del cervello e peso del corpo.  
# In questo modo possiamo sistemare i pesi del cervello nel nostro asse X e i pesi del corpo su Y. 
# Ecco a voi il grafico. 

# In[3]:

get_ipython().magic(u'matplotlib notebook')

plt.xlabel("brain")
plt.ylabel("body")
plt.scatter(x_values, y_values)
plt.show()


# Abbiamo semplicemente chiesto a matplotlib di creare un grafico in cui viene mostrato il nostro dataset. 
# Già ad occhio possiamo riscontrare una discreta correlazione. 
# Il nostro obiettivo quindi è che dato un nuovo peso saremo capaci di predire quale sia la taglia del suo cervello. 
# In che modo? Passiamo un attimo alle cose formali.
# Abbiamo una variabile indipendete che è X e una variabile dipendente che è Y. Quello che dobbiamo fare è trovare la relazione che intercorre fra di loro appunto utilizzando la Linear Regression. 
# Letteralmente dobbiamo trovare la linea che maggiormente si adatta ai nostri dati. Di cosa abbiamo bisogno? Di una semplice equazione.
# Y = mx + b. 
# In cui b è l'intercetta ed mx la sua inclinazione. 
# Non ci resta che applicarla al nostro grafico. 
# 

# In[9]:

body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)
body_reg_pred = body_reg.predict(x_values)


# Siamo usciti abbastanza indenni dalla parte matematica. Vediamo ora, chiamando il grafico, la nostra linea che abbiamo generato poco fa. 

# In[13]:

get_ipython().magic(u'matplotlib notebook')

plt.xlabel("brain")
plt.ylabel("body")
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg_pred, color = 'green', linewidth = 3)
plt.show()


# Yass, che ve ne pare? Come potete vedere la linea si adatta davvero molto bene ai nostri dati e scorrendo al di sopra di essa possiamo predire per ogni peso del cervello, il peso del corpo associato. La correlazione è davero molto forte.  
# 
# Quindi buttiamo giù un paio di fatti fondamentali di cui abbiamo parlato oggi:
# 
# Il machine learning o apprendimento automatico ci permette una volta definito un outcome di imparare step by step come raggiungerlo.
# 
# L'apprendimento supervisionato è quello che abbiamo utilizzato oggi. Ma altri metodi che vedremo magari più avanti sono l'unsupervised e l'apprendimento per rinforzo. 
# 
# La regressione lineare è quello che ci permette di scovare le relazioni tra variabili dipendenti e indipendenti. Molto utile nel caso vogliate cimentarvi in studi utilizzando metodi correlazionali. 
# 
# 
# Siamo praticamente giunti alla fine di questo tutorial. Tutto il materiale lo troverete sul link github che metterò in descrizione. L'ideale sarebbe ovviamente provare a replicarlo. 
# Nella prossima lezione invece proveremo ad integrare a quello fatto oggi una rete neurale. 
# Se questo video ti è stato davvero di aiuto clicca pure mi piace ed iscriviti per rimanere sempre aggiornato sui prossimi video.
# 
# Alla prossima!
