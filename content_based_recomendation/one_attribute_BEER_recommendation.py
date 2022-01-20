# THIS IS AN PRACTICAL EXAMPLE OF ONE ATTRIBUTE RECOMMENDATION CODE


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer ##Vetorizador de palavras
from sklearn.feature_extraction.text import CountVectorizer ## Vetorizador de palavras
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

#Import csv as pandas dataframe
db_beer = pd.read_csv('beers.csv')

#Cleaning data!
db_beer.drop(['ibu'], axis=1,inplace=True)
db_beer.dropna(inplace=True) 
db_beer['style'] = db_beer['style'].str.replace(' ','') 

#Creating vectors
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(db_beer['style'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(db_beer.index, index=db_beer['name']).drop_duplicates()


#Find first 10 beer names for test purposes - This is OPTIONAL!
indices[:10]

#Find all beer names for test puposes - This is OPTIONAL!
for names in db_beer['name']:
    print(names)


#Create function to get beer recommendation based on BEER STYLE!
def get_recommendations(beer, cosine_sim=cosine_sim):  
    
    idx = indices[beer] 
    sim_scores = list(enumerate(cosine_sim[idx])) 
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) 
    sim_scores = sim_scores[1:11]     
    beer_indices = [i[0] for i in sim_scores]   
    return db_beer['name'].iloc[beer_indices]
  
  
  # For each beer name you insert inside the function you will receive a list of the 10 most similiar beers based on beer style! - check README.md for examples
  get_recommendations('Gwar Beer')
