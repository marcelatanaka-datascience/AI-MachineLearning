import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer ## word vectorizer
from sklearn.feature_extraction.text import CountVectorizer ## word vectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2


# Connect to Database
def connect_db(user, password):
    conn = psycopg2.connect(
        dbname = 'dummy',
        user = user,
        host = 'dummy',
        password = password,
        port = 'dummy',
        keepalives_idle=600
    )
    conn.autocommit = True
    cur = conn.cursor()
    return conn, cur

db_con, db_cur = connect_db(user='dummy', password='dummy') 

# You can use an csv file to build your recommendation
dp_product = pd.read_csv('filename.csv')

# OR you can query your dataframe directly from your database
query = ''' 
SELECT
FROM
'''

db_product = pd.read_sql(query, rs_con)


# Clean your data! (here is the place where you can and should make your data the cleaner as you can)
db_product[db_product.isna().any(axis=1)] 
db_product.dropna() 


# Here is where the magic begins 
# Recommendation based only on one variable. The variable you chose is the one your algorithm is going to compare it's results

tfidf = TfidfVectorizer() # Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(db_product['VARIABLE_TO_PREDICT'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) # Compute the cosine similarity matrix

indices = pd.Series(db_product.index, index=db_product['VARIABLE_TO_GET_RECOMMENDATION']).drop_duplicates() #Construct a reverse map of indices and the variable you want to predict
indices[:10]

# Function that takes in VARIABLE TO GET RECOMMENDATION as input and outputs most similar PRODUCTS
def get_recommendations(product, cosine_sim=cosine_sim):  
    
    idx = indices[product] # Get the product index of the wine that matches the variable to get recommendation  
    sim_scores = list(enumerate(cosine_sim[idx])) # Get the pairwise similarity scores of all products with that product
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # Sort the products based on the similarity scores
    sim_scores = sim_scores[1:11] # Get the scores of the 10 most similar products    
    product_indices = [i[0] for i in sim_scores] # Get the product indices    
    return db_product['VARIABLE_TO_GET_RECOMMENDATION'].iloc[product_indices] # Return the top 10 most similar products
  
  # ET VOILÀ
  get_recommendations('variable to get recommendation')

