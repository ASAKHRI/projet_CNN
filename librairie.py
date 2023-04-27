# Import des librairies.
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import inspect
import streamlit as st
import numpy as np



# ////////////////////////////////////////////////////////////// DATA SQL //////////////////////////////////////////////////////////////////////////// #

### Fonction permettent de créer une table.
### Les paramètres : On donne en paramètre le nom de la table que nous souhaitons créer
def create_table(table_name:str):
    engine=create_engine('mysql+pymysql://root:@localhost/titanic')
    inspector=inspect(engine)
    if not table_name in inspector.get_table_names():

        # Initialisation des colonnes.
        df = pd.DataFrame({'SEXE':[], 'AGE':[] , 'PCLASS':[], "predict":[]})

        # Typage des colonnes de la Table SQL.
        df['SEXE'   ]  = df['SEXE'   ].astype('str')
        df['AGE'    ]  = df['AGE'    ].astype('float64')
        df['PCLASS' ]  = df['PCLASS' ].astype('float64')
        df['predict']  = df['predict'].astype('str')

        # envoie du DataFrame sur SQL.
        df.to_sql(name=table_name, con=engine, if_exists='fail', index=False)
    print(f"Création de la table {table_name} avec succès.")

# //////////////////////////////////////////////////////// Traitement Formulaire ///////////////////////////////////////////////////////////////////// #

### Fonction permettent d'enregistrer le formulaire en BDD (la target et les features).
### Les paramètres : la connexion à la BDD, et une liste des features.
def save_formulaire(conn, features:list):
    cursor = conn.cursor()
    sql = "INSERT INTO predictions_titanic (SEXE, AGE, PCLASS, predict) VALUES (%s,%s,%s,%s)"
    cursor.execute(sql, features)
    conn.commit()
    conn.close()
