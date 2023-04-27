# Import des utilitaires.
import streamlit as st
import pandas as pd 
import seaborn as sns 
import numpy as np

# =============================================== DATA CLEANING =============================================== #

# Importer les données
data = sns.load_dataset("titanic")

# Nettoyage
titanic = data[["survived", "pclass", "sex", "age"]]
titanic.dropna(axis=0, inplace=True)
titanic["sex"].replace(["male", "female"], [0, 1], inplace=True)


