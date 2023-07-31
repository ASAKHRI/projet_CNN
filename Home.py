# Import des utilitaires.
import streamlit as st
from librairie import create_table, save_formulaire
from modeling import prediction_survie
import pymysql




def main():

    # Connexion à la BDD.
    conn=pymysql.connect(host='localhost',port=int(3306),user='root', passwd='', db='cnn')

    # Accueil (titre & header)
    st.header("Projet CNN")
    st.title("Accueil")

    # On créer la table de prédiction s'il n'existe pas.
    create_table(table_name="prediction_chiffre")

    # Initialisation du formulaire
    formulaire={
        "GENDER" : st.selectbox("Veuillez saisir votre sexe", ["Homme", "Femme"]),
        "AGE"    : st.number_input("Veuillez saisir votre age", min_value=1, max_value=95),
        "PCLASS" : st.selectbox("Veuillez saisir le numéro de la classe", [1, 2, 3]),
    }

    # Si l'utilisateur clique sur envoyer :
    if st.button("Submit"):

        # Encodage de la feature gender.
        if formulaire["GENDER"] == "Femme":
            formulaire["GENDER"] = 0
        else:
            formulaire["GENDER"] = 1

        # Utilisation du modèle de prédiction.
        # On appelle la fonction prediction_survie() et on lui passe les réponses de l'utilisateur.
        pred = prediction_survie(features=[i for i in formulaire.values()])

        # On inscrit la prédiction dans le dictionnaire.
        if pred:
            formulaire["predict"] = "Vivant"
        else:
            formulaire["predict"] = "Mort"

        # Enregistrement du formulaire en BDD.
        # On appelle la fonction save_formulaire popur enregistrer les réponses de l'utilisateur en BDD
        save_formulaire(conn=conn, features=[i for i in formulaire.values()])

        # Affichage de la prédictions.
        if pred:
            st.header("Vous etes vivant !")
        else:
            st.header("Vous etes mort...")
main()
