from collections.abc import Mapping
from os import write
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math

# Set page title and favicon
st.set_page_config(page_title="Analyse en Composantes Principales (ACP)", page_icon=":chart_with_upwards_trend:")

# Set page layout
header = st.container()
data_exploration = st.container()
data_preprocessing = st.container()
data_analysis = st.container()
results = st.container()
footer = st.container()

with header:
    st.title("Analyse en Composantes Principales (ACP)")
    st.markdown("---")

with data_exploration:
    st.header("L'application de l’analyse en composante principale")
    st.write("Avant de commencer l'analyse en composantes principales, examinons les données à analyser.")
    st.write("Voici notre données:")
    
    
    # Ask user to enter his data
    data_input = st.text_input("Enter the input data in the following format (example):", 
                           "'Q1': [8, 4, 6, 10, 8, 0],  'Q2': [1, 6, 8, 4, 2, 3], 'Q3': [0, 5, 7, 7, 5, 6]")

# Convert input data to a pandas dataFrame
    X = pd.DataFrame(eval(f"{{{data_input}}}"), index=["I1", "I2", "I3", "I4", "I5", "I6"])


    # afficher original data
    st.write(X)

with data_preprocessing:
    st.subheader("Le tableau centré")
    st.write("Avant de procéder à l'analyse en composantes principales, il est important de centrer les données.")
    st.write("Voici la matrice centrée correspondante:")
    
    # calculer centered data
    X_centre = X - np.mean(X, axis=0)

    # afficher centered data
    st.write(X_centre)

with data_analysis:
    st.subheader("La matrice variance covariance")
    st.write("Maintenant que les données ont été centrées, nous pouvons procéder à l'analyse en composantes principales.")
    st.write("Voici la matrice de variance-covariance:")
    
    # trouver la matrice variance-covariance 
    X1 = X - np.mean(X, axis=0)
    V = 1/X1.shape[0]*(X1.transpose()@X1)

    # afficher variance-covariance matrix
    st.write(V)
    st.subheader("Recherche des axes principaux 𝑼𝒌 de la matrice (VM)")
    st.write("Nous pouvons maintenant calculer les valeurs propres et vecteurs propres de cette matrice:")
    
    # trouver eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(V)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    # afficher eigenvalues and eigenvectors
    st.write("Valeurs Propres (triées par ordre décroissant) :")
    st.write(eigenvalues)
    st.write("Vecteurs Propres :")
    st.write(eigenvectors)

    st.write("Nous pouvons également calculer la qualité de représentation en fonction du nombre de composantes principales:")
    
    # calculer la qualité de représentation
    quality = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    # trouver nombre de composants 
    k = np.where(quality >= 0.8)[0][0] + 1

    st.subheader("Qualité de Représentation et Nombre de Composantes :")

    # afficher la qualité representation
    quality_df = pd.DataFrame({
    'Composante Principale': range(1, len(quality)+1),
    'Qualité de Représentation': quality
        })
    st.dataframe(quality_df.style.format({'Qualité de Représentation': '{:.2%}'}))

    st.write(f"Le nombre de Composantes Principales à conserver pour expliquer 80% de la variance est : {k}")
    # calculer les composantes principales
    U = eigenvectors[:, :k]
    C = X_centre @ U
    # afficher les composantes principales
    st.subheader("Composantes Principales :")
    st.write("Vecteurs Propres des Composantes Principales :")
    st.dataframe(pd.DataFrame(U, columns=[f'CP{i+1}' for i in range(k)]))
    st.write("Les données centrées projetées sur les Composantes Principales :")
    st.dataframe(pd.DataFrame(C))

    #Les contributions d'inerties

with results:
    st.header("Les contributions aux inerties")
    st.write("Nous pouvons maintenant calculer les contributions aux inerties de chaque individu pour chaque axe :")
    contributions = pd.DataFrame(columns=['Axe ' + str(i) for i in range(1, k+1)], index=X.index)

    for i in range(len(X)):
        for j in range(k):
            contributions.iloc[i,j] = (C.iloc[i,j]**2) / ((X_centre.iloc[i,0])**2+(X_centre.iloc[i,1])**2+(X_centre.iloc[i,2])**2)  
    #affichage des contributions                  

    st.write(contributions)

    #calculer des contributions dans les deux axes     
    #   
    contributionsGlobale = pd.DataFrame(columns=['ρ' + str(i) for i in range(1)], index=X.index)
    for i in range(len(X)):
        for j in range(1):
            contributionsGlobale.iloc[i,j] = contributions.iloc[i,0]+contributions.iloc[i,1]

    #Afficher des contributions dans les deux axes 

    st.write("Nous pouvons maintenant calculer la contribution de l’individu par rapport au nouvel espace  :")
    st.write(contributionsGlobale)
    #Les contributions relatives
    contributionsRelative = pd.DataFrame(columns=['Axe ' + str(i) for i in range(1, k+1)], index=X.index)
    for i in range(len(X)):
        for j in range(k):
            contributionsRelative.iloc[i,j] = (1/len(X)*(C.iloc[i,j]**2)) / eigenvalues[j]  
    #affichage    
    st.write("Maintenant on va calculer la Contribution relative de chaque individu  :")   
    st.write(contributionsRelative)
    
#Représenter graphiquement les individus    
import plotly.express as px
import plotly.graph_objs as go

# Button d'affichage
show_plot = st.button('Afficher le graphique')


if show_plot:
    fig = px.scatter(C, x=C.columns[0], y=C.columns[1], color=X.index)
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(title='Graphique des Composantes Principales',
                      xaxis_title='Première Composante Principale',
                      yaxis_title='Deuxième Composante Principale',
                      font=dict(family='Arial', size=14))

    
    fig.add_shape(type='line',
                  x0=C[C.columns[0]].min(), y0=0, x1=C[C.columns[0]].max(), y1=0,
                  line=dict(color='black', width=1))
    fig.add_shape(type='line',
                  x0=0, y0=C[C.columns[1]].min(), x1=0, y1=C[C.columns[1]].max(),
                  line=dict(color='black', width=1))

    # Button pour masquer l'affichage
    hide_plot = st.button('Masquer le graphique')

    if hide_plot:
        st.write('')
    else:
        st.plotly_chart(fig)
