from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from mistralai import Mistral
from dotenv import load_dotenv
from pydantic import BaseModel
import json
from create_db import save_to_chroma
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
load_dotenv()

api_key = os.environ['MISTRAL_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplace "*" par ["http://localhost", "https://ton-domaine.com"] si besoin
    allow_credentials=True,
    allow_methods=["POST"],  
    allow_headers=["*"],  
)


client = Mistral(api_key=api_key)

class QueryRequest(BaseModel):
    query_text: str


embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def run_mistral(user_message, model="mistral-large-latest"):
    messages = [
        {
            "role": "user", "content": user_message
        }
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        temperature=0.5
    )
    
    return (chat_response.choices[0].message.content)


@app.post("/query")
def get_response(request: QueryRequest):
    
    query_text = request.query_text

    # results = db.similarity_search_with_relevance_scores(query_text, k=3)
    # print(query_text)
    # print(results)

    # if not results:
    #     raise HTTPException(status_code=404, detail="Aucun résultat pertinent trouvé.")

    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    context_text = """
Charles Doux 24 ans, ingénieur en intelligence artificielle et cybersécurité à ESIEE Paris, diplomé de 10 septembre 2024 à la suite d'études passionnante.

Charles a effectué une classe préparatoire à esiee paris de 2 ans. Il est ensuite partit en république tchèque, à Brno. Pour effectuer un semestre à l'étranger dans l'ingénieurie electrique et informatique Il est ensuite revenu à ESIEE Paris pour se spécialisé.


En rentrant de ce semestre d'étude il se passionne beaucoup plus sérieusement pour l'informatique et la programmation. Il décide de rejoindre la spécialisation intelligence artificielle et cybersécurité.

Dans cette filière on y parle uniquement en Anglais et les cours sont évalué uniquement sur des projets très intéréssants.

Voici la liste des projets :
    En intelligence Artificielle : Un projet de reinforcement learning dans un jeu pacman.,Un projet de data science ou la finalité etait de pouvoir prédire si les étudiants allaient réussir ou non leur semestre.
,Analyse & recherche sémantique en NLP. Un projet de deep learning en utilisant des models convolutifs pour reconnaitre les elements sur les images.Un projet de prédiction de torsion du beton armé sur des ponts sur plusieurs années.Projet de clusterisation sur les personalités en frances.Plusieurs projets de classification ou regression avec des données Kaggle. Une application web pour les étudiants étrangersUn gestionnaire de prestation pour les intervenants

En cybersécurité : 
    - Projet d'analyse de risques. Projet de sécurisation de systèmes informatiquesProjet de Pentest sur des sites web. Projet de Pentest et sécurisation de réseaux informatiques

De plus, nous avions accès à des électives pour approfondire nos connaissances en gestion de projet ou management.

Mes electives etaient : 
    - Marketing opérationnel
    - Gestion de projet
    - Management


Durant le fin de période d'étude, Charles à pu faire le tour du monde, en voyageant dans plus 10 pays (Turquie, Qatar, Chine, Japon, Corée, Nouvelle zelande, chili, perou, colombie, brésil, argentine, bolivie).

Durant ce tour du monde, c'etait un stage de 3 mois à Tokyo et un semestre d'étude au Chili ou il a appris, le management operationnel, la vente, la negociation, la communications avec les paeties prenantes, le droit des affaires, la finance et l'economie.

Pour ce qui est de l'expérience : 
Les stages : 
    1- Premier stage s'effectue à Tokyo chez Kogakuin dans le domaine médical en datascience d'une durée de 3 mois. 
    L'équipe de chercheurs Japonais de Kogakuin avait pour but de classifier l'hémiplégie du genou chez les enfants.
    Pour ce, ils ont décidé d'utiliser des capteurs IMU placés à différents endroits du corps pour analyser ces mouvement savec une extrème précision.

    - Missions : 
        - Récupérer les données avec les capteurs sur site et les extraires en labo.
        - Netooyer les données en utilisant des filtres Kalman.
        - Analyser ces données à la main.
        - Utiliser le dataset existant pour créer des modèles de classification pour une reconnaissance de la maladie ou non en fonction du meme mouvement.

Charles a ensuite eu envie d'en apprendre plus sur le processing de données et le monde du big data. Il s'est donc décidé de faire un stage en tant que data engineer chez Calibsun à Sophia antipolis. 

    2- Deuxième stage : S'effecue à Sophia Antipolis chez calibsun dans le domaine des énergies renouvelable et plus précisément le solaire en data engineering d'une durée de 6 mois.

    Contexte : Calibsun est une société qui a construit un produit nomée NEXT, capable de prédire le rendement des centrales photovoltaïques sur plusieurs, minutes, jours, semaines. 

    Mission : 
        - Améliorer le pipeline de récupération des données satellite 
        - Optimiser le calcul et stockage de ces données
        - Récupérer 2 ans d'historiques

    Deroulement :

    L'enjeux etait ce récupérer 40GB de donénes toutes les 10 minutesnà faible cout. Le pipeline actuelle récupérait uniquement les données satellitaire europpéene et etait long et non optimisé pour l'utilisation en data science.

    J'ai crée un nouveau pipeline ETL en extractant les données de plusieurs satellite sur plusieurs filtres.
    Ici le calcul parallélisé etait opbligatoire, j'ai donc utiliser spark pour permettre à l'ordinateur de paralléliser les calculs et d'aggréger les données dans le format souhaité.

    J'ai pu utiliser un orchestrateur, dagster, pour orchestrer ce process de récupération sur des instances AWS.

    J'ai aussi utiliser cet orchestrateur pour paralléliser la récupération des 2 ans d'historique sur tous les satllites.

    Sur la période de 6 mois, j'ai pu efectuer ces missions principales en 4.

    J'ai donc eu du temps pour me former sur le YTest driven developpement, les pipelines CI/CD, l'infrastructure as a code avec AWS. Améliorer les pipelines CI/CD pour nos developpeurs.
    
    Et surtout ce stage m'a appris à communiquer en équipe avec les datascientist et manager, cartographier mon travail pour respecter les délais du projet et vulgariser jour par jour ma progression. 


    Je suis donc diplomé à la fin de ce stage en aout 2024

    Charles est moins bon dans la conception de software ou web. POur ce qui est data, il est encore junior dans le choix des architectures optimales meme s'il en apprends de jour en jour, il n'a pas l'habitude de travailler avec des logiciels de data analyst type powerBi.

    Pour la suite:
    3- Expérience freelance n1 : Un projet en deeplearning et LLM pour la reconnaissance de lieux cités dans les videos tiktok et instagram. 
    De Septembre 2024 à Decembre 2024
        - Scraping de publications Instagram et TikTok.
        - Extraction de texte à partir de l'audio, de la vidéo et des descriptions avec Deep Learning.
        - Prétraitement du texte extrait avec NLP.
        - Classification du contenu à l'aide d’un LLM combiné à l’API Google Maps.
        - Développement d’une API et d’un pipeline CI/CD pour un déploiement automatisé sur AWS.
        - Intégration de l'application dans une application mobile FlutterFlow (listes, carte       authentification...).
    
    4- Expérience freelance n2 : Un projet en Gen AI et LLM dans une startup nomée Blumana, incubé à station F chez telecom. 
    De janvier 2025 à Mars 2025. 

    Missions : 
    - Scraping des avis utilisateurs sur le web et stockage dans une base de données PostgreSQL.
    - Prétraitement du texte avec le NLP.
    - Utilisation du Machine Learning et des LLM pour :
    - Transformer les feedbacks en insights exploitables.
    - Regrouper les avis en fonction des sentiments et des thèmes clés.
    - Optimiser l’ingénierie des prompts pour les LLMs.


Les compétences de Charles Doux sont donc : 
Compétences Techniques: 
    IA & Machine Learning & Deep Learning
        - Apprentissage supervisé et non supervisé (Feature Engineering, Optimisation de modèles)
        - Deep Learning (Vision par ordinateur, Apprentissage auto-supervisé)
        - Reinforcement Learning (Q-Learning)
        - LLMs (RAG, Prompt Engineering, Fine-tuning, Transformers, Bases vectorielles)
        - Déploiement API & MLOps (Mise en production et scalabilité des modèles)
    Big Data & Orchestration
        - ETL & Big Data (Data Lake, Data Warehouse, Traitement batch)
        - Orchestration & Monitoring (Dagster, Airflow)
    Langages de Programmation
        - Python, SQL, JavaScript
    Frameworks & Bibliothèques
        - Pandas, Polars, Spark, Scikit-learn, Langchain, TensorFlow, PyTorch, OpenCV, spaCy, DuckDB, Hugging Face Transformers, Matplotlib (Pyplot)
    Cloud & Bases de Données
        - AWS (S3, EMR, EC2, Watch, IAM...)
        - OVH, Supabase, PostgreSQL, ChromaDB
    DevOps & Déploiement
        - GitHub, CI/CD, Docker
    Développement Web & Mobile
        - VueJS, NodeJS, WordPress, FlutterFlow


Compétences Soft Skills
    - Résolution de problèmes & Pensée critique
    - Communication avec les parties prenantes
    - Collaboration & travail en équipe pluridisciplinaire
    - Autonomie & prise d’initiative

Actuellement Charles est à la recherche d'un CDI ou d'une nouvelle missions en Freelance dans le dommaine de l'intelligence Artificielle, il est disponible maintenatn.


Ingénieur IA et Data, il possède une solide expertise dans la conception et la mise en œuvre de solutions d’intelligence artificielle. Son savoir-faire se situe à l’intersection de l’intelligence artificielle et de l’ingénierie des données, ce qui lui permet non seulement de développer des modèles d’IA adaptés aux besoins spécifiques, mais aussi de garantir la qualité, la scalabilité et la fiabilité des pipelines de données qui les alimentent.

Au cours des derniers mois, il a  participé à des projets en entreprise exploitant le machine learning et l’IA générative dans divers domaines, notamment le NLP et la Computer vision. Son stage chez Calibsun lui  a permis de me spécialiser dans la création de pipelines ETL robustes pour les problématiques Big Data, l’optimisation des architectures de données et l’assurance de flux de données efficaces pour les applications basées sur l’IA. Cette double expertise lui permet de combler l’écart entre le développement IA, la qualité des données et son deploiement en production, garantissant ainsi que les modèles ne soient pas seulement théoriquement performants, mais aussi pratiques et évolutifs.

Passionné par l’innovation, il aime résoudre des problématiques complexes en combinant mathématiques appliquées, traitement des données et ingénierie logicielle. Fort d’une base technique solide développée dès son plus jeune âge, il aborde chaque projet avec rigueur analytique, esprit critique et créativité. 

Son aisance en communication, sa capacité à collaborer avec des équipes multidisciplinaires et son esprit d’initiative lui permettent de livrer des solutions exploitables et de haute qualité dans les délais impartis.

Son expérience couvre plusieurs secteurs, notamment la santé, les énergies renouvelables, le tourisme et le marketing, où j’ai su transformer des analyses techniques en valeur métier concrète.

Charles parle couramment anglais, ai un niveau conversationnel en espagnol et suis ouvert aux opportunités où


Ses passions sont sport, guitare, photographie.

Ses expérances salariales sont entre 40 000 et 45 000 euros par an en CDI.
Ou bien 400 euros par jours en freelance. 
"""

    prompt = f"""
        Tu es un Agent pour parler de Charles Doux, ton créateur, Charles t'as créer en une après-midi pour aider les recruteurs afin d'en apprendre plus sur Charles. Tu parles de manière professionnel et donne des réponses claire et concises.
        Tu es capable, de résumer le parcour et l'expérience de Charles.
        Fait des réponses courte sauf si on te demande plus de détail.

        Ne met pas de mots en gras, ni de titre, n'hésite pas à sauter des lignes pour la clarté et ajouter des -.

        SI on te parle d'expérience, fais référence à mes éxpériences en stage et en freelance. 
        Tu es aussi capable de donner une note de match face à une fiche de poste si on t'en envoie une fiche de poste.

        Tu t’appuieras sur ce contexte : 


        ————————-

        {context_text}

        ————————-

        Voici le texte reçu : 

        {query_text}

        Réponse : 
"""
    response = run_mistral(prompt)
    print(response)
    # cleaned_response = response.strip("```json").strip("```").strip("```python").strip()
    # cleaned_response = cleaned_response.replace("true","True")
    # cleaned_response = cleaned_response.replace("false","False")
    # response_dict = eval(response)
    # if response_dict['should_learn'] == True:
    save_to_chroma(chunks=None, text=[query_text])
    return {"response": response, "context": context_text}
