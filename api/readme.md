# Catégorisez automatiquement des questions
### Ingénieur Machine Learning - OpenClassRooms / CentraleSupelec

Stack Overflow est un site célèbre de questions-réponses liées au développement informatique.

Pour poser une question sur ce site, il faut entrer plusieurs tags afin de retrouver facilement la question par la suite. Pour les utilisateurs expérimentés, cela ne pose pas de problème, mais pour les nouveaux utilisateurs, il serait judicieux de suggérer quelques tags relatifs à la question posée.

Amateur de Stack Overflow, qui vous a souvent sauvé la mise, vous décidez d'aider la communauté en retour. Pour cela, vous développez un système de suggestion de tags pour le site. Celui-ci prendra la forme d’un algorithme de machine learning qui assignera automatiquement plusieurs tags pertinents à une question.

## Les données
Stack Overflow propose un outil d’export de données, [**StackExchange Data Explorer**](https://data.stackexchange.com/stackoverflow/query/new), qui recense un grand nombre de données authentiques de la plateforme d’entraide. 

Par défaut, il y a une limite sur le temps d'exécution de chaque requête SQL, ce qui peut rendre difficile la récupération de toutes les données d'un coup. D’autre part, les questions et tags associés ne sont pas toujours pertinents ou sont trop peu nombreux, voire inexistants. Pour récupérer plus de résultats pertinents, pensez à faire des requêtes avec des contraintes sur certaines données, pour filtrer par exemple les questions les plus vues, mises en favori ou jugées pertinentes par les internautes, ayant reçu une réponse et ayant au moins 5 tags.

#### Par exemple :
`SELECT TOP 500000 Title, Body, Tags, Id, Score, ViewCount, FavoriteCount, AnswerCount
FROM Posts 
WHERE PostTypeId = 1 AND ViewCount > 10 AND FavoriteCount > 10
AND Score > 5 AND AnswerCount > 0 AND LEN(Tags) - LEN(REPLACE(Tags, '<','')) >= 5`

## Contraintes 

* Appliquer des méthodes d’extraction de features spécifiques des données textuelles.
* Mettre en œuvre une approche non supervisée afin de proposer des mots clés.
* Mettre en œuvre une approche purement supervisée et comparer les résultats avec l’approche non supervisée. Plusieurs méthodes d’extraction de features seront testées et comparées ; au minimum :
  + une approche de type bag-of-words ;
  + 3 approches de Word/Sentence Embedding : Word2Vec (ou Doc2Vec, Glove…), BERT et USE. 
* Mettre en place une méthode d’évaluation propre, avec une séparation du jeu de données pour l’évaluation.
* Utiliser un logiciel de gestion de versions, par exemple Git, pour suivre les modifications du code final à déployer

# Installation et utilisation de l'API de prédiction de tags

L'API de prédiction de tags est conçue pour prédire des tags à partir de commentaires textuels. Voici les étapes pour installer et utiliser cette API :

## Installation
1. Assurez-vous d'avoir Python 3.x installé sur votre système.
2. Téléchargez les fichiers de l'API de prédiction de tags.
3. Installez les dépendances requises en exécutant la commande suivante dans un terminal :

`pip install -r requirements.txt`


## Utilisation
1. Lancez l'API en exécutant le fichier `app.py` dans un terminal :

`python app.py`

2. L'API sera accessible à l'adresse `http://localhost:5000`.
3. Pour vérifier que l'API fonctionne correctement, ouvrez un navigateur web et accédez à l'URL `http://localhost:5000/`. Vous devriez voir le message "Hello, I work!" s'afficher, ce qui indique que l'API est en cours d'exécution.
4. Pour prédire des tags à partir d'un fichier CSV, utilisez l'URL `http://localhost:5000/predict_tags` avec une requête POST. Vous pouvez utiliser des outils tels que Postman ou cURL pour envoyer des requêtes POST à l'API.
- Incluez le fichier CSV en utilisant la clé `file` dans la requête POST.
- Assurez-vous que le fichier CSV contient les colonnes "Body" (commentaires textuels) et "Tags" (tags correspondants).
- Les tags doivent être séparés par des virgules s'ils sont multiples pour un commentaire.
5. L'API renverra une réponse JSON contenant les prédictions de tags pour chaque commentaire.

Notez que vous pouvez personnaliser le modèle de prédiction en remplaçant le fichier du modèle `classification_genginemodel_fabrice_deprez_OPP5_062023.pkl` par votre propre modèle entraîné.

Assurez-vous également d'avoir les droits appropriés pour accéder aux fichiers et aux ports requis par l'API.

N'hésitez pas à ajuster les paramètres de l'API en fonction de vos besoins spécifiques.

# Déploiement
l'api est déployée sur http://kaneda.pythonanywhere.com/ via pythonAnywhere

utilisation : http://kaneda.pythonanywhere.com/predict_tags

repository : https://github.com/kaneda59/P5_Automatic_categorise_issues.git

vous pouvez tester l'api avec le fichier html : index.html
