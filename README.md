
# Création d'un Produit data pour la  satisfaction client

L’objectif ici est de prédire le score de satisfaction client pour une commande donnée en fonction de fonctionnalités telles que
le statut de la commande,
le prix,
le paiement, etc.

Afin d'y parvenir dans un scénario réel, nous devons créer un produit data prêt pour la production : pipeline pour prédire le score de satisfaction client pour la prochaine commande ou achat.


L'objectif de ce repository est de démontrer comment la création d’un projet data suivant les bonnes pratiques de la création d’un produit data  apporte de la valeur à vos métiers et donc votre entreprise

En vous proposant un cadre et un modèle sur lesquels baser votre propre travail.


Les notions vu  durant la phase de preprocessing
** Héritage en Python : Avez-vous déjà eu plusieurs classes avec des attributs et des méthodes similaires ?

Si tel est le cas, utilisez l'héritage pour organiser vos classes.

L'héritage nous permet de définir une classe parent et des classes enfants.

Une classe enfant hérite de toutes les méthodes et attributs de la classe parent.

** Classes abstraites : déclarer des méthodes sans implémentation

Parfois, vous souhaiterez peut-être que différentes classes utilisent les mêmes attributs et méthodes.

Mais la mise en œuvre de ces méthodes peut être légèrement différente dans chaque classe.

Un bon moyen d’implémenter cela consiste à utiliser des classes abstraites.

Une classe abstraite contient une ou plusieurs méthodes abstraites.

Une méthode abstraite est une méthode déclarée mais qui ne contient aucune implémentation. La méthode abstraite nécessite des sous-classes pour fournir des implémentations.



# Types de tests
Il est important de noter que de nombreux projets auront des besoins différents en termes de tests,

cette division peut donc ne pas être précise pour tous les projets, mais en termes généraux, les principaux types de tests dans MLOps sont :

Tests de logiciels : comprend les tests qui garantissent que le code respecte les exigences du projet. Il s'agit du type de tests normalement implémentés dans DevOps, tels que les tests unitaires, les tests d'intégration, les tests système et les tests d'acceptation.

Tests de modèle : comprend les tests qui définissent que le modèle fonctionne correctement, tels que tester qu'il peut être entraîné ou qu'il peut obtenir un score minimum lors d'une évaluation.

Tests de données : comprend les tests qui vérifient l’exactitude des données. Cela dépend fortement des exigences du projet et peut viser à garantir que l'ensemble de données analysé suit un schéma, contient suffisamment de données, etc. Beaucoup de choses peuvent être testées ici en fonction des besoins de l'équipe.

## Tests dans notre cas

Ces trois fonctions semblent être des tests unitaires conçus pour tester différents aspects d'un pipeline de préparation et de nettoyage des données. Décomposons chaque fonction :

1. data_test_prep_step :

Cette fonction teste la forme des ensembles de formation et de test après l'étape de nettoyage des données.
Il utilise des assertions pour vérifier si les formes de X_train, y_train, X_test et y_test correspondent aux valeurs attendues.

Si l'une des assertions échoue, elle enregistre une erreur à l'aide de logging.info et déclenche une exception pytest.fail.

2. check_data_leakage :

Cette fonction teste s'il y a une fuite de données entre les données de train et de test.
Il vérifie s'il existe une intersection entre les indices de X_train et X_test.
S'il y a une intersection, il enregistre une erreur à l'aide de logging.info et déclenche une exception pytest.fail.


3. test_ouput_range_features :

Cette fonction teste la plage de sortie de la variable cible (review_score) dans le DataFrame d'entrée.
Il affirme que la valeur maximale de review_score est inférieure ou égale à 5 et que la valeur minimale est supérieure ou égale à 0.

Si l'une des assertions échoue, elle enregistre une erreur à l'aide de logging.info et déclenche une exception pytest.fail.
Ces fonctions utilisent la bibliothèque pytest pour tester et journaliser à partir du module de journalisation des messages informatifs. Si l'une des assertions échoue pendant les tests, une exception pytest.fail sera levée, indiquant que le scénario de test spécifique a échoué.

Pour exécuter ces tests, vous utiliserez généralement un framework de test tel que pytest.

Lorsque vous exécutez la suite de tests, pytest découvrira et exécutera ces fonctions, signalant tout échec ou erreur rencontré lors des tests.
