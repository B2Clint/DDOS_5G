# DDOS_5G
Détection des attaques DDoS pour les réseaux 5G à l'aide d'un modèle CNN basé sur l'apprentissage fédéré à chiffrement asymétrique. 

<img width="1280" height="586" alt="chrono_5g" src="https://github.com/user-attachments/assets/0604aa34-fe8e-49e7-9958-12d3cd7a6e5d" />
<img width="2431" height="1064" alt="image" src="https://github.com/user-attachments/assets/10cac82f-271e-4bb3-80eb-8500693a2412" />

<br>

Problématique :
Le Slicing ou découpage du réseau 5G en tranches est l'une des technologies clés permettant d'offrir des ressources dédiées à différentes applications sur un même réseau physique virtualisée. Cependant, une attaque par déni de service (DoS) ou par déni de service distribué (DDoS) peut gravement endommager les performances des tranches de réseau telles que la bande passante et la latence. Face à ce problème, de nouvelles approches émergentes telles que les réseaux neuronaux convolutifs (CNN) s’avèrent être des solutions prometteuses pour renforcer la sécurité des réseaux.
<br>
Objectif :
Proposer un modèle efficace de détection d'attaques par déni de service distribué (DDoS) grâce à un modèle CNN basé sur l'apprentissage fédéré pour renforcer la sécurité des réseaux.
Le dataset utilisé pour l'entraînement et les tests du modèle est disponible sur IEEE DataPort - DoS/DDoS Attack Dataset for 5G Network Slicing.
Rendu attendu :
La procédure pourra commencer par la sélection des caractéristiques pertinentes (features), la sélection du modèle, la préparation des données d'apprentissage et l’entrainement du modèle en utilisant un algorithme d’optimisation.
A terme, le modèle devra être apte à identifier les attaques DDoS.
Les résultats obtenus via ce modèle devront être comparés à ceux du modèle cité en lien (Ce dernier devra d’abord être reproduit).
-	Un graphique des performances de précision (accuracy) de l’entrainement et de la validation est attendu.
-	Un graphique des performances de perte (loss) d'entraînement et de validation est également attendu.
-	Une matrice de confusion pour la détection des attaques et du trafic normal.
-	Les indicateurs importants tels que l'exactitude, la précision, la sensibilité, la spécificité et le score F1 doivent être évalués.
-	Un schéma explicatif global du modèle ainsi que l’algorithme d’optimisation utilisé.
<br>
pre-réquis:
tensorflow>=2.12.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0

seaborn>=0.12.0
cryptography>=41.0.0
