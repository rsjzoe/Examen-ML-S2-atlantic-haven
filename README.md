# **Rapport de Projet — Atlantic Haven Hotels**

## **Examen Final Machine Learning & Data Science — M1**

Réalisé au sein de **ISPM — Madagascar** ([www.ispm-edu.com](https://www.ispm-edu.com))

---

### **1. Informations sur le Groupe**

- RAVALISON fanevampifaliana Josia,Numéro :56
- RABEARISOA Heriniaina Liantsoa, Numéro: 44
- RAMANANTENASOA Elissaha, Numéro: 42
- RANARISON Haingoniaiko Lucka, Numéro: 60
- RAZAFIMAHONONA Volasoa , Numéro : 50
- RASOARIJAONA Volatiana Zoé, Numéro : 43

---

### **2. Résumé du Travail**

#### Problématique

Atlantic Haven Hotels subit des annulations qui laissent des chambres invendues et perturbent la planification opérationnelle. L'enjeu est de prédire suffisamment tôt la cible `reservation_annulee` afin d'anticiper le risque, **sans pénaliser inutilement** les clients qui maintiendront leur séjour. On cherche donc un compromis maîtrisé entre faux positifs et faux négatifs, mesuré par le F1-score sur la classe « annulation ».

#### Méthodologie adoptée

EDA (cible, valeurs manquantes, signaux catégoriels, dérive temporelle) → feature engineering sans fuite (dates, historique client, prix, réservation directe) → **validation temporelle** (hold-out des 20 % les plus récents du train, trié par `date_reservation`) → baseline régression logistique → comparaison de trois familles (LogReg, RandomForest, HistGradientBoosting) sur **le même jeu de validation** → **réglage du seuil** par maximisation du F1 sur la courbe précision-rappel → sélection du modèle final sur F1 **et** stabilité → analyse d'erreurs et génération de `submission.csv`.

#### Résultats obtenus

Meilleur F1 sur le jeu de validation temporel : **0,479** (RandomForest, seuil 0,386), avec précision 0,347, rappel 0,772 et ROC-AUC 0,650. Découverte importante : le signal est essentiellement **additif** — les modèles à arbres ne dépassent pas nettement la régression logistique (AUC ≈ 0,64–0,66 pour toutes les familles), et le principal levier de gain est le **réglage du seuil** (+1,8 pt de F1 par rapport au seuil 0,5), devant le feature engineering.

#### Mots-clés

classification binaire, annulation hôtelière, validation temporelle, F1-score, réglage du seuil, feature engineering, RandomForest, fuite de données.

---

### **3. Contenu du Repository**

- **notebook.ipynb** : EDA, prétraitement, modélisation, évaluation et soumission, exécutable de bout en bout ;
- **submission.csv** : prédictions sur `reservations_test.csv` (2 000 lignes, 3 colonnes, ordre des identifiants préservé) ;
- **README.md** : présent rapport ;
- Lien vers la vidéo de présentation :https://drive.google.com/file/d/18-FyqE21SdHp53bqcKSEbkrlQDnQ2YNm/view?usp=drive_link

---

### **4. Résultats de Modélisation**

Résultats sur **le même jeu de validation temporelle** (20 % les plus récents du train, du 2024-11-28 au 2025-05-24).

| Modèle                                 | Paramètres principaux                                                 |  F1-score | Précision | Rappel | ROC-AUC |
| -------------------------------------- | --------------------------------------------------------------------- | --------: | --------: | -----: | ------: |
| Régression logistique — baseline       | `class_weight=balanced`, seuil 0,5                                    |     0,460 |     0,364 |  0,627 |   0,662 |
| Régression logistique — seuil optimisé | `class_weight=balanced`, seuil 0,340                                  |     0,478 |     0,328 |  0,881 |   0,662 |
| HistGradientBoosting                   | `max_iter=400, lr=0,05, depth=6`, seuil 0,325                         |     0,469 |     0,340 |  0,755 |   0,635 |
| **Modèle final — RandomForest**        | `n_estimators=400, max_depth=12, min_samples_leaf=5`, seuil **0,386** | **0,479** |     0,347 |  0,772 |   0,650 |

**Seuil de décision retenu :** **0,386** (maximise le F1 sur la classe annulation, choisi sur la validation, jamais sur le test).

**Justification du choix du modèle final :**

RandomForest est retenu non seulement pour son meilleur F1, mais surtout pour sa **stabilité** : testé sur trois points de coupe temporels (75/80/85 %), il donne 0,484 / 0,479 / 0,481 (écart 0,005), contre 0,479 / 0,478 / 0,470 pour la régression logistique (écart 0,009). Il conserve donc une bonne performance sur les périodes les plus récentes — exactement ce qu'exige un test futur. Son importance de variables reste interprétable. La régression logistique demeure une alternative crédible (meilleur AUC, plus simple) si l'interprétabilité prime sur le dernier demi-point de F1.

---

### **5. Réponses aux Questions d'Analyse**

#### **Q1. Pourquoi le F1-score plutôt que l'accuracy ?**

La cible est déséquilibrée (25,8 % d'annulations). Un modèle trivial prédisant « jamais d'annulation » atteindrait 74,2 % d'accuracy tout en étant **inutile** (rappel nul sur la classe qui nous intéresse). Le F1-score sur la classe annulation combine précision et rappel : il ne récompense un modèle que s'il **détecte effectivement** les annulations sans multiplier les fausses alertes. C'est la métrique alignée avec l'objectif métier.

#### **Q2. Faux positif ou faux négatif : le plus grave ?**

- **Faux positif** : on prédit une annulation pour une réservation qui sera en réalité honorée. Risque : sur-réservation, sollicitations commerciales inutiles, dégradation de l'expérience d'un bon client.
- **Faux négatif** : on prédit un maintien pour une réservation qui sera annulée. Risque : chambre invendue, perte sèche de revenu, planification faussée.

Réponse nuancée : le **faux négatif** est généralement plus coûteux (revenu perdu difficilement récupérable), ce qui justifie un seuil bas privilégiant le rappel (0,386 → rappel 0,77). Mais cela n'a de sens **que si** l'action déclenchée est douce (rappel, incitation) et non l'annulation automatique : sinon les faux positifs deviennent très coûteux en satisfaction client. Le bon seuil dépend donc du coût réel de chaque action.

#### **Q3. Variables de feature engineering les plus utiles.**

Construction (sans fuite, uniquement des colonnes connues à la réservation) : `taux_annul_hist = annulations_passees / reservations_passees` (propension historique), `reservation_directe` (drapeau `agent_id` vide), variables calendaires (`arr_mois`, `arr_trimestre`, `arr_jour_sem`), `cout_total_estime = prix_moyen_nuit × nuits`, `prix_par_personne`, `a_enfants`, buckets de délai.

Gain observé : **modeste et honnête**. Sur la régression logistique, l'apport est négligeable (les colonnes brutes captent déjà le signal linéairement : +0,0003 de F1). Sur les arbres, l'apport est marginal sur le F1 (±0,002) mais légèrement positif sur l'AUC. Les variables engineerées **les plus utiles en importance de permutation** restent `reservation_directe` et les variables calendaires ; le gros du pouvoir prédictif vient toutefois de variables commerciales brutes (`tarif_remboursable`, `type_acompte`). La leçon : sur ces données, le **feature engineering n'est pas le principal levier** — le réglage du seuil l'est.

#### **Q4. Pourquoi un découpage aléatoire serait trompeur ?**

Les fichiers sont ordonnés dans le temps et le **test est postérieur au train**. Un split aléatoire laisserait fuiter des réservations « du futur » dans l'entraînement et évaluerait le modèle sur des périodes déjà vues, donnant une estimation **optimiste** non représentative du déploiement réel. Notre stratégie : tri par `date_reservation`, puis hold-out des **20 % les plus récents** (train fold jusqu'au 2024-11-28 ; validation du 2024-11-28 au 2025-05-24). Le seuil et le choix du modèle sont fixés sur cette validation ; on vérifie aussi la stabilité à 75/80/85 %.

#### **Q5. Profils de réservation les plus associés aux annulations.**

_(Circonstances observables et interactions de variables, pas des populations.)_

- Réservations **sans acompte** (`type_acompte = aucun` : 34 % d'annulation) et **tarif remboursable** (31 % vs 14 % pour non-remboursable) — faible coût de renoncement.
- **Canal en ligne / agence** (30 % / 28 %) vs canal entreprise (14 %) — engagement plus faible.
- **Délai de réservation long** combiné à un tarif remboursable — plus de temps et de flexibilité pour changer d'avis.
- Segments **groupe** (31 %) et **famille** (28 %), plus sensibles aux aléas d'organisation.

Ce sont des **combinaisons de conditions commerciales**, pas des caractéristiques intrinsèques de régions ou de clients.

#### **Q6. Valeurs manquantes et catégories jamais observées.**

Toutes les imputations sont **apprises dans le pipeline sur le fold d'entraînement uniquement**, puis appliquées à la validation et au test (aucune statistique du test ne sert à l'apprentissage → pas de fuite). Numériques : imputation par la **médiane** (`SimpleImputer`). Catégorielles : imputation par une catégorie **`"Inconnu"`**. `agent_id` vide est traité non comme un manquant mais comme l'information « **réservation directe** ». Catégories jamais vues à l'entraînement : le `OneHotEncoder` est réglé avec `handle_unknown="ignore"` (elles sont encodées en vecteur nul plutôt que de faire échouer la prédiction).

#### **Q7. Action recommandée en cas de forte probabilité d'annulation.**

Ne **jamais annuler automatiquement**. Actions proportionnées et graduées : (1) rappel/e-mail de confirmation courtois, (2) incitation douce à sécuriser la réservation (petit avantage, `surclassement`, flexibilité), (3) pour les probabilités très élevées, proposer un acompte partiel ou un tarif préférentiel non-remboursable, (4) côté opérationnel, ajuster prudemment la sur-réservation sur les créneaux à fort risque agrégé. Le score sert d'**aide à la décision**, pas de couperet.

#### **Q8. Performances comparables selon les régions / destinations ?**

Le F1 par région varie de **0,40 (Campania)** à **0,55 (Sicilia)** sur la validation ; par type de destination, de 0,40 (`urbaine_cotiere`) à 0,55 (`insulaire_mixte`). Ces écarts s'expliquent en partie par le **taux d'annulation local** et surtout par la **taille des sous-groupes** : Sardegna (n=67) ou Puglia (n=99) donnent des F1 très bruités et non fiables. Limite honnête : nous ne disposons pas d'assez de volume par petite région pour conclure à une inéquité structurelle ; il faudrait davantage de données récentes par région pour trancher.

#### **Q9. Analyse des erreurs.**

**5 faux positifs** (prédit annulé, réellement maintenu) — R002759, R006211, R004523, R005310, R001601 : tous partagent le **profil à risque** (`type_acompte = aucun`, `tarif_remboursable = oui`, plateforme en ligne, délai long 84–105 j). Le modèle applique la règle générale du profil, mais ces clients ont honoré. → le modèle **sur-généralise** le profil commercial.

**5 faux négatifs** (prédit maintenu, réellement annulé) — R001185, R009204, R008135, R001223, R002592 : tous du **profil « sûr »** (`type_acompte = total/partiel`, `tarif_remboursable = non`, canal `site_hotel`/`entreprise`, délai court). Ils ont annulé malgré tout → annulations **idiosyncratiques** que les variables disponibles ne permettent pas d'anticiper.

Raisons probables : le signal utile est concentré sur 2–3 variables commerciales ; les annulations « contre-profil » relèvent d'événements non observés (imprévu personnel, changement de plan). **Pistes d'amélioration** : enrichir les données (météo/événements locaux à la date d'arrivée, historique de comportement plus fin, délai avant annulation), tester une calibration des probabilités et un seuil dépendant du coût métier par segment, et collecter davantage de volume par petite région.

---

### **6. Conclusion et Recommandations**

Le modèle RandomForest atteint un F1 de 0,479 sur une validation temporelle réaliste, avec un rappel élevé (0,77) : il détecte la majorité des annulations au prix d'une précision modérée (0,35). Le signal des données étant essentiellement additif et plafonné (AUC ≈ 0,65), les gains viennent surtout du **réglage du seuil** et d'un traitement propre des manquants, plus que d'un feature engineering sophistiqué. Limites : précision faible (beaucoup de fausses alertes), petites régions peu fiables, plafond de séparabilité.

**Recommandation opérationnelle finale :** déployer le modèle comme **système d'alerte** (score de risque) et non de décision automatique. Concentrer les actions douces (rappels, incitations à sécuriser) sur les réservations à forte probabilité, en gardant un humain dans la boucle. Réévaluer le seuil selon le coût réel constaté d'un faux positif vs faux négatif, et ré-entraîner périodiquement sur les données récentes pour suivre l'évolution de la demande.

---

### **7. Reproductibilité**

- version de Python : **3.12.3**
- principales bibliothèques : **scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4**
- graine aléatoire : **`RANDOM_STATE = 42`** (fixée pour numpy et tous les modèles)
- procédure d'exécution : placer `reservations_train.csv` et `reservations_test.csv` à côté du notebook, puis exécuter toutes les cellules dans l'ordre (`Run All`) ; `submission.csv` est écrit à la dernière cellule
- durée approximative d'entraînement : **~40 secondes** (exécution complète du notebook sur noyau vierge)
- environnement utilisé : local / Google Colab / Kaggle (aucune dépendance externe au-delà de scikit-learn)

---

### **8. Bibliographie**

- Documentation scikit-learn — `RandomForestClassifier`, `ColumnTransformer`, `precision_recall_curve`, `permutation_importance`.
- Pedregosa et al., _Scikit-learn: Machine Learning in Python_, JMLR, 2011.
- Énoncé et dictionnaire de données du hackathon Atlantic Haven Hotels (ISPM, 2025–2026).
- Outil d'IA générative (assistant Claude) : appui à la structuration du pipeline, au débogage et à la rédaction ; l'ensemble des choix méthodologiques et des résultats a été vérifié par exécution reproductible.
