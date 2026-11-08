# Fiche de données (data card) — Modération de contenu multimodale

Projet : classification de publications **image + texte** (détection de contenu haineux) — module Deep Learning.
Les chiffres ci-dessous proviennent de l'exécution réelle du pipeline (`dvc repro`) ; le détail se trouve dans
`reports/data_report.html`, `data/logs/counters.csv` et `data/logs/run_summary.json`.

## 1. Sources

| Source | Contenu | Modalités | Volume brut | Licence / conditions | Accès |
|---|---|---|---|---|---|
| Hateful Memes (Meta AI, version Kaggle `parthplc/facebook-hateful-meme-dataset`) | mèmes labellisés haineux / non haineux | image + texte | 10 000 mèmes (train / dev / test) | licence propre au jeu de données (fichier `LICENSE.txt` livré avec l'archive) : usage de recherche, **à relire avant toute réutilisation** | Kaggle CLI |
| Sentiment140 (Kaggle `kazanova/sentiment140`) | tweets annotés par sentiment (0 négatif, 4 positif) | texte seul | 1 600 000 tweets | pas de licence explicite sur la page Kaggle ; données issues de Twitter (conditions de la plateforme) ; citation : Go, Bhayani, Huang (2009) | Kaggle CLI |
| Reddit (API officielle, via PRAW) | posts image + titre | image + texte | non collecté dans cette version | conditions d'utilisation de l'API Reddit ; connecteur testé et désactivé (`ingestion.reddit.enabled: false`) | OAuth2 |

## 2. Volumes après préparation (manifeste `data/processed/manifest.parquet`)

- 108 394 échantillons : 10 000 mèmes (Hateful Memes) et 98 394 tweets (Sentiment140, sous-échantillon équilibré de 100 000 puis nettoyage).
- Hateful Memes : 0 rejet (images toutes lisibles, aucun doublon exact), 6 651 groupes anti-fuite (1 634 groupes de plusieurs mèmes, le plus grand en contient 78).
- Découpage Hateful Memes labellisé (`test.jsonl` public sans labels, donc non utilisé comme jeu de test) : train 7 200 / val 900 / test 900, prévalence « haineux » de 36,7 % dans chaque ensemble.
- Sentiment140 : 20 140 tweets rejetés sur 1 600 000 (1,26 %) : identifiants ou textes dupliqués.

## 3. Prétraitements appliqués

1. Validation de schéma (Pandera) ; lignes invalides rejetées et journalisées avec leur raison.
2. Synchronisation image-texte dans les deux sens (chaque image du dossier a une ligne ou un rejet journalisé).
3. Nettoyage du texte : NFKC, entités HTML, caractères invisibles, URL -> `[URL]`, mentions -> `@user` ; texte brut conservé.
4. Détection de langue (`langdetect`, graine fixée) ; « und » si le texte est trop court.
5. Empreintes d'image : SHA-256 et hash perceptuel (pHash) ; doublons exacts (image + texte) supprimés.
6. Groupes anti-fuite : images quasi identiques (distance de Hamming <= 4) ou textes identiques ; découpage **stratifié par groupe**.
7. Table de correspondance des labels (haine / sentiment) versionnée : `reports/label_mapping.csv`.

## 4. Biais et limites connus

- **Annotation :** les labels Hateful Memes sont attribués par des annotateurs externes à Meta ; nous n'avons pas pu mesurer l'accord inter-annotateurs. Les labels Sentiment140 sont **automatiques** (émoticônes) donc bruités.
- **Cas confondants :** Hateful Memes contient des exemples construits pour que ni l'image ni le texte seuls ne suffisent ; ceci est voulu (et utile pour étudier la fusion) mais n'est pas représentatif d'un flux réel.
- **Langue :** données très majoritairement en anglais ; les performances ne se généralisent pas aux autres langues sans adaptation.
- **Prévalence :** 36,7 % de contenus haineux dans le jeu étiqueté, bien plus élevé que dans un flux réel : les métriques de précision ne se transposent pas telles quelles en production.
- **Période et source :** tweets de 2009 (Sentiment140), mèmes d'une seule plateforme : décalage de domaine.
- **Test public :** `test.jsonl` n'a pas de labels ; le jeu de test utilisé est issu d'un redécoupage de `train.jsonl` (+ `dev.jsonl`), non comparable aux scores officiels de la compétition.
- **Contenu sensible :** le jeu contient des contenus haineux ; accès à restreindre aux besoins du projet, ne pas redistribuer les images.

## 5. Reproductibilité

Code versionné avec Git (tag `task3`), données et artefacts versionnés avec DVC (`dvc.lock`), configuration dans
`configs/params.yaml`, graines fixées (`seed: 42`), versions des paquets dans `data/logs/run_summary.json`.
Reconstruction : `dvc repro` ; récupération des données : `dvc pull` (voir le README).
