# Boris-C5-Project

## Description

Ce projet implémente et compare différentes méthodes numériques pour simuler le mouvement de particules chargées dans des champs électriques et magnétiques, avec un focus particulier sur l'algorithme de Boris pour la dynamique relativiste et non-relativiste.

## Contenu du projet

### Fichiers principaux

- **`orbit_comparison.py`** : Fichier principal contenant l'implémentation de l'algorithme de Boris relativiste et les solutions analytiques correspondantes
- **`simple-boris-pusher.py`** : Implémentation simple de l'algorithme de Boris pour des cas non-relativistes
- **`Gaussian_laser.py`** : Simulation d'interaction particule-laser avec profil gaussien

### Algorithmes implémentés

#### 1. Algorithme de Boris standard
- Méthode numérique stable pour l'intégration des équations du mouvement
- Préservation de l'énergie dans le cas de champs purement magnétiques
- Adapté pour les particules non-relativistes

#### 2. Algorithme de Boris relativiste
- Extension relativiste utilisant le facteur de Lorentz γ = 1/√(1-(v/c)²)
- Intégration leapfrog pour une meilleure stabilité numérique
- Gestion des particules à haute énergie (jusqu'à 90% de la vitesse de la lumière)

#### 3. Solutions analytiques
- Solutions exactes pour validation des méthodes numériques
- Cas du mouvement cyclotron dans un champ magnétique uniforme
- Cas des champs croisés E⃗ × B⃗ avec effets relativistes

## Fonctionnalités

### Visualisation et analyse
- Comparaison trajectoires numériques vs analytiques
- Calcul et affichage des erreurs relatives
- Graphiques 2D et 3D des trajectoires
- Conservation de l'énergie (classique et relativiste)
- Analyse des composantes x, y, z séparément

### Cas d'étude
1. **Mouvement cyclotron simple** : Particule dans un champ magnétique uniforme
2. **Champs croisés** : Particule dans des champs E et B perpendiculaires
3. **Régime relativiste** : Électrons à haute énergie avec corrections relativistes

## Utilisation

### Prérequis
```bash
pip install numpy matplotlib scipy numba
```

### Exécution
```python
# Pour la comparaison Boris vs analytique
python orbit_comparison.py

# Pour une simulation simple
python simple-boris-pusher.py
```

### Paramètres principaux
- **Particule** : Électron (charge q = -e, masse m = mₑ)
- **Champs** : E⃗ et B⃗ configurables
- **Conditions initiales** : Position r₀ et vitesse v₀
- **Intégration** : Pas de temps dt, temps final tf

## Physique

### Équations du mouvement
L'algorithme de Boris résout les équations de Lorentz :
```
m(dv⃗/dt) = q(E⃗ + v⃗ × B⃗)
dr⃗/dt = v⃗
```

### Corrections relativistes
Pour les hautes énergies, le facteur γ corrige la masse effective :
```
γ = 1/√(1-(v/c)²)
m_eff = γm₀
```

## Résultats

Le projet génère :
- Graphiques de comparaison trajectoires/erreurs
- Validation de la conservation d'énergie
- Analyse de la précision numérique
- Fichiers PNG des visualisations

## Références

- Boris, J.P. (1970). "Relativistic plasma simulation-optimization of a hybrid code"
- Jackson, J.D. "Classical Electrodynamics" (voir `Page Jackson.pdf`)
- Documentation sur les méthodes de Vay (voir `vay_relativistic.pdf`)

## Structure du dépôt

```
Boris-C5-Project/
├── orbit_comparison.py          # Code principal
├── simple-boris-pusher.py       # Implémentation simplifiée  
├── Gaussian_laser.py           # Interaction laser-particule
├── analytical/                 # Solutions analytiques
├── Page Jackson.pdf           # Référence théorique
├── vay_relativistic.pdf       # Méthodes relativistes
├── validation_jackson_solutions.png  # Résultats de validation
├── transverse_motion.gif      # Animation du mouvement
└── README.md                  # Ce fichier
```

## Auteurs

Projet dans le cadre du cours C5 - Physique numérique et simulation.