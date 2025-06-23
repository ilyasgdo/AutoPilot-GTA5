# 🚗 AutoPilot-GTA5: Deep Reinforcement Learning for Autonomous Driving in GTA V

![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange)
![Status](https://img.shields.io/badge/status-WIP-yellow)

**AutoPilot-GTA5** est un projet de conduite autonome basé sur le **Deep Reinforcement Learning**, la **vision par ordinateur**, et un pipeline **CNN + LSTM**. L'entraînement se déroule dans **GTA V**, utilisé comme environnement de simulation réaliste.

---

## 🎯 Objectifs du projet

- 📹 Capturer les frames du jeu GTA V en temps réel
- 👁️ Analyser la scène avec **OpenCV** (lignes, feux, collisions) et **YOLOv8**
- 🧠 Prendre des décisions avec un agent **SAC** (Soft Actor-Critic)
- 📚 Entraîner un réseau **CNN + LSTM** pour intégrer la vision et la mémoire temporelle
- 🚦 Optimiser une fonction de récompense basée sur le respect du code de la route
- 🚘 Transférer l'agent sur une **voiture télécommandée réelle (phase 2)**

---

## 🧠 Architecture

```text
GTA V
 │
 ▼
[Frame Capture]
 │
 ▼
[OpenCV + YOLOv8] → Perception
 │
 ▼
CNN → LSTM → SAC Policy → Action (Tourner, Accélérer, Freiner)
 │
 ▼
Injection des actions dans GTA V
```

## 🔧 Technologies utilisées

| Composant | Techno |
|-----------|--------|
| Deep RL | PyTorch + SAC |
| Perception | OpenCV, YOLOv8 |
| Réseau neur. | CNN (ResNet) + LSTM |
| Environnement | GTA V (mod ou capture OBS) |
| Action agent | PyAutoGUI / Interface |
| Système | Python 3.10, CUDA |

## 📁 Structure du projet

```bash
AutoPilot-GTA5/
│
├── capture/               # Capture vidéo GTA
├── perception/            # OpenCV / YOLO
├── rl/                    # Réseaux et SAC
├── utils/                 # Fonction de récompense, logs
├── train.py               # Script d'entraînement
├── test.py                # Évaluation
├── config.yaml
└── README.md
```

## 🚀 Installation

```bash
git clone https://github.com/ton-utilisateur/AutoPilot-GTA5.git
cd AutoPilot-GTA5
pip install -r requirements.txt
```

⚠️ **Prérequis** : une carte GPU (ex: RTX 3080 Ti) avec CUDA installé.

## 🧪 Entraînement

```bash
python train.py --config config.yaml
```

Tu peux modifier les hyperparamètres dans `config.yaml`.

## 🧠 Fonctionnalités implémentées

- ✅ Capture de frames GTA V
- ✅ Traitement OpenCV des lignes et collisions
- ✅ Détection YOLO des véhicules, feux rouges, stops
- ✅ Modèle CNN + LSTM
- ✅ Algorithme SAC avec replay buffer séquentiel
- ✅ Fonction de récompense personnalisée
- ✅ Enregistrement du modèle PyTorch

## 📦 Fonctionnalités à venir

- 🔄 Fine-tuning YOLO sur des scènes GTA
- 📦 Export du modèle pour carte embarquée (Raspberry Pi / Jetson Nano)
- 🧭 Ajout de capteurs virtuels (raycasting LIDAR)
- 🏎️ Déploiement sur voiture télécommandée réelle

## 👨‍💻 Auteur

**Ilyas Ghandaoui** – [@ilyas-gh](https://github.com/ilyas-gh)

Étudiant ingénieur informatique, passionné de deep learning, robotique et IA embarquée.

## 📜 Licence

Ce projet est sous licence MIT.

---

🌟 **Star le projet si tu trouves ça cool !**

🚀 Pour toute contribution, suggestion ou collaboration, n'hésite pas à ouvrir une issue ou une pull request.
