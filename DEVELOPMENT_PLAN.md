# 📋 Plan de Développement - AutoPilot-GTA5

> Guide détaillé pour l'implémentation du système de conduite autonome dans GTA V

---

## 🔹 Phase 1 – Pipeline de capture & vision

### 1.1 - 🎥 Capture vidéo GTA V

**📌 Objectif :** extraire les frames du jeu GTA V en temps réel

**✅ Méthodes possibles :**
- `pyautogui.screenshot()` ou `mss` pour capturer l'écran
- Utiliser **OBS** + `imageio` si besoin de performance optimale

**📁 Fichier :** `capture/screen_capture.py`

```python
import pyautogui
import numpy as np
import cv2

def get_frame():
    """Capture une frame de GTA V et la redimensionne"""
    img = pyautogui.screenshot(region=(0, 0, 1280, 720))
    frame = np.array(img)
    frame = cv2.resize(frame, (224, 224))
    return frame
```

### 1.2 - 👁️ Traitement OpenCV

**📌 Objectif :** détecter lignes blanches, feux rouges, panneaux

**🔧 Algorithmes :**
- Filtrage de couleur (HSV) pour lignes et feux
- **Canny** + **HoughLinesP** pour détection de lignes
- Contours / Masques pour feux de circulation

**📁 Fichier :** `perception/lane_detection.py`

### 1.3 - 📦 Intégration YOLOv8

**📌 Objectif :** détecter voitures, piétons, feux rouges, stops

**🎯 Implémentation :**
- Utiliser le modèle `yolov8n.pt` (rapide pour test)
- Retourner une liste de classes détectées + positions

**📁 Fichier :** `perception/detector.py`

---

## 🔹 Phase 2 – Fonction de récompense

### 2.1 - ⚖️ Définir la logique de récompense

**🎯 Système de points :**

| Action | Récompense |
|--------|------------|
| Franchissement de ligne | **-5** |
| Collision détectée (via YOLO) | **-10** |
| Griller un feu rouge / stop | **-20** |
| Avancer sans erreur | **+0.01** |
| Dépassement réussi | **+5** |
| Atteindre un checkpoint | **+15** |

**📁 Fichier :** `utils/reward_function.py`

---

## 🔹 Phase 3 – Modèle CNN + LSTM

### 3.1 - 🧠 Feature extractor CNN

**🏗️ Architecture :**
- **ResNet18** sans la dernière couche FC
- Sortie : `(batch, seq_len, 512)`

### 3.2 - ⏱️ LSTM pour mémoire temporelle

**🔄 Configuration :**
- LSTM input : `(seq_len, batch, 512)`
- Output : dernier état → action

**📁 Fichier :** `rl/model.py`

---

## 🔹 Phase 4 – Replay Buffer Séquentiel

### 4.1 - 🧩 Buffer temporel

**💾 Structure :**
- Stocker **N frames** par transition
- Chaque transition = `(frames[], action, reward, done)`

**📁 Fichier :** `rl/replay_buffer.py`

---

## 🔹 Phase 5 – Agent SAC

### 5.1 - 🤖 Agent de Reinforcement Learning

**🏛️ Composants :**
- **Policy** = CNN + LSTM → MLP
- **2 Critiques** Q1 et Q2
- **Alpha** auto-ajusté
- **Target update** (polyak average)

**📁 Fichiers :** `rl/sac.py` et `rl/agent.py`

> 💡 **Option :** partir d'un fork de **Stable-Baselines3** ou implémenter SAC from scratch

---

## 🔹 Phase 6 – Boucle d'entraînement

### 6.1 - 🔄 Training loop principal

```python
while not done:
    # Capture et traitement
    frame = get_frame()
    infos = process_frame_with_yolo_opencv(frame)
    reward = compute_reward(infos)
    
    # Stockage dans buffer
    buffer.add(sequence, action, reward, done)
    
    # Entraînement périodique
    if timestep % update_freq == 0:
        agent.train(buffer.sample())
    
    # Prédiction et action
    act = agent.predict_action(sequence)
    send_action_to_gta(act)
```

**📁 Fichier :** `train.py`

---

## 🔹 Phase 7 – Sauvegarde & Test

### 7.1 - 💾 Sauvegarde du modèle

**🔄 Fréquence :** enregistrer tous les X steps en `.pt` (PyTorch)

```python
torch.save(model.state_dict(), "models/policy.pt")
```

### 7.2 - 🧪 Script de test

**🎮 Fonctionnalités :**
- Charger le modèle entraîné
- Tourner GTA en mode "demo"
- Évaluer avec rendu activé

**📁 Fichier :** `test.py`

---

## 🔹 Phase 8 – Déploiement RC (futur)

### 8.1 - 📦 Export du modèle

**🔧 Formats de conversion :**
- `torch.jit.trace()` ou **ONNX** pour format embarqué

### 8.2 - 🚗 Code embarqué

**🔄 Pipeline :**
1. Lire image caméra de la voiture
2. Traitement OpenCV + CNN+LSTM
3. Envoi commandes vers moteur / direction

**🖥️ Plateformes possibles :**
- **Raspberry Pi 5**
- **Jetson Nano**
- **ESP32** + co-processeur

---

## 📊 Suivi des performances

### 📈 Intégration Tensorboard

**📋 Métriques à suivre :**
- **Reward total** par épisode
- **Loss policy / critic**
- **Score moyen** sur N épisodes
- **Temps de convergence**

---

## 🗂️ Structure finale du projet

```
AutoPilot-GTA5/
├── capture/
│   └── screen_capture.py
├── perception/
│   ├── lane_detection.py
│   └── detector.py
├── rl/
│   ├── model.py
│   ├── replay_buffer.py
│   ├── sac.py
│   └── agent.py
├── utils/
│   └── reward_function.py
├── models/                 # Modèles sauvegardés
├── logs/                   # Logs Tensorboard
├── train.py
├── test.py
├── config.yaml
└── requirements.txt
```

---

## 🚀 Ordre d'implémentation recommandé

1. **Phase 1** : Capture + OpenCV + YOLO
2. **Phase 2** : Fonction de récompense
3. **Phase 3** : Modèle CNN+LSTM
4. **Phase 4** : Replay Buffer
5. **Phase 5** : Agent SAC
6. **Phase 6** : Boucle d'entraînement
7. **Phase 7** : Test et sauvegarde
8. **Phase 8** : Déploiement (optionnel)

---

**🎯 Objectif final :** Un agent capable de conduire de manière autonome dans GTA V en respectant le code de la route, avec possibilité de transfert vers une voiture RC réelle.