# Installation

Prérequis : Python 3.8+ installé.

1) Créer et activer un environnement virtuel (Windows PowerShell) :

```powershell
python -m venv venv
# PowerShell
.\venv\Scripts\Activate.ps1
# cmd.exe
.\venv\Scripts\activate.bat
# Git Bash / WSL
source venv/Scripts/activate
```

2) Mettre à jour pip :

```bash
python -m pip install --upgrade pip
```

3) Installer les dépendances depuis `requirements.txt` :

```bash
pip install -r requirements.txt
```

4) Lancer le notebook :

```bash
# démarrer Jupyter Notebook
jupyter notebook deeplearning-machine.ipynb
# ou démarrer JupyterLab
jupyter lab
```

Conseils :
- Si vous utilisez VS Code, sélectionnez l'interpréteur Python du venv via la palette de commandes (Python: Select Interpreter) avant d'ouvrir le notebook.
- Les fichiers HDF5 attendus sont dans le dossier `datasets/` : `trainset.hdf5` et `testset.hdf5`.

Si vous rencontrez des erreurs lors de l'installation, copiez l'erreur et exécutez :

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
