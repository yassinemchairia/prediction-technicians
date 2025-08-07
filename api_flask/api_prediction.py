from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import requests
import threading
import time
from datetime import datetime
import logging
import traceback

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_flask.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MODEL_PATH = "./technicien_model.pkl"
FEATURES_PATH = "./model_features.txt"
SPRING_API_URL = "http://springboot:8087/prediction/donnees"
UPDATE_INTERVAL = 300  # 5 minutes

# Chargement initial
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        expected_features = [line.strip() for line in f.readlines()]
    logger.info("Modèle et features chargés avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement initial: {str(e)}")
    raise

def get_period_of_day(hour):
    """Détermine la période de la journée"""
    if 5 <= hour < 12:
        return 'MATIN'
    elif 12 <= hour < 18:
        return 'APRES_MIDI'
    elif 18 <= hour < 22:
        return 'SOIREE'
    else:
        return 'NUIT'

def process_new_data(df):
    """Transforme les nouvelles données dans le format d'entraînement"""
    try:
        # Vérification des colonnes requises
        required_columns = {'dateDebut', 'specialite', 'typeIntervention', 
                          'priorite', 'dureeEnHeures', 'technicienId'}
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            logger.warning(f"Colonnes manquantes: {missing_cols}")
            return pd.DataFrame()

        # Conversion de la date
        if pd.api.types.is_list_like(df['dateDebut'].iloc[0]):
            df['dateDebut'] = df['dateDebut'].apply(
                lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else pd.NaT
            )

        df['dateDebut'] = pd.to_datetime(
            df['dateDebut'],
            format='%Y-%m-%dT%H:%M:%S',
            errors='coerce'
        )

        # Filtrage des données invalides
        invalid_dates = df['dateDebut'].isnull()
        if invalid_dates.any():
            logger.warning(f"{invalid_dates.sum()} dates invalides trouvées")
            df = df[~invalid_dates]
            if df.empty:
                return pd.DataFrame()

        # Traitement des features
        df['heureDeLaJournee'] = df['dateDebut'].dt.hour
        df['dureeEnHeures_norm'] = df['dureeEnHeures'] / max(df['dureeEnHeures'].max(), 1.0)
        df['periodeJournee'] = df['heureDeLaJournee'].apply(get_period_of_day)
        
        # Nettoyage des strings
        for col in ['specialite', 'typeIntervention', 'priorite']:
            df[col] = df[col].astype(str).str.strip()
        
        # Features composites
        df['specialite_priorite'] = df['specialite'] + "_" + df['priorite']
        df['type_periode'] = df['typeIntervention'] + "_" + df['periodeJournee']

        # Encodage one-hot
        X = pd.get_dummies(df.drop('technicienId', axis=1))
        
        # Ajouter les colonnes manquantes et réorganiser
        for feat in expected_features:
            if feat not in X.columns:
                X[feat] = 0
        X = X[expected_features]
        
        return X, df['technicienId'].astype(int)

    except Exception as e:
        logger.error(f"Erreur dans process_new_data: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(), None

def update_model():
    """Mise à jour périodique du modèle"""
    while True:
        try:
            logger.info("Début de la mise à jour du modèle...")
            
            # Récupération des données
            response = requests.get(
                SPRING_API_URL,
                headers={'Accept': 'application/json'},
                timeout=15
            )
            
            if response.status_code != 200:
                logger.error(f"Erreur HTTP {response.status_code}: {response.text[:200]}")
                time.sleep(UPDATE_INTERVAL)
                continue

            new_data = response.json()
            if not new_data:
                logger.info("Aucune nouvelle donnée disponible")
                time.sleep(UPDATE_INTERVAL)
                continue

            logger.info(f"Reçu {len(new_data)} nouvelles interventions")
            
            # Conversion et traitement
            df = pd.DataFrame(new_data)
            X, y = process_new_data(df)
            
            if X.empty:
                logger.warning("Aucune donnée valide après traitement")
                time.sleep(UPDATE_INTERVAL)
                continue

            # Réentraînement
            model.fit(X, y)
            joblib.dump(model, MODEL_PATH)
            logger.info(f"Modèle mis à jour avec {len(df)} nouvelles interventions")

        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de connexion: {str(e)}")
        except Exception as e:
            logger.error(f"Erreur inattendue: {str(e)}")
            logger.error(traceback.format_exc())
        
        time.sleep(UPDATE_INTERVAL)

def prepare_features(data):
    """Prépare les features pour la prédiction"""
    try:
        # Validation des entrées
        required_fields = ['dateDebut', 'specialite', 'typeIntervention', 'priorite']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Champ requis manquant: {field}")
        
        # Conversion de la date
        date_str = str(data['dateDebut']).split('.')[0]
        intervention_time = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
        
        # Calcul des features
        features = {
            'specialite': str(data['specialite']).strip(),
            'typeIntervention': str(data['typeIntervention']).strip(),
            'priorite': str(data['priorite']).strip(),
            'dureeEnHeures_norm': min(float(data.get('dureeEstimee', 4)) / 24, 1.0),
            'heureDeLaJournee_norm': intervention_time.hour / 23,
            'periodeJournee': get_period_of_day(intervention_time.hour),
            'specialite_priorite': f"{data['specialite']}_{data['priorite']}",
            'type_periode': f"{data['typeIntervention']}_{get_period_of_day(intervention_time.hour)}"
        }
        
        # Conversion en DataFrame
        df = pd.DataFrame([features])
        
        # Encodage one-hot
        features_encoded = pd.get_dummies(df)
        
        # Ajouter les colonnes manquantes et réorganiser
        for feature in expected_features:
            if feature not in features_encoded.columns:
                features_encoded[feature] = 0
        features_encoded = features_encoded[expected_features]
        
        return features_encoded
    
    except Exception as e:
        logger.error(f"Erreur dans prepare_features: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    try:
        data = request.get_json()
        logger.info(f"Requête de prédiction reçue: {data}")
        
        # Préparation des features
        features_encoded = prepare_features(data)
        
        # Prédiction
        probas = model.predict_proba(features_encoded)[0]
        top_3 = sorted(zip(model.classes_, probas), key=lambda x: -x[1])[:2]
        
        result = {
            'techniciens': [int(tech_id) for tech_id, _ in top_3],
            'probabilites': [float(prob) for _, prob in top_3],
            'status': 'success'
        }
        logger.info(f"Résultat de la prédiction: {result}")
        
        return jsonify(result)
    
    except ValueError as e:
        logger.error(f"Erreur de validation: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'message': 'Erreur interne du serveur'}), 500

if __name__ == '__main__':
    try:
        # Démarrer le thread de mise à jour
        updater = threading.Thread(target=update_model, daemon=True)
        updater.start()
        logger.info("Thread de mise à jour démarré")
        
        # Démarrer l'API Flask
        logger.info("Démarrage de l'API Flask...")
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    
    except Exception as e:
        logger.critical(f"Erreur critique au démarrage: {str(e)}")
        logger.critical(traceback.format_exc())
