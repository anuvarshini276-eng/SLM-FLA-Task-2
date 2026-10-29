"""Run: python app.py; visit http://127.0.0.1:5000."""
import json
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from dfa import run_dfa, validate
from model import RNN

ROOT = Path(__file__).resolve().parent


def create_app(model_path=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 4096
    path = Path(model_path) if model_path else ROOT / 'artifacts' / 'model.npz'
    model = RNN.load(path) if path.exists() else None

    @app.get('/')
    def home():
        return render_template('index.html')

    @app.get('/api/metrics')
    def metrics():
        path = ROOT / 'artifacts' / 'metrics.json'
        return jsonify(json.loads(path.read_text(encoding='utf-8')) if path.exists() else {})

    @app.post('/api/predict')
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or 'string' not in payload:
            return jsonify(error='Send a JSON object with a string field.'), 400
        try:
            text = payload['string']
            validate(text)
        except ValueError as error:
            return jsonify(error=str(error)), 400
        exact = run_dfa(text)
        neural = None
        if model is not None:
            probability = float(model.forward([text])[0])
            neural = {'accepted': probability >= .5, 'accept_probability': probability,
                      'agrees': (probability >= .5) == exact['accepted']}
        return jsonify(string=text, exact=exact, neural=neural,
                       outside_training_lengths=len(text) > 24,
                       model_message=None if model else 'Model missing: run python train.py and restart Flask.')

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error='Request too large.'), 413

    return app


if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=5000, debug=False)
