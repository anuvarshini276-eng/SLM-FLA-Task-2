import csv
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import create_app
from dfa import run_dfa
from model import RNN


class ProjectTests(unittest.TestCase):
    def test_exact_language_exhaustively(self):
        for length in range(11):
            for bits in itertools.product('01', repeat=length):
                s = ''.join(bits)
                result = run_dfa(s)
                self.assertEqual(result['accepted'], s.endswith('0'))
                self.assertEqual(len(result['trace']), len(s)+1)
                for i, row in enumerate(result['trace'][1:], 1):
                    self.assertEqual(row['state'], 'q1' if s[i-1] == '0' else 'q0')

    def test_invalid_inputs(self):
        for s in ['2', '10 ', ' 10', '\n', '０', None, 10, '0'*257]:
            with self.assertRaises(ValueError):
                run_dfa(s)

    def test_gradient_finite_difference(self):
        model = RNN(hidden=3)
        strings, y = ['', '0', '101'], np.array([0., 1., 0.])
        _, gradients = model.gradients(strings, y)
        for name, values in model.p.items():
            for index in np.ndindex(values.shape):
                original = values[index]
                values[index] = original + 1e-5
                high, _ = model.gradients(strings, y)
                values[index] = original - 1e-5
                low, _ = model.gradients(strings, y)
                values[index] = original
                self.assertAlmostEqual((high-low)/2e-5, gradients[name][index], places=6)

    def test_export_and_padding(self):
        model = RNN.load(ROOT/'artifacts/model.npz')
        strings = ['', '0', '1', '1010', '1'*256, '0'*256]
        batched = model.forward(strings)
        np.testing.assert_allclose(batched, [model.forward([s])[0] for s in strings])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.npz'
            model.save(path)
            np.testing.assert_array_equal(batched, RNN.load(path).forward(strings))

    def test_dataset_integrity(self):
        sets = []
        for split, expected in [('train',4201),('validation',900),('test',900),('long_test',1000)]:
            with (ROOT/f'data/{split}.csv').open(newline='') as file:
                rows = list(csv.DictReader(file))
            strings = {r['string'] for r in rows}
            self.assertEqual(len(rows), expected)
            self.assertEqual(len(strings), expected)
            for row in rows:
                self.assertEqual(int(row['label']), int(row['string'].endswith('0')))
                self.assertEqual(int(row['length']), len(row['string']))
            sets.append(strings)
        for a, b in itertools.combinations(sets, 2):
            self.assertFalse(a & b)
        self.assertIn('', sets[0])

    def test_api(self):
        client = create_app().test_client()
        for path in ['/', '/static/style.css', '/static/app.js', '/api/metrics']:
            with client.get(path) as response:
                self.assertEqual(response.status_code, 200)
        for s in ['', '0', '1', '1010', '1011', '0'*256]:
            response = client.post('/api/predict', json={'string':s})
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data['exact']['accepted'], s.endswith('0'))
            self.assertTrue(0 <= data['neural']['accept_probability'] <= 1)
            self.assertEqual(data['outside_training_lengths'], len(s)>24)
        for payload in [{}, [], None, {'string':False}, {'string':'10x'}, {'string':'0'*257}]:
            self.assertEqual(client.post('/api/predict',json=payload).status_code,400)
        self.assertEqual(client.post('/api/predict',data='{broken',content_type='application/json').status_code,400)
        self.assertEqual(client.post('/api/predict',data='x'*5000,content_type='application/json').status_code,413)

    def test_missing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            client = create_app(Path(directory)/'absent.npz').test_client()
            result = client.post('/api/predict',json={'string':'0'}).get_json()
            self.assertIsNone(result['neural'])
            self.assertTrue(result['exact']['accepted'])

    def test_saved_model_quality(self):
        from train import evaluate
        with (ROOT/'data/test.csv').open(newline='') as file:
            strings = [row['string'] for row in csv.DictReader(file)]
        result = evaluate(RNN.load(ROOT/'artifacts/model.npz'), strings)
        saved = json.loads((ROOT/'artifacts/metrics.json').read_text())
        self.assertEqual(result, saved['test'])
        self.assertGreaterEqual(result['accuracy'], .95)


if __name__ == '__main__':
    unittest.main()
