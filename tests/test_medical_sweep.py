import json, math, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from aperture_medical_sweep import run as S  # noqa: E402


class TestGrid(unittest.TestCase):
    def test_grid_size(self):
        g = S.grid()
        self.assertEqual(len(g), 27)  # 3 ranks x 3 passes x 3 lrs

    def test_cell_keys_unique(self):
        g = S.grid()
        self.assertEqual(len({c['key'] for c in g}), 27)

    def test_alpha_twice_rank(self):
        for c in S.grid():
            self.assertEqual(c['alpha'], 2 * c['rank'])

    def test_cell_key_format(self):
        c = S.cell_key(4, 16, 5e-5)
        self.assertEqual(c, 'r4-p16-lr5e-05')


class TestSelection(unittest.TestCase):
    def _mk(self):
        td = Path(tempfile.mkdtemp())
        cells = []
        vals = {(4,1,5e-5):0.9,(4,4,5e-5):0.7,(4,16,2e-4):0.8}
        for r in (4,16,64):
            for p in (1,4,16):
                for lr in (5e-5,2e-4,8e-4):
                    cd = td / 'units' / f'u__r{r}-p{p}-lr{lr:g}'
                    cd.mkdir(parents=True)
                    nll = vals.get((r,p,lr), 1.0 + 0.01*r + 0.001*p)
                    (cd / 'fit.json').write_text(json.dumps(
                        {'name': f'r{r}-p{p}-lr{lr:g}',
                         'calibration_nll': nll}))
                    cells.append({'pair': ['g34', 'camelyon17-hospital_3'],
                                  'seed': 0, 'unit': 'u',
                                  'cell': f'r{r}-p{p}-lr{lr:g}',
                                  'rank': r, 'passes': p, 'lr': lr,
                                  'alpha': 2 * r, 'reused': False,
                                  'source': None, 'cell_dir': str(cd)})
        return td, {'cells': cells}

    def test_select_min_cal_nll(self):
        td, manifest = self._mk()
        rec = S.select(('g34', 'camelyon17-hospital_3'), 0, manifest, td)
        self.assertEqual(rec['selection']['4']['cell'], 'r4-p4-lr5e-05')

    def test_tie_break(self):
        td, manifest = self._mk()
        # force a tie on rank 4
        for c in manifest['cells']:
            if c['rank'] == 4:
                (Path(c['cell_dir']) / 'fit.json').write_text(json.dumps(
                    {'name': c['cell'], 'calibration_nll': 0.5}))
        rec = S.select(('g34', 'camelyon17-hospital_3'), 0, manifest, td)
        # tie-break: lower passes first -> p1
        self.assertEqual(rec['selection']['4']['passes'], 1)


if __name__ == '__main__':
    unittest.main()
