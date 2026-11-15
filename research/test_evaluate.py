"""Regression checks for evaluator physics, legal observations and swallowed errors."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.evaluate import HardEnv, run, totals
from starter_kit.environment import CampusBatteryEnv


def sample():
    return pd.DataFrame(dict(Fecha=pd.to_datetime([
        "2026-01-01 00:00", "2026-01-01 01:00", "2026-02-01 00:00", "2026-02-01 01:00"]),
        load_kw=[2000., 1800., 1500., 1700.], pv_kw=[0., 2500., 0., 3000.],
        precio_eur_per_kwh=[.1, .2, .3, .4], co2_kg_per_kwh=[.4, .3, .2, .1],
        irr=[0., 500., 0., 600.], tamb=[10.]*4, tpan=[15.]*4, wind=[2.]*4))


class TestEvaluator(unittest.TestCase):
    def test_physics_matches_supplied_simulator(self):
        data = sample()
        a = HardEnv(data)
        b = CampusBatteryEnv(data, spec=dict(capacity_kwh=6000, power_kw=1500,
            efficiency=.9, soc_init=.5, soc_min=0, soc_max=1, dt_hours=1))
        for action in [-1500., 1500., -4000., 1500.]:
            a.step(action); b.step(action)
        np.testing.assert_allclose(a.soc_series, b.soc_series)
        np.testing.assert_allclose(a.grid, b.grid)
        self.assertEqual(a.infraccions_per_step, b.infraccions_per_step)
        self.assertEqual(a.evaluate_results(), b.evaluate_results())

    def test_markets_stop_at_gaps_and_telemetry_matches_kit(self):
        env = HardEnv(sample())
        obs = env.reset()
        self.assertEqual(obs['preus_properes_24h'], [.1, .2])
        self.assertEqual(obs['noves_dades_5min']['ultim_consum_kw'], 2000.)
        obs, _ = env.step(0.)
        self.assertEqual(obs['preus_properes_24h'], [.2])
        obs, _ = env.step(0.)
        self.assertEqual(obs['preus_properes_24h'], [.3, .4])
        self.assertEqual(obs['noves_dades_5min']['ultim_consum_kw'], 1800.)

    def test_soc_reset_and_aggregate_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'agent.py').write_text('class BatteryAgent:\n def act(self, obs):\n  return -500.0\n')
            carry = run(sample(), directory)
            reset = run(sample(), directory, gap_soc='reset')
        self.assertEqual(carry['soc_resets'], 0)
        self.assertEqual(reset['soc_resets'], 1)
        self.assertAlmostEqual(reset['periods'][1]['soc_initial_kwh'], 3000.)
        self.assertLess(carry['periods'][1]['soc_initial_kwh'], 3000.)
        self.assertAlmostEqual(sum(b['co2_participant_kg'] for b in carry['periods']), carry['co2_participant_kg'])
        self.assertAlmostEqual(carry['score'], .7*carry['co2_saved_pct']+.3*carry['cost_saved_pct'])

    def test_hidden_init_and_act_errors_detected(self):
        examples = [
            'class BatteryAgent:\n def __init__(self):\n  try:\n   raise ValueError("bad model")\n  except Exception:\n   pass\n def act(self, obs):\n  return 0.0\n',
            'class BatteryAgent:\n def act(self, obs):\n  try:\n   raise RuntimeError("hidden prediction")\n  except Exception:\n   return 0.0\n']
        for code in examples:
            with tempfile.TemporaryDirectory() as directory:
                Path(directory, 'agent.py').write_text(code)
                with self.assertRaises(RuntimeError):
                    run(sample(), directory)
                result = run(sample(), directory, strict=False)
                self.assertGreater(result['errors'], 0)

    def test_unrounded_aggregate_not_mean(self):
        r = totals(np.array([90., 10.]), np.array([100., 100.]),
                   np.array([1., 10.]), np.array([1., 10.]))
        self.assertAlmostEqual(r['score'], 100*(1100-190)/1100)
        self.assertNotAlmostEqual(r['score'], 50.)


if __name__ == '__main__':
    unittest.main()
