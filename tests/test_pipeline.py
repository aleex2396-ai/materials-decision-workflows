import json
from crfml.discovery import discover
from crfml.control import rollout
import numpy as np


def test_discovery_feedback_to_controller_serializes_and_reproduces():
    one = discover(37, steps=2, pool_size=30)
    two = discover(37, steps=2, pool_size=30)
    assert one == two
    assert len(set(one['selected'])) == len(one['selected'])
    assert one['recommendation'] in one['selected']
    result = rollout(one['true_gain'],one['predicted_gain'],np.zeros(5),'safe_probe')
    assert result['violations'] == 0
    assert json.loads(json.dumps({'discovery':one, 'control':result}))['control']['policy'] == 'safe_probe'
