from world_action_model.transformers.wa_transforms_lerobot import WATransformsLerobot

from robonana.data.fact_transforms import RoboNanaTransforms


def test_transform_inherits_fact_transform():
    assert issubclass(RoboNanaTransforms, WATransformsLerobot)
