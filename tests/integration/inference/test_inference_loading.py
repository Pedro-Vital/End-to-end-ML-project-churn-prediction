def test_model_version_is_available(predictor):
    """PredictionService should expose a valid model version."""
    version = predictor.model_version
    assert version is not None
    assert isinstance(version, (str, int))
