def test_model_version_loaded(predictor):
    """
    Ensure model version is correctly read from MLflow.
    """
    _ = predictor.model
    assert predictor.model_version is not None
    assert isinstance(predictor.model_version, (str, int))
