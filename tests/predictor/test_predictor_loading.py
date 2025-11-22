def test_model_lazy_loading(predictor):
    """
    Ensure the model loads lazily and only when accessed.
    """
    # not loaded yet
    assert predictor._model is None

    _ = predictor.model

    # now it must be loaded
    assert predictor._model is not None


def test_model_version_loaded(predictor):
    """
    Ensure model version is correctly read from MLflow.
    """
    _ = predictor.model
    assert predictor._model_version is not None
    assert isinstance(predictor._model_version, (str, int))
