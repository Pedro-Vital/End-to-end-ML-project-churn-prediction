def test_predict_returns_expected_keys(predictor, sample_input):
    """
    Test that prediction output contains the expected structure.
    """
    result = predictor.predict(sample_input)

    assert isinstance(result, dict)
    assert "predictions" in result
    assert "model_version" in result
    assert "timestamp" in result
    assert "num_samples" in result


def test_predict_number_of_samples(predictor, sample_input):
    """
    Ensure num_samples matches the input size.
    """
    result = predictor.predict(sample_input)
    assert result["num_samples"] == len(sample_input)


def test_predict_values_are_lists(predictor, sample_input):
    """
    Ensure predictions are returned as list for JSON compatibility.
    """
    result = predictor.predict(sample_input)
    assert isinstance(result["predictions"], list)
