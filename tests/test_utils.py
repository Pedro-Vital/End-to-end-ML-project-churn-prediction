from pathlib import Path
from churn_project.utils import read_yaml
from box import ConfigBox

def test_read_yaml_returns_configbox():
    path = Path("config/schema.yaml")
    yaml_data = read_yaml(path)

    # top-level should be ConfigBox
    assert isinstance(yaml_data, ConfigBox)

    # safe_load produces dicts
    assert isinstance(yaml_data.columns, dict) 
    # It's necessary, so we can use .items() in validate_data_types method

    # check one example key
    assert "Attrition_Flag" in yaml_data.columns