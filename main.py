from dotenv import load_dotenv

from churn_project.orchestrator.training_flow import training_flow

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    training_flow(trigger_reason="main.py trigger")
