FROM python:3.12-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock README.md /app/

# Install Poetry
RUN pip install poetry

# Export dependencies and install them
RUN poetry self add poetry-plugin-export
RUN poetry export -f requirements.txt --without-hashes --only fastapi -o requirements.txt
RUN pip install -r requirements.txt

# Remove poetry
RUN pip uninstall -y poetry

# Copy source code
COPY src /app/src
# The app hinges on modules from outside the api directory, so we copy the entire src directory

# Install the churn_project package
RUN pip install -e /app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "churn_project.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
