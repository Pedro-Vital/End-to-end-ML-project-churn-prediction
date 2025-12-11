FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

RUN pip install poetry
RUN poetry self add poetry-plugin-export
RUN poetry export -f requirements.txt --without-hashes --only streamlit -o requirements.txt
RUN pip install -r requirements.txt
RUN pip uninstall -y poetry

COPY frontend /app/frontend

EXPOSE 8501

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
