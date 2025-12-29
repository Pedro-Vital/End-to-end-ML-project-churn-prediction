### Grafana Dashboard

1. We need to log into Grafana with user: admin and password: admin and then create a new password
2. In "Connections", we add **Prometheus** as our data source
3. We go to **Dashboards → Import**
4. Paste the dashboard JSON file from [`monitoring/grafana/dashboard.json`](./monitoring/grafana/dashboard.json)
5. When Grafana asks for a datasource mapping, set:\
   **DS_PROMETHEUS → your Prometheus datasource**
6. Click **Import**. The dashboard loads immediately and shows our informative metrics.
