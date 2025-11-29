## 🔧 Configuration & Secrets Handling

This project uses a **simple and safe configuration pattern**:

* **`config.yaml.example`** → committed to the repository
* **`config.yaml`** → ignored by Git and contains real credentials
* **Environment variables** → optional and override values in `config.yaml`

This prevents leaking database credentials on GitHub while keeping local setup straightforward.

---

## 📁 Step 1 — Create your `config.yaml`

Start by copying the example file:

```bash
cp config.yaml.example config.yaml
```

Inside `config.yaml`, you will see:

```yaml
db_host: null
db_user: null
db_password: null
db_name: null
```

You have two ways to provide your database credentials:

---

### **🔹 Option A — Fill the values directly**

```yaml
db_host: "localhost"
db_user: "root"
db_password: "12345"
db_name: "bank_db"
```

This is the simplest approach for local development.

---

### **🔹 Option B — Use environment variables (preferred for CI)**

```bash
export DB_HOST=localhost
export DB_USER=root
export DB_PASSWORD=12345
export DB_NAME=bank_db
```

These values automatically override whatever is in `config.yaml`.

---

config (config.yaml, schema.yaml, params.yaml)
entities (config and artifact)
configuration manager in src config
components
pipeline
