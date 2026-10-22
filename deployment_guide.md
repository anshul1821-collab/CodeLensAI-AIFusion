# CodeLens AI — Deployment & Production Checklist

This guide outlines the steps required to deploy **CodeLens AI** to production on any cloud platform (Render, Railway, Fly.io, Heroku, AWS, or DigitalOcean).

---

## 📋 Production Readiness Checklist

- [x] **Django Security Checks Passed**: `python manage.py check --deploy`
- [x] **Automated Test Suite Verified**: 16/16 tests passing (`python manage.py test assistant`)
- [x] **Zero Hardcoded Secrets**: Secrets and keys loaded from `.env` via `python-dotenv`
- [x] **Safe Code Execution**: Static analysis only — no `eval()` or code execution
- [x] **CSRF Protection Active**: All state-modifying endpoints validate CSRF tokens
- [x] **File Upload Restrictions**: 10MB file limit, extension whitelist (`.pdf`, `.txt`)
- [x] **Cross-Platform Compatibility**: Pure Python implementation runs on Windows, Linux, and macOS

---

## 🛠️ Step-by-Step Production Deployment

### 1. Collect Static Files
Before serving via a production web server (Nginx, Gunicorn, or WhiteNoise), run:
```bash
python manage.py collectstatic --noinput
```

### 2. Environment Variables Configuration
Set the following environment variables on your hosting provider dashboard:
```env
DEBUG=False
SECRET_KEY=generate-a-strong-random-50-character-string-for-production
ALLOWED_HOSTS=your-app-domain.com,www.your-app-domain.com
GEMINI_API_KEY=AIzaSy... (optional for cloud LLM and embeddings)
EMBEDDER_PREFERENCE=auto
```

### 3. Production WSGI Server
For Linux / cloud deployments, use **Gunicorn**:
```bash
pip install gunicorn
gunicorn codelens_project.wsgi:application --bind 0.0.0.0:$PORT --workers 3
```

For Windows production servers, use **Waitress**:
```bash
pip install waitress
waitress-serve --port=8000 codelens_project.wsgi:application
```

---

## 🐳 Optional Dockerfile (Containerized Deployment)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

RUN python manage.py migrate --noinput
RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["gunicorn", "codelens_project.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3"]
```

---

## ⚠️ Troubleshooting: `OperationalError: unable to open database file`

If you encounter this error after deploying, it is caused by one of two things:
1. **Migrations were never run on the server**: `db.sqlite3` is in `.gitignore` (for security), so a fresh clone has no database file until `python manage.py migrate` runs.
2. **Directory Write Permissions**: SQLite requires write permission on **both** the file `db.sqlite3` **and its parent directory** (to write `.db.sqlite3-wal` lock and journal files).

### Solution by Platform:

#### 1. Render (Web Service)
- In the Render Dashboard -> Your Service -> **Settings**:
  - **Build Command**: `./build.sh`
    *(or: `pip install -r requirements.txt && python manage.py migrate --noinput && python manage.py seed_samples && python manage.py collectstatic --noinput`)*
  - **Start Command**: `gunicorn codelens_project.wsgi:application`
- Click **Manual Deploy -> Deploy latest commit**.

#### 2. PythonAnywhere
1. Open a **Bash Console** from your PythonAnywhere Dashboard.
2. Navigate to your project directory and activate your virtualenv:
   ```bash
   cd ~/CodeLensAI-AIFusion
   workon myvirtualenv   # or: source ~/.virtualenvs/.../bin/activate
   python manage.py migrate
   python manage.py seed_samples
   chmod 775 ~/CodeLensAI-AIFusion
   ```
3. Go to the **Web** tab and click the green **Reload** button.

#### 3. Linux VPS / EC2 / DigitalOcean (Gunicorn / Nginx / Apache)
Ensure the web server process user (`www-data` or `nginx`) owns the project directory so SQLite can create journal files:
```bash
cd /path/to/project
python manage.py migrate
sudo chown -R www-data:www-data /path/to/project
sudo chmod 775 /path/to/project
sudo chmod 664 /path/to/project/db.sqlite3
sudo systemctl restart gunicorn
```

#### 4. Railway / Fly.io
In Railway -> Project Settings -> **Deploy**:
- Ensure Build/Pre-deploy command runs:
  ```bash
  python manage.py migrate --noinput && python manage.py seed_samples
  ```

#### 5. Vercel / AWS Lambda (Serverless)
Serverless filesystems are read-only except `/tmp`.
The project's `settings.py` now automatically detects `VERCEL` and redirects SQLite to `/tmp/db.sqlite3`.

