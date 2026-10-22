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
