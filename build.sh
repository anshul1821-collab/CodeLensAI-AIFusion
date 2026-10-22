#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Run migrations to create/update database tables
python manage.py migrate --noinput

# Populate sample knowledge base documents
python manage.py seed_samples

# Collect static files for production serving
python manage.py collectstatic --noinput
