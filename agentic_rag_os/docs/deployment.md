# Agentic RAG OS — Deployment Guide (Google Cloud)

## Prerequisites

- Google Cloud SDK installed: `gcloud --version`
- Docker installed: `docker --version`
- Project ID: `agentic-rag-os`
- Project name: `rag-os`

---

## Step 1: Set up Google Cloud project

```bash
# Set project
gcloud config set project agentic-rag-os

# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com
```

---

## Step 2: Create GitHub OAuth App

1. Go to [github.com/settings/applications/new](https://github.com/settings/applications/new)
2. Set:
   - **Application name**: Agentic RAG OS
   - **Homepage URL**: `https://ragos-<hash>-uc.a.run.app` (get URL after first deploy, then update)
   - **Authorization callback URL**: `https://ragos-<hash>-uc.a.run.app/api/v1/auth/github/callback`
3. Note your **Client ID** and **Client Secret**

---

## Step 3: Store secrets in Secret Manager

```bash
# Secret key (generate a strong one)
echo -n "$(python3 -c 'import secrets; print(secrets.token_hex(32))')" | \
  gcloud secrets create ragas-secret-key --data-file=-

# GitHub OAuth
echo -n "YOUR_GITHUB_CLIENT_ID" | \
  gcloud secrets create ragas-github-client-id --data-file=-

echo -n "YOUR_GITHUB_CLIENT_SECRET" | \
  gcloud secrets create ragas-github-client-secret --data-file=-
```

---

## Step 4: Build & deploy with Cloud Build

From the repository root:

```bash
cd /path/to/agentic-rag-gym

gcloud builds submit \
  --config=agentic_rag_os/cloudbuild.yaml \
  --project=agentic-rag-os \
  .
```

This will:
1. Build the Docker image
2. Push to `gcr.io/agentic-rag-os/ragos`
3. Deploy to Cloud Run as `ragos` in `us-central1`

---

## Step 5: Get the service URL

```bash
gcloud run services describe ragos \
  --region=us-central1 \
  --format='value(status.url)'
```

---

## Step 6: Update GitHub OAuth callback URL

After getting the Cloud Run URL, go back to your GitHub OAuth App and update:
- **Homepage URL**: `https://<your-cloud-run-url>`
- **Callback URL**: `https://<your-cloud-run-url>/api/v1/auth/github/callback`

Then update the Cloud Run environment variables:

```bash
gcloud run services update ragos \
  --region=us-central1 \
  --update-env-vars="RAGAS_GITHUB_REDIRECT_URI=https://<your-url>/api/v1/auth/github/callback,RAGAS_FRONTEND_URL=https://<your-url>"
```

---

## Step 7: Verify deployment

```bash
# Health check
curl https://<your-url>/api/health

# API docs
open https://<your-url>/api/docs

# Landing page
open https://<your-url>
```

---

## Step 8: Set up Cloud Build trigger (CI/CD)

```bash
gcloud builds triggers create github \
  --name="ragos-deploy" \
  --repo-name="agentic-rag-gym" \
  --repo-owner="williyam-m" \
  --branch-pattern="^main$" \
  --build-config="agentic_rag_os/cloudbuild.yaml" \
  --project=agentic-rag-os
```

---

## Local Development

```bash
# Install dependencies
pip install -r agentic_rag_os/requirements.txt

# Copy and configure env
cp agentic_rag_os/.env.example .env
# Edit .env with your GitHub OAuth credentials

# Run the server
python -m agentic_rag_os.run
# Server starts at http://localhost:8080

# Or with Docker
docker compose -f agentic_rag_os/docker-compose.yml up --build
```

---

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `RAGAS_SECRET_KEY` | JWT signing secret (min 32 chars) | Yes |
| `RAGAS_GITHUB_CLIENT_ID` | GitHub OAuth App client ID | Yes (for login) |
| `RAGAS_GITHUB_CLIENT_SECRET` | GitHub OAuth App client secret | Yes (for login) |
| `RAGAS_GITHUB_REDIRECT_URI` | OAuth callback URL | Yes |
| `RAGAS_FRONTEND_URL` | Frontend base URL (for redirects) | Yes |
| `RAGAS_DATABASE_URL` | SQLAlchemy async DB URL | No (default: SQLite) |
| `RAGAS_PORT` | Server port | No (default: 8080) |
| `RAGAS_DEBUG` | Enable debug mode | No (default: false) |
| `RAGAS_FREE_TIER_MAX_MB` | Free tier storage limit | No (default: 5) |
| `RAGAS_PREMIUM_TIER_MAX_MB` | Premium tier storage limit | No (default: 500) |

---

## Production Checklist

- [ ] `RAGAS_SECRET_KEY` set to cryptographically random value
- [ ] GitHub OAuth App callback URL updated to production URL
- [ ] `RAGAS_DEBUG=false`
- [ ] Database backed by PostgreSQL (not SQLite) for multi-instance
- [ ] Cloud Run min-instances ≥ 1 to avoid cold starts
- [ ] Secret Manager used for all secrets (not env vars directly)
- [ ] Cloud CDN or Cloud Armor configured for public traffic
