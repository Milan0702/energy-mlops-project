# GitHub + Jenkins Setup Guide

## Step 1: Create GitHub Repository

1. Go to https://github.com → Click **"+"** → **New repository**
2. **Repository name**: `energy-mlops-project`
3. **Description**: `Smart Energy Consumption Prediction — MLOps Demo`
4. **Visibility**: Public
5. **DO NOT** check "Initialize this repository" (we already have files)
6. Click **Create repository**
7. Copy the HTTPS URL shown (e.g., `https://github.com/YOURUSERNAME/energy-mlops-project.git`)

---

## Step 2: Push Code to GitHub

In your terminal, run these commands (replace YOUR_USERNAME and REPO_URL):

```powershell
# Add remote origin
git remote add origin https://github.com/YOUR_USERNAME/energy-mlops-project.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

When prompted for username → enter your GitHub username
When prompted for password → enter your **Personal Access Token** (not your password)

### Create Personal Access Token (if you don't have one):
1. Go to https://github.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click **Generate new token (classic)**
3. **Note**: `jenkins-token`
4. **Expiration**: 90 days
5. **Scopes**: Check `repo` (full repository access)
6. Click **Generate token** → Copy it immediately

---

## Step 3: Install Jenkins (Windows)

1. Download Jenkins LTS installer: https://www.jenkins.io/download/
2. Run the installer, accept defaults
3. Jenkins installs as a Windows Service at http://localhost:8080
4. Find the initial admin password:
   ```
   C:\ProgramData\Jenkins\.jenkins\secrets\initialAdminPassword
   ```
5. Open http://localhost:8080 → Paste the password → Click Continue
6. Choose **"Install suggested plugins"** → Wait for installation
7. Create admin user → Click **Save and Finish**

### Enable Docker Access for Jenkins

1. Open **Docker Desktop** → Settings → General
2. Check **"Expose daemon on tcp://localhost:2375 without TLS"**
3. Click Apply & Restart

---

## Step 4: Create Jenkins Pipeline Job

1. Open http://localhost:8080
2. Click **New Item**
3. **Name**: `energy-mlops`
4. Select **Pipeline** → Click **OK**
5. Scroll to **Pipeline** section
6. **Definition**: **Pipeline script from SCM**
7. **SCM**: **Git**
8. **Repository URL**: `https://github.com/YOUR_USERNAME/energy-mlops-project.git`
9. **Credentials**: Click **Add** → **Jenkins**
   - Kind: Username with password
   - Username: `YOUR_GITHUB_USERNAME`
   - Password: `YOUR_PERSONAL_ACCESS_TOKEN`
   - Click **Add**
10. Select the credential from dropdown
11. **Branch**: `*/main`
12. **Script Path**: `Jenkinsfile`
13. Click **Save**

---

## Step 5: Run the Pipeline

1. Open your `energy-mlops` job in Jenkins
2. Click **Build Now**
3. Click the build number → Click **Console Output**
4. Watch the 4 stages:
   - **Checkout** - Code pulled from GitHub
   - **Build Docker Images** - Images built
   - **Run Training** - Model trained, saved to models/
   - **Deploy All Services** - All containers started

Green checkmarks = Success!

---

## Step 6: Verify Everything Works

After pipeline completes:

| Service | URL |
|---------|-----|
| Dashboard UI | http://localhost |
| API | http://localhost:8000 |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |

---

## Useful Jenkins Commands

```powershell
# View build logs
docker compose logs -f

# Restart pipeline manually
# Open http://localhost:8080/job/energy-mlops/build

# Stop all containers
docker compose down
```

---

## Troubleshooting

### Jenkins can't pull from GitHub
- Verify the Personal Access Token has `repo` scope
- Check credentials are correctly added in Jenkins job config

### Docker not found in Jenkins
- Ensure Docker Desktop is running
- Verify "Expose daemon" is enabled in Docker Desktop settings

### Pipeline fails at "Deploy"
- Check that no other service is using port 80, 5432, 5000, 8000, 9090, 3000
