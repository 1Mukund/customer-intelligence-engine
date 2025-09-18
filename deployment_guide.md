# 🚀 Customer Intelligence Engine - Deployment Guide

## 🌟 **Quick Deployment Options**

### 1. **Streamlit Cloud (Recommended for beginners)**
- ✅ **Free hosting**
- ✅ **Automatic deployments**
- ✅ **Easy setup**

**Steps:**
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect GitHub repo
4. Deploy!

### 2. **Docker Deployment (Recommended for production)**
```bash
# Quick start
./deploy.sh

# Or manually
docker-compose up -d
```

### 3. **Cloud Platforms**

#### **AWS EC2**
```bash
# Launch EC2 instance
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start

# Clone and deploy
git clone <your-repo>
cd ci_engine
./deploy.sh
```

#### **Google Cloud Run**
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/ci-engine
gcloud run deploy --image gcr.io/PROJECT-ID/ci-engine --platform managed
```

#### **Azure Container Instances**
```bash
# Create resource group
az group create --name ci-engine-rg --location eastus

# Deploy container
az container create \
  --resource-group ci-engine-rg \
  --name ci-engine \
  --image <your-image> \
  --ports 8501
```

#### **DigitalOcean App Platform**
- Connect GitHub repo
- Select Dockerfile
- Deploy automatically

### 4. **VPS Deployment (Cheap & Reliable)**

#### **Setup on Ubuntu VPS:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone and deploy
git clone <your-repo>
cd ci_engine
./deploy.sh
```

## 🔧 **Configuration**

### **Environment Variables (.env)**
```env
# Add your configuration
OLLAMA_HOST=http://localhost:11434
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### **Nginx Reverse Proxy (Optional)**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 **Monitoring & Scaling**

### **Health Checks**
- Built-in health check at `/_stcore/health`
- Docker health checks included
- Automatic restart on failure

### **Scaling Options**
- **Horizontal**: Multiple container instances
- **Vertical**: Increase CPU/RAM
- **Load Balancer**: Distribute traffic

## 🔒 **Security Considerations**

1. **Environment Variables**: Store secrets in `.env`
2. **HTTPS**: Use SSL certificates
3. **Authentication**: Add login if needed
4. **Firewall**: Restrict access to necessary ports
5. **Updates**: Keep dependencies updated

## 💰 **Cost Estimates**

| Platform | Cost/Month | Pros |
|----------|------------|------|
| Streamlit Cloud | Free | Easy, automatic |
| DigitalOcean Droplet | $5-20 | Cheap, reliable |
| AWS EC2 t3.micro | $8-15 | Scalable |
| Google Cloud Run | $0-10 | Pay per use |
| Azure Container | $10-30 | Enterprise ready |

## 🚀 **Recommended Setup for Production**

1. **VPS (DigitalOcean/Linode)** - $10/month
2. **Docker deployment** with auto-restart
3. **Nginx reverse proxy** with SSL
4. **Automated backups**
5. **Monitoring** (optional)

## 📞 **Support**

For deployment help:
- Check logs: `docker-compose logs`
- Restart: `docker-compose restart`
- Update: `git pull && docker-compose up -d --build`