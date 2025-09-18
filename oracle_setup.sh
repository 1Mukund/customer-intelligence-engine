#!/bin/bash

echo "🚀 Setting up CI Engine on Oracle Cloud..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Git
sudo apt install -y git

# Clone your repository
echo "📥 Clone your repository:"
echo "git clone <your-github-repo-url>"
echo "cd ci_engine"

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull llama3 model
ollama pull llama3

# Setup firewall
sudo ufw allow 8501
sudo ufw allow 22
sudo ufw --force enable

echo "✅ Setup complete!"
echo "🔧 Next steps:"
echo "1. Clone your repository"
echo "2. Run: docker-compose up -d"
echo "3. Access at: http://your-vm-ip:8501"