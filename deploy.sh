#!/bin/bash

echo "🚀 Deploying Customer Intelligence Engine..."

# Build and run with Docker Compose
docker-compose down
docker-compose build --no-cache
docker-compose up -d

echo "✅ Deployment complete!"
echo "🌐 Access your app at: http://localhost:8501"
echo "📊 Ollama API at: http://localhost:11434"

# Show logs
echo "📝 Showing logs..."
docker-compose logs -f ci-engine