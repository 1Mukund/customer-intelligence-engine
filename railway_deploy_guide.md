# 🚀 Railway Deployment Guide

## 📋 **Prerequisites**
- GitHub account
- Railway account (free)
- Your code pushed to GitHub

## 🎯 **Step-by-Step Deployment**

### **Step 1: Prepare Your Repository**
```bash
# Make sure all files are committed
git add .
git commit -m "Ready for Railway deployment"
git push origin main
```

### **Step 2: Deploy on Railway**

1. **Go to Railway**
   - Visit [railway.app](https://railway.app)
   - Click "Start a New Project"
   - Sign in with GitHub

2. **Deploy from GitHub**
   - Click "Deploy from GitHub repo"
   - Select your `ci_engine` repository
   - Railway will automatically detect the Dockerfile

3. **Configure Environment Variables** (Optional)
   - Go to your project dashboard
   - Click "Variables" tab
   - Add any environment variables you need

4. **Deploy!**
   - Railway will automatically build and deploy
   - You'll get a public URL like: `https://your-app.railway.app`

### **Step 3: Set Up Ollama (Important!)**

Since Railway doesn't support Ollama directly, you have two options:

#### **Option A: Use External Ollama Service**
```bash
# Deploy Ollama on a separate service (like Render or fly.io)
# Then set OLLAMA_HOST environment variable in Railway
```

#### **Option B: Use OpenAI API (Recommended for Railway)**
Let me create an OpenAI fallback for you:

```python
# Add this to your environment variables in Railway:
OPENAI_API_KEY=your_openai_api_key
USE_OPENAI=true
```

### **Step 4: Monitor Deployment**
- Check the "Deployments" tab for build logs
- Monitor the "Metrics" tab for performance
- Use the "Logs" tab for debugging

## 🔧 **Railway Configuration Files**

Your repository includes:
- ✅ `railway.json` - Railway configuration
- ✅ `Dockerfile` - Container configuration  
- ✅ `requirements.txt` - Python dependencies
- ✅ `.env.example` - Environment variables template

## 💰 **Railway Pricing**
- **Free Tier**: $5 credit monthly
- **Usage**: ~$0.10-0.50/day for small apps
- **Scaling**: Automatic based on traffic

## 🚨 **Important Notes**

1. **Ollama Limitation**: Railway doesn't support Ollama directly
2. **Memory**: Free tier has 512MB RAM limit
3. **Sleep**: Apps sleep after 30 minutes of inactivity
4. **Custom Domain**: Available on paid plans

## 🔄 **Alternative: Hybrid Approach**

Deploy your app on Railway + Ollama on another free service:

1. **Main App**: Railway (Streamlit interface)
2. **Ollama**: Render/Fly.io (AI processing)
3. **Connection**: API calls between services

## 📞 **Troubleshooting**

### **Common Issues:**
- **Build fails**: Check Dockerfile and requirements.txt
- **App crashes**: Check logs in Railway dashboard
- **Ollama errors**: Set up external Ollama service

### **Quick Fixes:**
```bash
# Redeploy
git commit --allow-empty -m "Trigger redeploy"
git push origin main

# Check logs
# Go to Railway dashboard > Logs tab
```

## ✅ **Success Checklist**
- [ ] Code pushed to GitHub
- [ ] Railway project created
- [ ] App deployed successfully
- [ ] Public URL accessible
- [ ] Ollama/AI service configured
- [ ] All features working

## 🎉 **You're Live!**
Once deployed, share your URL with the world! 🌍