# Deployment Guide - HuggingFace Spaces

## Prerequisites

- HuggingFace account
- Git installed locally
- Trained model files

## Step-by-Step Deployment

### 1. Create a New Space

1. Go to https://huggingface.co/new-space
2. Choose a name for your space (e.g., `fraud-detection-api`)
3. Select **Docker** as the SDK
4. Choose visibility (Public or Private)
5. Click "Create Space"

### 2. Initialize Git Repository

```bash
cd fraud-detector
git init
git add .
git commit -m "Initial commit: Fraud Detection API"
```

### 3. Add HuggingFace Remote

```bash
# Replace YOUR_USERNAME and YOUR_SPACE with your details
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
```

### 4. Push to HuggingFace

```bash
git push -u origin main
```

**Note:** You may be prompted for credentials:
- Username: Your HuggingFace username
- Password: Use a **HuggingFace Access Token** (not your password)
  - Get token from: https://huggingface.co/settings/tokens

### 5. Monitor Build

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE`
2. Click on "Logs" tab to monitor the Docker build
3. Build typically takes 3-5 minutes

### 6. Access Your API

Once deployed, your API will be available at:
```
https://YOUR_USERNAME-YOUR_SPACE.hf.space
```

Test it:
```bash
curl https://YOUR_USERNAME-YOUR_SPACE.hf.space/health
```

## Troubleshooting

### Build Fails - Missing Models

**Problem:** Models not found in `models/` directory

**Solution:**
1. Ensure model files are committed to git
2. Check `.gitignore` doesn't exclude `.joblib` files
3. Verify models are in correct location

### Out of Memory Error

**Problem:** Docker container runs out of memory

**Solution:**
1. Reduce model size (use only necessary models)
2. Implement lazy loading
3. Request more resources from HuggingFace

### Port Issues

**Problem:** Application not accessible

**Solution:**
Ensure Dockerfile uses port 7860 (HuggingFace standard):
```dockerfile
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

## Updating Your Deployment

When you make changes:

```bash
git add .
git commit -m "Update: description of changes"
git push origin main
```

HuggingFace will automatically rebuild and redeploy.

## Advanced Configuration

### Environment Variables

Add secrets via HuggingFace Space settings:
1. Go to Space Settings → Repository secrets
2. Add key-value pairs
3. Access in `app.py`:

```python
import os
SECRET_KEY = os.getenv("SECRET_KEY")
```

### Custom Domain

For production, consider:
1. Upgrading to HuggingFace Pro
2. Setting up custom domain
3. Adding CDN/caching layer

## Monitoring

### Check Logs

```bash
# View real-time logs in HuggingFace UI
# Or use API:
curl https://huggingface.co/api/spaces/YOUR_USERNAME/YOUR_SPACE/logs
```

### Usage Analytics

HuggingFace provides basic analytics:
- Request count
- Response times
- Error rates

Access from Space settings dashboard.

## Cost Considerations

**Free Tier:**
- Limited CPU/RAM
- May sleep after inactivity
- Suitable for demos/testing

**Paid Options:**
- Persistent compute
- GPU access
- Higher resource limits
- Custom containers

## Security Checklist

Before going to production:

- [ ] Add authentication
- [ ] Implement rate limiting
- [ ] Set up CORS properly
- [ ] Use HTTPS only
- [ ] Monitor for abuse
- [ ] Set resource limits
- [ ] Add input validation
- [ ] Implement logging
- [ ] Regular security updates
- [ ] Model versioning strategy

## Support

- Documentation: https://huggingface.co/docs/hub/spaces-overview
- Community: https://discuss.huggingface.co
- Issues: https://github.com/huggingface/hub-docs/issues
