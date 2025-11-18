# GitHub Pages Setup Instructions

Your evolution algorithm is ready to be hosted online! Follow these simple steps:

## 🚀 Enable GitHub Pages

1. Go to your repository on GitHub:
   ```
   https://github.com/pelnystak/evolution
   ```

2. Click on **Settings** (gear icon in the top menu)

3. Scroll down to find **Pages** in the left sidebar (under "Code and automation")

4. Under **Source**, select:
   - **Branch**: `main`
   - **Folder**: `/ (root)`

5. Click **Save**

6. Wait 1-2 minutes for GitHub to deploy

7. Your site will be live at:
   ```
   https://pelnystak.github.io/evolution/
   ```

8. Access the evolution algorithm directly at:
   ```
   https://pelnystak.github.io/evolution/index.html
   ```

## ✅ Verification

Once enabled, you should see a message like:
> ✅ Your site is live at https://pelnystak.github.io/evolution/

## 🎯 What's Hosted

The pure frontend version (`index.html`) will be accessible to anyone with just a web browser - no installation needed!

## 🔄 Auto-Updates

Every time you push changes to the `main` branch, GitHub Pages will automatically rebuild and update your site within a few minutes.

## 📝 Optional: Custom Domain

If you want to use a custom domain (like evolution.yourdomain.com):
1. Go to Pages settings
2. Enter your custom domain
3. Configure DNS records as instructed by GitHub

## 🐛 Troubleshooting

**Site not loading?**
- Wait 2-3 minutes after enabling
- Check the Pages settings for any error messages
- Verify the main branch contains index.html
- Try accessing with https:// (not http://)

**Changes not appearing?**
- GitHub Pages caching can take 1-5 minutes to update
- Try hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
- Check the Actions tab to see if deployment is in progress
