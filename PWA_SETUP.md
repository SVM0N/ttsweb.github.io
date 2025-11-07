# PWA Setup Guide for PDF Audio Player

This document explains how to set up and use the Progressive Web App (PWA) features of the PDF Audio Player.

## What is a PWA?

A Progressive Web App allows the web application to be installed on your mobile device and work like a native app, with features like:
- **Add to Home Screen**: Install the app on your device
- **Offline Support**: Use cached resources when offline
- **Standalone Mode**: Runs in full-screen without browser UI
- **Fast Loading**: Cached resources load instantly

## Files Added for PWA Support

1. **manifest.json** - PWA configuration file
2. **service-worker.js** - Offline caching and resource management
3. **icons/** - App icons in various sizes
4. **Updated index.html** - Added PWA meta tags and service worker registration

## Generating App Icons

### Option 1: Use the Icon Generator (Recommended)

1. Open `icons/generate-icons.html` in your web browser
2. Click "Download" for each icon size
3. Save each PNG file with the suggested filename
4. All icons will be saved to your Downloads folder
5. Move all generated icons to the `icons/` directory

### Option 2: Generate Icons Manually

If you have image editing software (Photoshop, GIMP, etc.):

1. Open `icons/icon.svg` in your editor
2. Export to PNG at these sizes:
   - 72x72px → icon-72x72.png
   - 96x96px → icon-96x96.png
   - 128x128px → icon-128x128.png
   - 144x144px → icon-144x144.png
   - 152x152px → icon-152x152.png
   - 192x192px → icon-192x192.png
   - 384x384px → icon-384x384.png
   - 512x512px → icon-512x512.png
3. Save all icons to the `icons/` directory

### Option 3: Use Online Tools

1. Go to https://realfavicongenerator.net/ or similar service
2. Upload `icons/icon.svg`
3. Download the generated icon package
4. Extract and place icons in the `icons/` directory

## Installing the PWA on Mobile Devices

### iOS (iPhone/iPad)

1. Open the website in **Safari** (must use Safari, not Chrome)
2. Tap the **Share** button (square with arrow pointing up)
3. Scroll down and tap **"Add to Home Screen"**
4. Edit the name if desired
5. Tap **"Add"**
6. The app icon will appear on your home screen

### Android

1. Open the website in **Chrome**
2. Tap the **three-dot menu** (⋮) in the top right
3. Tap **"Add to Home screen"** or **"Install app"**
4. Tap **"Add"** or **"Install"**
5. The app icon will appear on your home screen

### Desktop (Chrome, Edge)

1. Open the website in a supported browser
2. Look for the **install icon** (⊕) in the address bar
3. Click it and confirm the installation
4. The app will open in a standalone window

## Testing the PWA

### Check Service Worker Registration

1. Open the website
2. Open browser DevTools (F12)
3. Go to the **Console** tab
4. Look for: `✅ Service Worker registered successfully`

### Test Offline Functionality

1. Open the website and load a PDF with audio
2. Open DevTools → **Application** → **Service Workers**
3. Check **"Offline"** checkbox
4. Reload the page - cached resources should still load
5. Uncheck "Offline" when done testing

### Verify Manifest

1. Open DevTools → **Application** tab
2. Click **Manifest** in the sidebar
3. Verify all settings are correct
4. Check that icons appear properly

## Updating the PWA

When you make changes to the app:

1. Update the cache version in `service-worker.js`:
   ```javascript
   const CACHE_NAME = 'pdf-audio-player-v2'; // Increment version
   ```

2. Users will see an update prompt on next visit
3. They can reload to get the latest version

## Troubleshooting

### PWA Not Installing

- **iOS**: Must use Safari browser
- **Android**: Must use Chrome or supported browser
- **HTTPS Required**: PWAs require HTTPS (or localhost for testing)

### Service Worker Not Registering

- Check browser console for errors
- Verify `service-worker.js` is accessible
- Clear browser cache and try again
- Make sure you're on HTTPS or localhost

### Icons Not Showing

- Verify icons exist in `icons/` directory
- Check that file names match `manifest.json`
- Clear cache and reinstall the app
- Use DevTools → Application → Manifest to check icon URLs

### Offline Mode Not Working

- Ensure service worker is registered (check console)
- Wait a few seconds after first visit for caching
- Check DevTools → Application → Cache Storage
- Verify resources are being cached

## Customization

### Change App Colors

Edit `manifest.json`:
```json
{
  "background_color": "#your-color",
  "theme_color": "#your-color"
}
```

Also update in `index.html`:
```html
<meta name="theme-color" content="#your-color">
```

### Change App Name

Edit `manifest.json`:
```json
{
  "name": "Your App Name",
  "short_name": "Short Name"
}
```

### Add More Cached Resources

Edit `service-worker.js`:
```javascript
const urlsToCache = [
  './',
  './index.html',
  './your-file.js',
  // Add more files...
];
```

## Best Practices

1. **Always test on actual devices** - Simulators may not reflect real behavior
2. **Use HTTPS in production** - Required for service workers
3. **Keep cache versions updated** - Helps manage updates
4. **Test offline functionality** - Ensure critical features work offline
5. **Optimize icon sizes** - Keep icons under 100KB each
6. **Update manifest for branding** - Match your brand colors and names

## Resources

- [PWA Documentation](https://web.dev/progressive-web-apps/)
- [Service Worker API](https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API)
- [Web App Manifest](https://developer.mozilla.org/en-US/docs/Web/Manifest)
- [Can I Use PWA](https://caniuse.com/serviceworkers)

## Support

For issues or questions:
1. Check browser console for errors
2. Verify all files are in correct locations
3. Test in different browsers
4. Clear cache and try fresh install
