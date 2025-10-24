# 👋 START HERE - Avatar BCI Project

**Version: v1.0.0** 🤖 *Auto-versioned* | [Changelog](CHANGELOG.md) | [Contributing](CONTRIBUTING.md)

## 🎯 Quick Start (2 Steps)

### Step 1: Make sure you're in the project directory

### Step 2: Run the application
```bash
./run_gui.sh
```

**That's it!** The script handles everything automatically.

---

## 📖 New User? Read This First

### What is This Project?
Avatar is a Brain-Computer Interface (BCI) application that:
- Reads brainwaves using OpenBCI headset
- Uses machine learning to predict intended actions
- Controls drones and robots with your mind

### Do I Need Hardware?
**No!** The application works without hardware:
- Runs in simulation mode without BCI headset
- Can test UI without drone
- Can test UI without NAO robot

Hardware is optional for testing.

---

## 📚 Documentation Guide

Choose the document that fits your needs:

### Detailed Guides
📄 **SETUP_GUIDE.md** - Complete setup instructions  
- Installation steps
- Project structure
- Full troubleshooting guide
- Feature documentation

### Original Documentation
📄 **README.md** - Project overview and contribution guide  
- Project history
- Contribution guidelines
- Team information

---

## 🎮 Application Features

Once running, you'll see these tabs:

### Top Navigation
1. **Brainwave Reading** - Real-time BCI data and predictions
2. **Manual Drone Control** - Control DJI Tello drone
3. **Manual NAO Control** - Control NAO robot

### Bottom Navigation
1. **Brainwave Visualization** - Data visualization
2. **File Shuffler** - Privacy-preserving data tools
3. **Transfer Data** - Data transfer utilities
4. **Developers** - Team statistics and charts

---

## ⚠️ Common Questions

### "I see error messages about drone/NAO"
**This is normal!** If you don't have the physical hardware connected, you'll see errors. The application will continue to work - you can still test the UI and other features.

### "I see a warning about pkg_resources"
**This is harmless!** It's a warning from the BrainFlow library and doesn't affect functionality.

### "The application won't start"
Try this:
```bash
chmod +x run_gui.sh
./run_gui.sh
```

If that doesn't work, see **SETUP_GUIDE.md** for detailed troubleshooting.

---

## 🚀 Ready to Start?

```bash
./run_gui.sh
```

Enjoy the Avatar BCI project! 🧠🤖✨

---

## 🤖 New: Automated Versioning!

This project now uses **automated versioning**! When you contribute:
- Use conventional commits (e.g., `feat(bci): add new feature`)
- Versions and changelogs are generated automatically
- See [CONTRIBUTING.md](CONTRIBUTING.md) for commit format

## 🆘 Need Help?

1. **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md) for commit guidelines
2. **Setup**: [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed installation
3. **Contact**:
   - Discord: https://discord.gg/mxbQ7HpPjq
   - Email: 3c.scsu@gmail.com
   - GitHub: https://github.com/3C-SCSU/Avatar/issues

---
