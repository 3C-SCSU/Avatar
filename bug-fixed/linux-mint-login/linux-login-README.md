# Fix for Login Bypass Bug on Linux Mint (Issue #359)

**Contributors:** Evan Brisbin, Minh Quan Tran, Artyom Megega
**Date:** 10/10/2025
**Related Issue:** [#359 – Bug: Solve login or reinstall Linux Mint](https://github.com/3C-SCSU/Avatar/issues/359)

---

## 🧩 Summary

On some Chromebooks running Linux Mint, the system would automatically log in at startup — skipping the password prompt even though login security was enabled. This was a critical security issue identified in Issue #359.

Our team investigated the problem and implemented a fix by adjusting the LightDM configuration and login manager settings.

---

## 🧠 Root Cause

The issue was traced to incorrect or conflicting settings within **LightDM**, Linux Mint’s display manager.  
Specifically, the configuration file contained parameters that enabled automatic user login.

---

## 🔧 Steps Taken to Resolve

1. **Opened the LightDM configuration file**  
   Path: `/etc/lightdm/lightdm.conf`

2. **Commented out the following lines:**

   ```ini
   # autologin-guest=false
   # autologin-user-timeout=0
   ```

   These lines were responsible for controlling automatic and guest logins.

3. **Adjusted Login Manager settings (GUI):**

   - Enabled **“Allow manual login”**
   - Cleared saved **username** and **login delay** fields

4. **Rebooted the system** to verify the fix.

---

## 🖼️ Screenshots

Below are screenshots showing the configuration and results of the fix:

![Commented Code Lines](./docs/images/commented-code.jpg)
![Login Window Configuration](./docs/images/login-window.jpg)

---

## ✅ Result

After applying these changes:

- The Chromebook now **requires a password** at every login.
- Automatic and guest logins are **disabled**.
- The login process works as intended without skipping authentication.

---

## 🧭 Future Recommendations

- Double-check LightDM settings after OS updates or reinstallation.
- Consider adding a post-install script to enforce secure login defaults.
- Continue disabling password-based remote logins and use SSH keys for admin access.

---

## 🗂️ Files Modified

- `/etc/lightdm/lightdm.conf`
- System Settings → Login Window (manual login, username field cleared)

---

## 🧾 Verification

Tested successfully on Linux Mint installation running on a repurposed Chromebook.  
The login screen now consistently prompts for password authentication.

---

**Fix implemented and verified.**  
This pull request documents the resolution of the login bypass bug described in Issue #359.
