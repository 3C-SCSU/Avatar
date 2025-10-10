# 🐛 Bug Fixes Repository

Welcome to the **bug-fixed** directory of the [Avatar Project](https://github.com/3C-SCSU/Avatar).  
This folder serves as a centralized collection of **documented bug resolutions** across the project.

Each subfolder inside this directory contains:

- A detailed description of the issue and how it was resolved.
- Any relevant screenshots, logs, or code snippets.
- Associated files or patches contributed by the development team.

---

## 📂 Folder Structure

```
bug-fixed/
│
├── linux-mint-login/
│   ├── README.md
│   ├── images/
│   │   ├── commented-code.png
│   │   └── login-window.png
│   └── additional-docs/
│
└── (more folders for other issue fixes)
```

---

## 🧭 Purpose

The goal of this directory is to maintain **transparency and traceability** of all major bug fixes within the Avatar project.  
By documenting each fix thoroughly, we ensure:

- Future contributors can understand the context behind fixes.
- Similar issues can be resolved faster.
- The project remains stable and well-maintained.

---

## 🧩 Current Entries

### 🟢 `linux-mint-login/`

**Issue:** [#359 – Bug: Solve login or reinstall Linux Mint](https://github.com/3C-SCSU/Avatar/issues/359)  
**Summary:**  
Resolved a login bypass bug where Linux Mint would skip password authentication at startup.  
Fix involved modifying LightDM configuration and login manager settings.

**Files included:**

- `README.md` — detailed report and fix documentation
- `images/` — screenshots of configuration changes and results
- `additional-docs/` — any extra supporting files (if applicable)

---

## 🤝 Contributions

If you’ve fixed a bug and want to document it:

1. Create a new subfolder under `bug-fixed/` named after the issue or topic.
2. Add a Markdown file describing the fix (similar to `README.md`).
3. Include any related screenshots, code diffs, or test logs.
4. Submit a **pull request** with your additions.

---

## 🧾 Maintainers

This section is maintained by members of the **Avatar Project Development Team**.  
Please follow the repository’s [CONTRIBUTING.md](../CONTRIBUTING.md) guidelines for documentation and pull requests.

---

> _Documenting fixes today prevents bugs tomorrow._
