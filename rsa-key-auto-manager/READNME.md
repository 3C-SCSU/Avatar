# RSA Key Automation Manager

## Overview

The RSA Key Automation Manager automates SSH public key management on Linux servers.  
It adds expiry timestamps to user keys, removes expired ones automatically, and ensures synchronized metadata for improved security and reliability.

This script is developed as part of the Avatar Open-Source Project (3C-SCSU) at St. Cloud State University, focusing on secure key lifecycle management across multi-user environments.

---

## Features

- **Automatic Expiry Management**
  - Adds expiry timestamps to authorized keys.
  - Cleans up expired keys automatically.
- **User-Scoped Key Control**

  - Supports all users under `/home/*` and `/root`.
  - Handles each user’s `authorized_keys` file independently.

- **Metadata Integration**

  - Embeds expiry info inside SSH key comments:
    ```bash
    ssh-ed25519 AAAA... user@host {"expiry":"2025-12-12T00:00"}
    ```

- **Cron-Based Scheduling**

  - Registers automated jobs:
    - Every Friday 23:59 → remove expired keys
    - Every Monday 00:00 → add expiry to new keys

- **Thread-Safe File Access**

  - Uses Python’s `filelock.FileLock` to prevent race conditions during writes.

- **Comprehensive Logging**
  - Records all actions in `/var/log/ssh_key_cleanup.log`.

---

## Installation

Follow these steps to install and configure the program:

1. **Clone the repository**
   ```bash
   git clone https://github.com/3C-SCSU/avatar.git
   cd avatar
   ```
2. **Locate the script**
   ```bash
   ssh_key_manager.py
   ```
3. **Set file permissions**
   ```bash
   chmod +x ssh_key_manager.py
   ```
4. **Install dependency**
   ```bash
   pip install filelock
   ```

## Usage

1. **Add expiry to all existing keys**
   ```bash
   sudo python3 ssh_key_manager.py init
   ```
   Scans all user directories and appends expiry metadata if missing.
2. **Clean expired keys**
   ```bash
   sudo python3 ssh_key_manager.py cleanup
   ```
   Removes expired keys and logs all changes.
3. **Register cron jobs**
   ```bash
   sudo python3 ssh_key_manager.py register_cron
   ```
   Adds two recurring jobs to the system crontab:
   ```bash
   */1 * * * * /usr/bin/python3 '/path/to/ssh_key_manager.py' init >> /var/log/ssh_key_init_cron.log 2>&1
   */2 * * * * /usr/bin/python3 '/path/to/ssh_key_manager.py' cleanup >> /var/log/ssh_key_cleanup_cron.log 2>&1
   ```

## Program Structure

```scss
ssh_key_manager.py
│
├── parse_expiry()              # Parse expiry string into datetime
├── is_expired()                # Check if a key is expired
├── add_expiry_tag()            # Add expiry metadata to a key line
├── init_keys()                 # Initialize expiry tags for all users
├── cleanup_expired_keys()      # Remove expired keys and log results
├── register_cron()             # Add cron jobs for automation
└── main()                      # Command-line entry point
```

## Example Workflow

```bash
# 1. Initialize expiry metadata for all users
sudo python3 ssh_key_manager.py init

# 2. Check results
cat ~/.ssh/authorized_keys

# 3. Remove expired keys manually
sudo python3 ssh_key_manager.py cleanup
```

## Logging

All cleanup actions are stored in:

```lua
/var/log/ssh_key_cleanup.log
```

Example log entry:

```sql:
2025-10-26 12:00:00 - Expired key removed for /home/lee: ssh-ed25519 AAAA... user@host
```

## Notes and Troubleshooting

Ensure SSH permissions are correct:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
sudo chown $USER:$USER ~/.ssh/authorized_keys
```

If remote login fails after editing keys, manually re-add your public key:

```bash
echo 'ssh-ed25519 AAAA... user@host {"expiry":"2025-12-12T00:00"}' >> ~/.ssh/authorized_keys
```

Default expiry date is set to:

```bash
2025-12-12T00:00
```

You can change this by modifying the DEFAULT_EXPIRY variable in the script.

## Contributors

| Name                  | Role                                           | Contribution                                                                      |
| --------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------- |
| Jiali Zhao            | Developer / Primary Author /Project Supervisor | Designed and implemented most of the program logic, automation flow, and testing. |
| Kenneth R. Uebel      | Collaborator                                   | Assisted in testing and documentation review.                                     |
| Saad A. Pervez Mughal | Collaborator                                   | Assisted in testing and documentation review.                                     |

## System Requirements

-Python 3.8+
-Linux (Debian-based)
-cron service enabled
-filelock Python library

## License

This project is licensed under the MIT License.
You are free to use, modify, and distribute this program with attribution.

Maintained by: 3C-SCSU Avatar Team
