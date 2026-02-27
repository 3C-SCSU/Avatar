#!/usr/bin/env python3
# Author: Aditiya Malla
# CSCI 310 Midterm - Option 3

import datetime
import re
from pathlib import Path

BEGIN = "# BEGIN MANAGED KEYS"
END = "# END MANAGED KEYS"

def main():
    ak_path = Path.home() / ".ssh" / "authorized_keys"
    
    if not ak_path.exists():
        print("authorized_keys not found")
        return

    lines = ak_path.read_text().splitlines()

    if BEGIN not in lines or END not in lines:
        print("Managed block not found")
        return

    start = lines.index(BEGIN)
    end = lines.index(END)

    today = datetime.date.today()

    new_block = []
    for line in lines[start+1:end]:
        if "expiry=" in line:
            date_str = re.search(r"expiry=(\d{4}-\d{2}-\d{2})", line)
            if date_str:
                expiry = datetime.date.fromisoformat(date_str.group(1))
                if expiry >= today:
                    new_block.append(line)
            else:
                new_block.append(line)
        else:
            new_block.append(line)

    updated = lines[:start+1] + new_block + lines[end:]

    backup = ak_path.with_suffix(".bak")
    backup.write_text("\n".join(lines) + "\n")
    ak_path.write_text("\n".join(updated) + "\n")

    print("Expired keys removed (if any).")
    print("Backup created:", backup)

if __name__ == "__main__":
    main()

