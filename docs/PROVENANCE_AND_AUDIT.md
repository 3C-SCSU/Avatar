# Provenance & Audit Architecture

## 1. Purpose

This document defines the technical provenance and audit mechanism used to preserve a verifiable record of pull-request activity within the repository.

The system is implemented through GitHub Actions and is designed to create a consistent technical record of:

* Pull-request and repository metadata
* Exact commit identifiers
* Author and committer information
* Commit timestamps
* Available cryptographic commit signatures
* CI runner execution environment
* GitHub Actions workflow execution metadata
* SHA-256 integrity hashes of generated audit records

The objective is to preserve a transparent chain of technical evidence for project contributions. This system maintains a verifiable record of pull request metadata, commit history, cryptographic signature information, and CI execution details. Each audit record is accompanied by a SHA-256 checksum to support integrity verification and long-term traceability.

These records provide a consistent technical reference for reviewing contribution history, revision timelines, and repository activity across distributed development environments.

> **Scope:** This system documents contribution provenance and CI execution to support project auditing, transparent collaboration, and long-term traceability.


---

## 2. Pipeline Overview

The provenance audit is automatically executed by:

```text
.github/workflows/pr-audit.yml
```

The workflow is triggered when a pull request is:

* Opened
* Synchronized with new commits
* Reopened

The resulting processing pipeline is:

```text
┌─────────────────────┐
│   Contributor Git   │
│       Commit        │
└──────────┬──────────┘
           │
           │ Pull Request
           ▼
┌─────────────────────┐
│      GitHub         │
│ Repository / PR     │
└──────────┬──────────┘
           │
           │ PR event
           ▼
┌─────────────────────┐
│   GitHub Actions    │
│  Hosted CI Runner   │
└──────────┬──────────┘
           │
           │ Inspect exact
           │ PR head commit
           ▼
┌─────────────────────┐
│   Audit Manifest    │
│    pr_audit.txt     │
└──────────┬──────────┘
           │
           │ SHA-256
           ▼
┌─────────────────────┐
│ Integrity Digest    │
│  pr_audit.sha256    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ GitHub Actions      │
│      Artifact       │
└─────────────────────┘
```

Each stage contributes verifiable metadata to the provenance record, creating a consistent audit trail across the contribution and CI workflow.


---

## 3. Pull Request Provenance

For each triggered pull request, the workflow records repository-level metadata identifying the contribution being audited.

The audit manifest records:

| Field                | Purpose                                              |
| -------------------- | ---------------------------------------------------- |
| PR Number            | Identifies the pull request                          |
| PR Title             | Human-readable contribution identifier               |
| PR State             | Records the PR state at audit time                   |
| Contributor Login    | Identifies the GitHub account associated with the PR |
| Head Commit SHA      | Identifies the exact commit being audited            |
| Source Repository    | Identifies where the PR originates                   |
| Source Branch        | Identifies the source branch                         |
| Target Repository    | Identifies the destination repository                |
| Target Branch        | Identifies the destination branch                    |
| PR Created Timestamp | Records PR creation time                             |
| PR Updated Timestamp | Records the latest PR update time                    |

The **head commit SHA** is particularly important because it provides an immutable Git object identifier for the specific revision examined by the audit.

The workflow does not rely solely on the branch name, since branch contents may change over time. The SHA identifies the exact commit associated with the audited PR event.

---

## 4. Commit Metadata

The workflow inspects the exact PR head commit and records its:

* Commit SHA
* Author and committer
* Author and committer timestamps (ISO 8601)

Author and committer identities are recorded separately to preserve Git's distinction between the original author of a change and the person who committed it.


---

## 5. Cryptographic Commit Signatures

The workflow inspects Git commit objects for cryptographic signatures, including SSH and OpenPGP/GPG signatures.

When present, the audit manifest records the signature's presence and detected type, along with the raw signature material.

```text
Signature Present: YES
Signature Type: SSH

--- Raw Signature ---
gpgsig -----BEGIN SSH SIGNATURE-----
gpgsig ...
gpgsig -----END SSH SIGNATURE-----
```

By extracting the signature directly from the Git commit object, the workflow preserves the original cryptographic signature material as part of the audit record. This provides a useful reference for subsequent inspection and verification.

---

## 6. CI Execution Environment

The audit runs on GitHub-hosted infrastructure using `runs-on: ubuntu-latest`.

```yaml
runs-on: ubuntu-latest
```

The workflow records the following metadata:

**Runner Environment**

```text
Runner OS
Runner Architecture
uname -a
```

**Workflow Execution**

```text
Repository
Workflow
Run ID
Run Number
```

The `uname -a` output captures kernel and host information visible at runtime, while the GitHub Actions identifiers link the audit artifact to its originating workflow execution.

---

## 7. Audit Artifact Integrity

After the audit manifest is generated, the workflow calculates a SHA-256 digest:

```bash
sha256sum pr_audit.txt > pr_audit.sha256
```

The resulting files are stored together as a GitHub Actions artifact:

```text
audit_artifacts/
├── pr_audit.txt
└── pr_audit.sha256
```

The SHA-256 digest provides a deterministic integrity value for the generated audit manifest.

If the contents of `pr_audit.txt` are changed after the digest is calculated, recalculating the SHA-256 value will produce a different digest.

This provides a straightforward mechanism for detecting subsequent modification of the audit record.

---

## 8. Chain of Custody

The provenance model connects the following identifiers:

```text
Pull Request
      │
      ▼
PR Head Commit SHA
      │
      ▼
Git Commit Metadata
      │
      ├── Author
      ├── Committer
      ├── Timestamps
      └── Cryptographic Signature
      │
      ▼
GitHub Actions Run
      │
      ├── Workflow
      ├── Run ID
      ├── Runner OS
      ├── Runner Architecture
      └── Kernel Information
      │
      ▼
Audit Manifest
      │
      ▼
SHA-256 Digest
      │
      ▼
Stored CI Artifact
```

This creates a reproducible relationship between the repository event, the exact Git object being audited, the CI execution that examined it, and the resulting audit record.

The intent is that a later reviewer can correlate these identifiers rather than relying solely on manually maintained records.

---

## 9. Distributed Execution Boundaries

The provenance pipeline spans multiple components, each contributing distinct technical evidence to the audit record.

| Component                   | Provenance Contribution                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------------------------- |
| Git Commit Object | Commit objects, author and committer metadata, timestamps, and embedded cryptographic signatures        |
| GitHub Repository           | Repository identity, pull request details, source and target branches, and commit references            |
| GitHub Actions              | Workflow execution identifiers, runner architecture, operating system, and runtime environment metadata |
| Audit Manifest              | Consolidated contribution metadata, commit signature information, and execution records                 |
| Integrity Verification      | SHA-256 digest used to verify the audit manifest's integrity                                            |

### Execution Environment and Evidence

The workflow runs on GitHub-hosted CI infrastructure. Runtime commands such as `uname -a` capture information about the environment executing the audit, while Git metadata and GitHub event data provide the corresponding contribution and revision context.

These sources are brought together in a single audit artifact, linking the pull request, the specific commit under review, and the CI execution that generated the record.

This distributed architecture provides a consistent, machine-generated provenance record across repository activity and automated execution, supporting transparent contribution review and long-term auditability.

---

## 10. Contributor Signature Guidelines

Contributors are encouraged to sign Git commits using SSH or OpenPGP/GPG to provide additional cryptographic provenance.

* **SSH signing:** Sign commits using an SSH key configured in Git.
* **OpenPGP/GPG signing:** Sign commits using a GPG key configured in Git.
* **Key security:** Never commit private keys or expose them through repository files or CI workflows.
* **Identity verification:** Associate signing keys with contributor identities through a trusted verification process.

For setup instructions, see GitHub's documentation on [signing commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits).


---

## 11. Open Custodial Architecture

The project follows an open-source custodial governance model designed to maintain transparent, decentralized, and verifiable contribution records. Provenance is established through Git's content-addressed commit history, repository activity, and independently executed CI audits.

Responsibilities are distributed across defined trust boundaries:

| Component           | Responsibility                                                        |
| ------------------- | --------------------------------------------------------------------- |
| Contributors        | Create and submit commits, with optional cryptographic signatures     |
| Git                 | Maintains commit objects and content-based identifiers                |
| GitHub              | Hosts repository history, pull requests, and review activity          |
| GitHub Actions      | Generates provenance manifests and SHA-256 integrity checksums        |
| Project Maintainers | Govern contribution review, acceptance, and repository integration    |
| Audit Artifacts     | Preserve records for subsequent inspection and integrity verification |

This separation of responsibilities supports decentralized contribution and transparent review. Cryptographic identifiers and integrity checks provide verifiable references to recorded project activity, while retained audit artifacts support long-term traceability and accountability.


---

## 12. Development Provenance and Independence

The audit pipeline maintains a verifiable record of contribution activity across distributed development environments. Commit metadata, cryptographic signature information, repository history, and independently executed CI records provide complementary evidence of how project revisions progress through the development workflow.

By preserving these records alongside their associated commit identifiers and integrity checksums, the project maintains a consistent technical reference for reviewing contribution timelines, revision history, and development activity over time.


---

## 13. Audit Record Example

A completed audit may contain sections similar to:

```text
=== PULL REQUEST AUDIT MANIFEST===

--- Event Information ---
Event Name: pull_request
Event Action: opened

--- PR Information ---
PR Number: 42
PR Title: Example contribution
PR State: open
Contributor Login: contributor
Head Commit SHA: <commit-sha>

--- Repository Information ---
Source Repository: contributor/project
Source Branch: feature/example

Target Repository: organization/project
Target Branch: main

--- Timestamp Information ---
PR Created: <UTC timestamp>
PR Updated: <UTC timestamp>

--- CI Runner Node Execution Environment ---
Runner OS: Linux
Runner Architecture: X64
<uname -a output>

=== COMMIT METADATA ===
Audited PR Head SHA: <commit-sha>

Commit SHA: <commit-sha>
Author: <author>
Committer: <committer>
Author Date: <ISO-8601 timestamp>
Committer Date: <ISO-8601 timestamp>

=== COMMIT SIGNATURE ===
Signature Present: YES
Signature Type: SSH

--- Raw Signature ---
<embedded signature>

--- Workflow Information ---
Repository: <repository>
Workflow: contributor Provenance & PR Audit
Run ID: <run-id>
Run Number: <run-number>
```

The corresponding SHA-256 digest is stored alongside the manifest.

---

## Summary

The PR provenance workflow converts repository activity into a verifiable technical audit record.

For each audited pull request, the system connects:

```text
Pull Request
    ↓
Commit SHA
    ↓
Commit Metadata
    ↓
Signature Information
    ↓
CI Execution
    ↓
Audit Manifest
    ↓
SHA-256 Integrity Digest
    ↓
Stored Artifact
```

Together, these records provide a transparent and traceable history of project contributions, supporting repository governance, distributed development, and long-term auditability.
