# Agent Instructions

> This file is the single source of truth for AI coding assistants working in this project.
> It is mirrored across CLAUDE.md, AGENTS.md, and GEMINI.md so the same instructions load in any AI environment.

## 1. Architecture Overview

You operate as the **Orchestration Layer** in a system that separates intent from execution.

- **Read instructions first, then act.** Before writing code or solving a problem, check if a Skill exists in `skills/` that covers the domain. Skills contain exact specifications, UUIDs, algorithms, and patterns that you must follow precisely.
- **Push complexity into deterministic code.** 90% accuracy per step = 59% success over 5 steps. Write reliable, testable scripts instead of winging complex logic inline.
- **Self-anneal when things break.** Read the error → fix the root cause → test → update the relevant Skill or this document with what you learned. Errors are system-improvement opportunities, not dead ends.

## 2. Skills System

Skills live in `skills/` and follow this structure:

```
skills/
├── skill-name/
│   ├── SKILL.md          (required — YAML frontmatter + instructions)
│   ├── scripts/          (optional — deterministic helper scripts)
│   ├── references/       (optional — schemas, API docs, detailed specs)
│   └── assets/           (optional — templates, images, boilerplate)
```

**Routing rule:** When you receive a task involving a specific domain (e.g., BLE sensors, payment systems, 3D rendering), search `skills/` for a matching Skill **before** writing any code. Read the SKILL.md frontmatter `description` to determine relevance. If a Skill matches, load and follow its instructions exactly.

**Creating new Skills:** Use the `skills/skill-creator/` Skill for guidance on creating new Skills.

## 3. Branching & Git Workflow

- **NEVER push directly to `main`.**
- **Always** create a feature branch: `git checkout -b feature/name-of-feature` or `fix/name-of-fix`.
- **Commit messages:** Clear, descriptive, conventional commits (e.g., `feat: added setup screen`, `fix: header padding on iOS`).
- **Local testing first:** Build and test the Android APK locally using `npm run build:android` to verify on a physical device before submitting a PR.
- When work is complete and tested locally, push the branch and instruct the user to create a **Pull Request (PR)** on GitHub.
- **Do not merge PRs locally.** Instruct the user to merge via the GitHub UI.

## 4. CI/CD & Versioning

Our pipeline is designed to save expensive macOS runner minutes and keep the TestFlight history clean.

- **Pull Requests:** GitHub Actions will **NOT** auto-build TestFlight on PRs. PRs are for code review and linting only.
- **Merging to Main:** Triggers a GitHub Action to build the **Android Release APK**, uploaded as an artifact. Does **NOT** push to TestFlight.
- **iOS TestFlight (Manual):** Only run when `main` has enough stable features. Trigger manually from the GitHub Actions tab → "Build Apps" → Run Workflow.
- **Releases & Versioning:** Production deployments happen **ONLY** when a GitHub Release is created. The Release Tag (e.g., `v1.0.1`) determines the app version. The Action extracts the tag and updates `package.json` automatically.
- **NEVER** manually bump versions in codebase files.

## 5. General Coding Standards

- **Be proactive but safe.** Suggest refactors when needed, fix failing tests when found.
- **Keep it clean.** Remove debug `<Text>`, `console.log`, or diagnostic logs before finalizing a feature.
- **Cross-Platform React Native.** Ensure all UI/UX changes look consistent on both iOS and Android. Use platform-specific code only when absolutely necessary.
- **Backend Integrations (AWS Amplify).** Respect the Amplify folder structure and existing models. Do not modify auto-generated Amplify backend files directly unless using the Amplify CLI/tools.
- **Check for existing tools first.** Before writing a script, check if one already exists in the relevant Skill's `scripts/` directory.

## 6. Operating Principles

1. **Skills-first.** Always check `skills/` before starting domain-specific work.
2. **Self-anneal.** Fix errors → update tools → test → update Skill/instructions → system gets stronger.
3. **Update instructions as you learn.** This file and Skills are living documents. When you discover API constraints, better approaches, common errors, or timing expectations — update the relevant document. But don't overwrite without asking unless explicitly told to.
4. **Local files are for processing.** Deliverables live in cloud services or on-device where users can access them.

## 7. QA Checklist

After completing any significant build (new screen, feature, BLE integration, backend setup, ML model), generate a `TEST_PLAN.md` in the project root without being asked.

Derive every test case from what was actually built — not from a generic template. Read the relevant code and flows, identify the critical user paths, and write 5–15 concrete device-testable tests.

```markdown
# Test Plan: [Feature Name]

**Date:** YYYY-MM-DD
**Built:** [1-sentence summary]
**Tester:** _______________
**Device(s):** _______________

## Tests

| # | Action | Expected result | Result | Notes |
|---|--------|-----------------|--------|-------|
| 1 | [Concrete action] | [Concrete outcome] | ☐ Pass  ☐ Fail | |

## Overall

☐ PASS  ☐ FAIL

**Blocking issues:**
```

- Every test must be doable on a physical device
- Mark **(CRITICAL)** on tests where failure blocks shipping
- Concrete beats complete: "tap Login → lands on Dashboard in <2s" not "login works"
