---
name: release-notes
description: "Create release notes for a new version tag. Gathers all commits, PRs, issues fixed, and breaking changes since a previous release. Creates the release notes markdown file, tags the repo, and pushes. Asks the user to confirm the base version to diff against."
license: Apache-2.0
metadata:
  author: mcp-gateway-registry
  version: "1.0"
---

# Release Notes Skill

Use this skill when the user wants to create release notes for a new version. This skill gathers all changes since a previous release, writes structured release notes following the project's established format, tags the repo, and pushes.

## Input

The skill takes a version tag as input:
- Format: `{major}.{minor}.{patch}` (e.g., `1.24.0`) - **no `v` prefix**, semver only
- Older releases (pre-`1.23.0`) used a `v` prefix (e.g., `v1.0.22`) - existing artifacts under `release-notes/v*.md` and tags `v1.0.x` are preserved as-is, but **new releases must use the bare-semver convention**
- If the user provides a `v`-prefixed version for a new release, strip the prefix and confirm

## Output

Creates a release notes file in `release-notes/` and tags the repo:
- `release-notes/{version}.md` - Release notes markdown file (e.g., `release-notes/1.24.0.md`)
- Git tag `{version}` pointing to the commit that includes the release notes

## Workflow

### Step 1: Determine the New Version Tag

1. Parse the version from user input. If not provided, ask the user what version to release.
2. Normalize to **bare semver** format (e.g., `v1.24.0` becomes `1.24.0`). Never prepend `v` for new releases.
3. Verify the tag does not already exist: `git tag -l {version}`.
4. If it exists, ask the user if they want to move it or choose a different version.

### Step 2: Determine the Base Version (Ask User to Confirm)

The release notes are incremental from a previous version. Determine the base version:

1. List existing release notes files (covers both old `v`-prefixed and new bare-semver names):
   ```bash
   ls release-notes/*.md
   ```
2. List existing git tags (any version-shaped tag, prefixed or bare):
   ```bash
   git tag --sort=-v:refname | grep -E '^v?[0-9]+\.[0-9]+\.[0-9]+'
   ```
3. Find the most recent tag. Note that the project switched from `v`-prefixed (`v1.0.22`) to bare-semver (`1.23.0`, `1.24.0`) - the most recent bare-semver tag is the right base for a new release.
4. **Ask the user to confirm the base version** using AskUserQuestion. Present the most recent tag as the recommended option and the 2-3 previous tags as alternatives. The user may want to skip intermediate tags (e.g., diff from `1.23.0` to `1.25.0`, skipping `1.24.0`).

### Step 3: Gather All Changes Between Base and HEAD

Run these commands in parallel to gather change data:

```bash
# All commits (including merges) between base and HEAD
git log {base_tag}..HEAD --oneline

# Non-merge commits only (for detailed change analysis)
git log {base_tag}..HEAD --oneline --no-merges

# Merge commits (to extract PR numbers)
git log {base_tag}..HEAD --oneline --grep="Merge pull request"

# Contributors -- direct authors of commits on main
# WARNING: this misses co-authors of squash-merged PRs (Step 4 #10 explains).
git log {base_tag}..HEAD --format="%aN" | sort | uniq -c | sort -rn

# Contributors -- per-PR commit authors (catches squash-merge co-authors)
# Squash merges collapse N branch commits into 1 commit on main authored by
# the merger, so `git log` above will not show the actual code authors.
# `gh pr view --json commits` returns the original branch commits with their
# authors intact, which is the only reliable way to credit everyone.
for pr in $(git log {base_tag}..HEAD --oneline --grep="Merge pull request\|(#[0-9]\+)$" | grep -oE "#[0-9]+" | tr -d '#' | sort -u); do
  gh pr view $pr --json number,author,commits \
    --jq '"PR #\(.number) | opener: \(.author.login) | commit_authors: \([.commits[].authors[].name] | unique | join(", "))"' 2>/dev/null
done

# Env var changes
git diff {base_tag}..HEAD -- .env.example

# Helm chart changes (any file change inside charts/ requires
# `helm dependency build/update` for stack-chart consumers, even if
# Chart.yaml dependency lists are unchanged - subchart templates,
# values, and helpers are repackaged into .tgz on dependency rebuild).
git diff {base_tag}..HEAD -- charts/ --stat

# If ANY of these report changes, the upgrade instructions MUST tell
# Helm/EKS users to run `helm dependency build` and `helm dependency update`:
git diff {base_tag}..HEAD --stat -- 'charts/registry/' 'charts/auth-server/' 'charts/mcpgw/' 'charts/mcp-gateway-registry-stack/' 'charts/mongodb-configure/' 'charts/keycloak-configure/'

# Helm chart dependency-list changes (separate signal: added/removed deps)
git diff {base_tag}..HEAD -- charts/registry/Chart.yaml charts/auth-server/Chart.yaml charts/mcp-gateway-registry-stack/Chart.yaml charts/mcpgw/Chart.yaml

# Closed issues since the base tag was cut
# Use the base tag's date as the floor; gh issue list does not natively
# support "closed-since-tag", so we filter by closedAt timestamp.
BASE_TAG_DATE=$(git log -1 --format=%cI {base_tag})
gh issue list --state closed --limit 200 --json number,title,closedAt,labels \
  --jq ".[] | select(.closedAt >= \"$BASE_TAG_DATE\") | \"\(.number) | \(.title) | \(.closedAt)\""

# Closed issues referenced by merged PRs in this release (most reliable mapping)
# For each PR number, the PR body usually has "Closes #N" or "Fixes #N" -- gh
# resolves these via the closingIssuesReferences field.
for pr in $(git log {base_tag}..HEAD --oneline --grep="Merge pull request" | grep -oE "#[0-9]+" | tr -d '#' | sort -u); do
  gh pr view $pr --json number,title,closingIssuesReferences \
    --jq '"\(.number) | \(.title) | closes: \(.closingIssuesReferences | map("#\(.number)") | join(","))"' 2>/dev/null
done
```

### Step 4: Categorize Changes

Analyze all commits and PRs to categorize them:

1. **Major Features**: New capabilities that warrant their own section with description and PR link. Look for commits with `feat:` prefix or PRs labeled `enhancement`/`feature-request`.

2. **Breaking Changes**: Changes that require user action during upgrade. Check for:
   - Helm chart dependency additions/removals (Chart.yaml changes)
   - Renamed or removed environment variables (.env.example diff)
   - Auth mechanism changes
   - API endpoint changes (removed or renamed routes)
   - Database schema changes

3. **New Environment Variables**: Extract from `.env.example` diff -- any new variables added.

4. **Bug Fixes**: Commits with `fix:` prefix or PRs labeled `bug`.

5. **Security Fixes**: Commits mentioning security, CVE, injection, bypass, XSS, etc.

6. **Infrastructure/Helm Changes**: Changes to charts/, terraform/, docker/.

7. **Dependency Updates**: Dependabot PRs and manual dependency bumps.

8. **Documentation**: Commits with `docs:` prefix.

9. **Closed Issues**: Issues closed in the release window. Build from the
   `closingIssuesReferences` of every merged PR in this release (most reliable -
   GitHub auto-closes issues referenced by `Closes #N` / `Fixes #N` in PR
   bodies), and supplement with manually-closed issues whose `closedAt` is
   between the base-tag commit date and HEAD. De-duplicate by issue number.

10. **Contributors**: Build the union of TWO sources, because neither alone is
    complete:
    - **Direct authors on main**: `git log {base_tag}..HEAD --format="%aN"` --
      catches anyone who pushed commits directly or whose PR was rebase/merge-
      committed.
    - **Per-PR commit authors**: for every merged PR in this release, run
      `gh pr view <num> --json commits --jq '[.commits[].authors[].name] | unique'`
      and union the results -- catches co-authors of **squash-merged** PRs,
      whose branch commits get collapsed into a single commit on main authored
      by the merger. Without this step, every contributor on a squash-merged
      branch except the merger is silently dropped.

    Then for each unique contributor name, resolve to a GitHub username by
    looking up a PR they appeared on:
    - If they opened a PR: `gh pr view <num> --json author --jq .author.login`.
    - If they were a co-author only (no PR opened): `gh pr view <num> --json commits --jq '.commits[].authors[] | select(.name == "<Display Name>") | .login'`
      (the per-commit `authors` array carries the GitHub login).
    - As a final fallback verify with `gh api users/<candidate>`; a 404 means
      the guess is wrong. **Never synthesize a username from a display name**
      ("Amit Arora" -> "amitarora" was wrong; the actual login is `aarora79`).

### Step 5: Write Release Notes

Create the file `release-notes/{version}.md` following this exact structure
(note: bare semver, no `v` prefix, e.g. `release-notes/1.24.0.md`):

```markdown
# Release {version} - {Short Title Summarizing Major Features}

**{Month} {Year}**

---

## Upgrading from {base_version}

This section covers everything you need to know to upgrade from {base_version} to {version}.

### Breaking Changes

{List each breaking change with clear explanation and remediation steps.
If no breaking changes, write: "There are no breaking changes in this release."}

### New Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| {VAR_NAME} | {default} | {description} |

{If no new env vars, write: "No new environment variables in this release."}

### Upgrade Instructions

#### Docker Compose

```bash
cd mcp-gateway-registry
git pull origin main
git checkout {version}

# Review new env vars in .env.example and update your .env if needed
# Then rebuild and restart:
./build_and_run.sh
```

#### Kubernetes / Helm (EKS)

```bash
cd mcp-gateway-registry
git pull origin main
git checkout {version}

# {If helm dependency changes: "REQUIRED: Rebuild dependencies"}
cd charts/mcp-gateway-registry-stack
helm dependency build
helm dependency update

# Update values.yaml if needed, then upgrade:
helm upgrade mcp-gateway . -f your-values.yaml
```

{CRITICAL: Include the `helm dependency build` and `helm dependency update`
commands whenever ANY file under `charts/` changes between base and HEAD,
not just Chart.yaml dependency-list changes. The packaged subchart `.tgz`
files inside `charts/mcp-gateway-registry-stack/charts/` are gitignored
and only get repackaged when consumers run `helm dependency build/update`.
If subchart templates/values/helpers changed, a plain `git pull` followed
by `helm upgrade` will use the OLD packaged subcharts, missing the changes.

Only omit `helm dependency build/update` if `git diff {base_tag}..HEAD --stat -- charts/`
shows ZERO files changed.}

#### Terraform / ECS

```bash
cd mcp-gateway-registry
git pull origin main
git checkout {version}

# Update your .tfvars with any new variables
cd terraform/aws-ecs
terraform plan
terraform apply
```

---

## Major Features

### {Feature Name}

{Description of the feature -- what it does, why it matters, key capabilities as bullet points.}

[PR #{number}](https://github.com/agentic-community/mcp-gateway-registry/pull/{number})

{Repeat for each major feature.}

---

## What's New

{Group changes by category using subsections. Use bullet points with PR/commit references.}

### {Category Name}
- {Change description} (#{pr_number})
- {Change description} (#{pr_number})

{Common categories: Deployment, Helm Chart Improvements, Security Fixes,
Authentication, Infrastructure, Frontend Improvements, Documentation.
Only include categories that have changes.}

---

## Bug Fixes

- {Bug fix description} (#{pr_number})
- {Bug fix description} (#{pr_number})

---

## Closed Issues

| Issue | Title | Closed By |
|-------|-------|-----------|
| #{issue_number} | {issue_title} | {PR #{pr_number} or "manual"} |

{List all issues closed in the release window, sorted by issue number
descending. "Closed By" is the PR that closed the issue (via
`closingIssuesReferences`) or "manual" for issues closed without a PR
reference. If no issues were closed in this window, write:
"No issues were closed in this release window."}

---

## Pull Requests Included

| PR | Title |
|----|-------|
| #{number} | {title} |

{List ALL merged PRs between base and HEAD, sorted by PR number descending.}

---

## Security Dependency Updates

| Package | Previous | Updated | Scope |
|---------|----------|---------|-------|
| {package} | {old_version} | {new_version} | {scope} |

{Only include this section if there are dependency version bumps.}

---

## Contributors

Thank you to all contributors for this release:

- **{Full Name}** ([@{github_username}](https://github.com/{github_username}))

{List all contributors from the UNION of (a) direct authors on main and
(b) per-PR commit authors via `gh pr view --json commits` (Step 4 #10).
Resolve every GitHub username from a real PR -- via `.author.login` if they
opened a PR, or via `.commits[].authors[].login` if they were a co-author
only. Never synthesize usernames from display names. Sort by commit count
descending.}

---

## Support

- [GitHub Issues](https://github.com/agentic-community/mcp-gateway-registry/issues)
- [GitHub Discussions](https://github.com/agentic-community/mcp-gateway-registry/discussions)
- [Documentation](https://github.com/agentic-community/mcp-gateway-registry/tree/main/docs)

---

**Full Changelog:** [{base_version}...{version}](https://github.com/agentic-community/mcp-gateway-registry/compare/{base_version}...{version})
```

### Step 6: Present Draft for User Review

After writing the release notes file:

1. Tell the user the file has been created at `release-notes/{version}.md`
2. Present a brief summary:
   - Number of major features
   - Number of PRs included
   - Number of bug fixes
   - Number of closed issues
   - Any breaking changes
   - Contributor count
3. Ask the user to review the file and confirm it looks good, or request changes

### Step 7: Commit, Tag, and Push

Once the user confirms the release notes are ready:

1. **Commit the release notes:**
   ```bash
   git add release-notes/{version}.md
   git commit -m "docs: Add {version} release notes"
   ```

2. **Push the commit:**
   ```bash
   git push origin main
   ```

3. **Create or move the git tag** to point at this latest commit (which includes the release notes):
   ```bash
   # If tag already exists, delete it locally and remotely first
   git tag -d {version} 2>/dev/null || true
   git push origin :refs/tags/{version} 2>/dev/null || true

   # Create tag on current HEAD (bare semver, no v prefix)
   git tag {version}

   # Push tag
   git push origin {version}
   ```

4. **Verify:**
   ```bash
   git log --oneline -1
   git tag -l {version} --format="%(refname:short) -> %(objectname:short)"
   ```

5. Tell the user the release notes are committed and the tag is created and pushed.

## Important Rules

- **Never skip the user confirmation** for base version in Step 2. The user may want to create release notes that span multiple versions.
- **Never include emojis** in the release notes file. The project CLAUDE.md prohibits emojis in documentation.
- **Never include Claude Code attribution** or "Co-Authored-By" lines in commits.
- **Always use the `release-notes/` directory** at the project root for the output file.
- **Always include upgrade instructions** for all three deployment methods (Docker Compose, Helm/EKS, Terraform/ECS).
- **Always list breaking changes first** in the upgrade section -- this is the most critical information for operators.
- **Always verify Helm Chart.yaml diffs** to detect dependency additions/removals -- these are the most common breaking changes for EKS users.
- **Always check the full `charts/` tree diff**, not just `Chart.yaml`. If ANY file under `charts/` changed between base and HEAD, the upgrade instructions MUST include `helm dependency build` and `helm dependency update` for stack-chart consumers. The packaged `.tgz` subcharts inside `charts/mcp-gateway-registry-stack/charts/` are gitignored and only repackage when those commands run -- a plain `git pull` + `helm upgrade` will silently use stale subcharts.
- **Always credit squash-merge co-authors.** `git log {base}..HEAD` only sees the squashed commit's single author, so co-authors on the source branch get dropped. Always also iterate every merged PR with `gh pr view <num> --json commits --jq '[.commits[].authors[].name] | unique'` and union the results into the contributor list.
- **Never synthesize GitHub usernames from display names.** Resolve every login from a real PR (`gh pr view <num> --json author` for openers, or `.commits[].authors[].login` for co-authors), and verify uncertain ones with `gh api users/<candidate>` (404 = wrong guess). Past mistakes: "Amit Arora" -> `amitarora` (wrong; actual: `aarora79`); "Nathan Fernandes Pedroza" -> `nathanfernandes` (wrong; actual: `nathanzilgo`).

## Example Usage

```
User: /release-notes v1.0.16