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
- Format: `v{major}.{minor}.{patch}` (e.g., `v1.0.16`)
- The user may also provide the version without the `v` prefix (e.g., `1.0.16`) -- always normalize to `v`-prefixed format

## Output

Creates a release notes file in `release-notes/` and tags the repo:
- `release-notes/v{version}.md` - Release notes markdown file
- Git tag `v{version}` pointing to the commit that includes the release notes

## Workflow

### Step 1: Determine the New Version Tag

1. Parse the version from user input. If not provided, ask the user what version to release.
2. Normalize to `v`-prefixed format (e.g., `1.0.16` becomes `v1.0.16`).
3. Verify the tag does not already exist: `git tag -l v{version}`.
4. If it exists, ask the user if they want to move it or choose a different version.

### Step 2: Determine the Base Version (Ask User to Confirm)

The release notes are incremental from a previous version. Determine the base version:

1. List existing release notes files:
   ```bash
   ls release-notes/v*.md
   ```
2. List existing git tags (version tags only):
   ```bash
   git tag --sort=-v:refname | grep '^v[0-9]'
   ```
3. Find the most recent release notes file and the most recent git tag. Present these to the user:
   ```
   Found release notes: v1.0.15, v1.0.14, v1.0.13, ...
   Found git tags: v1.0.15, v1.0.13, v1.0.12, ...

   This release (v1.0.16) appears to be incremental from v1.0.15.
   Confirm base version, or specify a different one if you want to skip versions.
   ```
4. **Ask the user to confirm the base version** using AskUserQuestion. Present the most recent tag as the recommended option and the 2-3 previous tags as alternatives. The user may want to skip intermediate tags (e.g., diff from v1.0.13 to v1.0.16, skipping v1.0.14 and v1.0.15).

### Step 3: Gather All Changes Between Base and HEAD

Run these commands in parallel to gather change data:

```bash
# All commits (including merges) between base and HEAD
git log {base_tag}..HEAD --oneline

# Non-merge commits only (for detailed change analysis)
git log {base_tag}..HEAD --oneline --no-merges

# Merge commits (to extract PR numbers)
git log {base_tag}..HEAD --oneline --grep="Merge pull request"

# Contributors
git log {base_tag}..HEAD --format="%aN" | sort | uniq -c | sort -rn

# Env var changes
git diff {base_tag}..HEAD -- .env.example

# Helm chart changes
git diff {base_tag}..HEAD -- charts/ --stat

# Helm chart dependency changes (breaking change indicator)
git diff {base_tag}..HEAD -- charts/registry/Chart.yaml charts/auth-server/Chart.yaml charts/mcp-gateway-registry-stack/Chart.yaml

# Recently closed issues
gh issue list --state closed --limit 50 --json number,title,closedAt --jq '.[] | "\(.number) | \(.title) | \(.closedAt)"'
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

9. **Contributors**: Unique contributor list from git log.

### Step 5: Write Release Notes

Create the file `release-notes/v{version}.md` following this exact structure:

```markdown
# Release v{version} - {Short Title Summarizing Major Features}

**{Month} {Year}**

---

## Upgrading from v{base_version}

This section covers everything you need to know to upgrade from v{base_version} to v{version}.

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
git checkout v{version}

# Review new env vars in .env.example and update your .env if needed
# Then rebuild and restart:
./build_and_run.sh
```

#### Kubernetes / Helm (EKS)

```bash
cd mcp-gateway-registry
git pull origin main
git checkout v{version}

# {If helm dependency changes: "REQUIRED: Rebuild dependencies"}
cd charts/mcp-gateway-registry-stack
helm dependency build
helm dependency update

# Update values.yaml if needed, then upgrade:
helm upgrade mcp-gateway . -f your-values.yaml
```

{Note: If there are NO Helm chart dependency changes (no Chart.yaml dependency
additions/removals), omit the helm dependency build/update commands and just
show the helm upgrade command.}

#### Terraform / ECS

```bash
cd mcp-gateway-registry
git pull origin main
git checkout v{version}

# Update your .tfvars with any new variables
cd terraform/aws-ecs
terraform plan
terraform apply
```

#### DockerHub Images

Pre-built images are available:

```bash
docker pull mcpgateway/registry:v{version}
docker pull mcpgateway/auth-server:v{version}
docker pull mcpgateway/currenttime-server:v{version}
docker pull mcpgateway/realserverfaketools-server:v{version}
docker pull mcpgateway/mcpgw-server:v{version}
docker pull mcpgateway/fininfo-server:v{version}
docker pull mcpgateway/metrics-service:v{version}
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

{List all contributors from git log, sorted by commit count descending.
Map known email addresses to GitHub usernames where possible.}

---

## Support

- [GitHub Issues](https://github.com/agentic-community/mcp-gateway-registry/issues)
- [GitHub Discussions](https://github.com/agentic-community/mcp-gateway-registry/discussions)
- [Documentation](https://github.com/agentic-community/mcp-gateway-registry/tree/main/docs)

---

**Full Changelog:** [v{base_version}...v{version}](https://github.com/agentic-community/mcp-gateway-registry/compare/v{base_version}...v{version})
```

### Step 6: Present Draft for User Review

After writing the release notes file:

1. Tell the user the file has been created at `release-notes/v{version}.md`
2. Present a brief summary:
   - Number of major features
   - Number of PRs included
   - Number of bug fixes
   - Any breaking changes
   - Contributor count
3. Ask the user to review the file and confirm it looks good, or request changes

### Step 7: Commit, Tag, and Push

Once the user confirms the release notes are ready:

1. **Commit the release notes:**
   ```bash
   git add release-notes/v{version}.md
   git commit -m "docs: Add v{version} release notes"
   ```

2. **Push the commit:**
   ```bash
   git push origin main
   ```

3. **Create or move the git tag** to point at this latest commit (which includes the release notes):
   ```bash
   # If tag already exists, delete it locally and remotely first
   git tag -d v{version} 2>/dev/null || true
   git push origin :refs/tags/v{version} 2>/dev/null || true

   # Create tag on current HEAD
   git tag v{version}

   # Push tag
   git push origin v{version}
   ```

4. **Verify:**
   ```bash
   git log --oneline -1
   git tag -l v{version} --format="%(refname:short) -> %(objectname:short)"
   ```

5. Tell the user the tag is created and pushed, and provide the DockerHub push command:
   ```
   To publish images to DockerHub with this tag:
   make publish-dockerhub-version VERSION=v{version}
   ```

## Important Rules

- **Never skip the user confirmation** for base version in Step 2. The user may want to create release notes that span multiple versions.
- **Never include emojis** in the release notes file. The project CLAUDE.md prohibits emojis in documentation.
- **Never include Claude Code attribution** or "Co-Authored-By" lines in commits.
- **Always use the `release-notes/` directory** at the project root for the output file.
- **Always include upgrade instructions** for all three deployment methods (Docker Compose, Helm/EKS, Terraform/ECS).
- **Always list breaking changes first** in the upgrade section -- this is the most critical information for operators.
- **Always verify Helm Chart.yaml diffs** to detect dependency additions/removals -- these are the most common breaking changes for EKS users.
- **DockerHub image list** should match the components defined in `scripts/publish_containers.sh` in the `COMPONENTS` array. Read this file to get the current list rather than hardcoding.

## Example Usage

```
User: /release-notes v1.0.16