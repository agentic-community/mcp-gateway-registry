---
name: explainer
description: Explain a GitHub issue or pull request at 100, 200, and 300 level. Verifies every claim against the code, then writes a markdown and a self-contained HTML version into .scratchpad/issue-NNNN/ or .scratchpad/pr-NNNN/ and opens it for preview. Starts no server; offers a command for anyone who wants one. Use when asked to explain, write up, or produce an explainer for an issue or PR.
license: Apache-2.0
metadata:
  author: mcp-gateway-registry
  version: "1.1"
---

# Explainer Skill

Turn a GitHub issue or pull request into an explainer a colleague can read at whatever depth they need. Three levels, one document: 100 for anyone, 200 for how it works, 300 for the code.

Apply the [writing](../writing/SKILL.md) skill to every word of prose. Invoke it, then run its revision pass before you write the file. No emojis and no em-dashes anywhere, per the repo rules.

## Inputs

Accept any of these and resolve to a number plus a kind:

- A full URL: `https://github.com/agentic-community/mcp-gateway-registry/issues/1832`
- A PR URL: `.../pull/1711`
- A bare reference: `#1832`, `1832`, `owner/repo#1832`

A URL containing `/pull/` is a PR. Everything else needs a check, because `gh issue view` succeeds on PRs too. Run `gh pr view <n> --json number` and treat a clean exit as a PR.

Write output to `.scratchpad/issue-NNNN/` or `.scratchpad/pr-NNNN/`. Create it with `mkdir -p`. `.scratchpad/` is gitignored, so nothing you write here can be committed.

## Step 1: gather

For an issue:

```bash
gh issue view <n> --json number,title,state,labels,author,createdAt,body,comments
```

For a PR, get the discussion and the diff:

```bash
gh pr view <n> --json number,title,state,author,createdAt,body,comments,files,additions,deletions,baseRefName,headRefName
gh pr diff <n>
```

Record the commit you verified against, and use it to pin every code link:

```bash
git rev-parse --short HEAD
```

## Step 2: verify before you explain

This step is what separates a useful explainer from a paraphrase. Do not restate the author's claims. Check them.

For every factual claim in the issue or PR body:

1. Open the file it names and confirm the symbol, line, and behavior still exist.
2. Confirm the mechanism really works the way the author says. Read the dependency's own source when the claim is about library behavior, rather than trusting recollection.
3. Look for the claim's siblings. A bug in one call site usually has more; grep the pattern repo-wide and count them.
4. Note where the report is right, and note where it has drifted. Code moves, and an issue filed against an older release often describes something that has since changed. Say so, with the current line reference.
5. Look for the trap in the obvious fix. If the naive patch would introduce a different bug or a security regression, that belongs in the explainer, usually as a callout in the 300 section.

Carry only verified claims into the document. If something matters but you could not confirm it, say that plainly in the open-questions section rather than smoothing it over.

## Step 3: write the three levels

Same content in both files, written once as markdown and then ported to HTML. Levels are cumulative depth, not three summaries of the same thing.

**100, what and why.** Anyone technical can follow it without knowing this codebase. What breaks or what the change does, who it affects, why it is worth attention. No line numbers and no code blocks. Roughly 150 to 250 words.

**200, how it works.** The mechanism and where it lives. Name the components and files, include one diagram, give the sequence of what happens. File references are fine; save line-level detail for 300. Roughly 400 to 600 words.

**300, deep dive.** The code. Line-pinned references, short excerpts of the lines that matter, the traps, the blast radius, what a fix has to preserve, the test surface, and open questions. As long as it needs to be.

End with open questions for the author when the report left anything ambiguous. Three or four at most, each one answerable.

## Step 4: the markdown file

Write `explainer.md` in the output directory. Use `##` for the three level headings so the structure survives a plain markdown reader.

Open with a one-line italic byline. It must carry three links, so a reader landing on the file alone can get to the source, the person, and the code state:

- **The issue or PR**, linked by its number: `[PR #1831](https://github.com/<owner>/<repo>/pull/1831)` or `[issue #1832](https://github.com/<owner>/<repo>/issues/1832)`.
- **The author**, linked to their GitHub profile: `[AUTHOR_LOGIN](https://github.com/AUTHOR_LOGIN)`. Use the `login` from `gh`, not the display name, since the login is what resolves as a profile URL. Never guess a profile URL for a name you did not get from `gh`.
- **The commit you verified against**, linked to its commit page.

For a PR, also give the size (`N added, M deleted, across K files`) and name the base state you checked the before-behavior against. For an issue, name the release it was filed against when the reporter stated one.

```markdown
*Verified against `main` at [8f2bfacb](https://github.com/OWNER/REPO/commit/8f2bfacb) for the pre-change state, and the PR head at `b7fe89d7`. Source: [PR #1831](https://github.com/OWNER/REPO/pull/1831) by [AUTHOR_LOGIN](https://github.com/AUTHOR_LOGIN). 124 added, 6 deleted, across 2 files.*
```

Pull the author login and the PR head from the same `gh` call you already made:

```bash
gh pr view <n> --json author,headRefOid -q '.author.login + " " + .headRefOid[0:8]'
gh issue view <n> --json author -q .author.login
```

## Step 5: the HTML file

Copy the template, then fill it in:

```bash
cp .claude/skills/explainer/assets/template.html <outdir>/explainer.html
```

Replace `{{TITLE}}` (twice, in `<title>` and `<h1>`), `{{BYLINE}}`, `{{BODY}}`, and `{{FOOTER}}`.

`{{BYLINE}}` renders inside a `<p class="byline">`, so pass HTML, not markdown. It carries the same three links as the markdown byline (issue or PR, author profile, verified commit):

```html
Verified against <code>main</code> at <a href="...../commit/8f2bfacb">8f2bfacb</a>. Source:
<a href="...../pull/1831">PR #1831</a> by <a href="https://github.com/AUTHOR_LOGIN">AUTHOR_LOGIN</a>.
``` The template is self-contained by design: all CSS inline, no webfonts, no CDN, no JavaScript. It must render from `file://` with no network. Never add an external reference.

Keep the level structure the template's nav expects:

```html
<h2 id="level-100"><span class="level-badge l100">100</span>What is going on</h2>
<p class="level-intro">One line on who this section is for.</p>
```

Classes the template provides: `.callout` with a `.callout-label` for the one thing a reader must not miss, `.level-badge` with `.l100` / `.l200` / `.l300`, and `figure` / `figcaption` for diagrams. Tables are styled already.

### Diagrams

Hand-author inline SVG. No Mermaid and no diagram library, because both need a CDN and would break offline rendering. Use the template's SVG classes so the diagram follows light and dark mode: `.box` for nodes, `.flow` for normal arrows, `.bad` plus `.bad-text` for the failing path, `.label` for annotations, `.mono` for code-ish text.

Prefer one diagram that earns its place over three that decorate. A request flow with the broken hop marked, or a before-and-after of a code path, is usually the one worth drawing.

```html
<figure>
  <svg viewBox="0 0 640 120" role="img" aria-label="Describe the diagram for screen readers">
    <rect class="box" x="8" y="30" width="150" height="54" rx="6"/>
    <text x="83" y="62" text-anchor="middle">registry pod</text>
    <path class="bad" d="M166 57 H 300" marker-end="url(#arrow-bad)"/>
    <text class="bad-text" x="233" y="46" text-anchor="middle">ConnectTimeout</text>
    <rect class="box" x="308" y="30" width="150" height="54" rx="6"/>
    <text x="383" y="62" text-anchor="middle">upstream</text>
    <defs>
      <marker id="arrow-bad" viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/>
      </marker>
    </defs>
  </svg>
  <figcaption>Caption that says what the reader should take away.</figcaption>
</figure>
```

Give every `svg` a `role="img"` and an `aria-label`. Put each marker in a `defs` block inside the same SVG, since ids must not collide across diagrams on one page.

### Code links

Pin links to the commit you verified against, never to `main`, so line numbers stay accurate:

```
https://github.com/agentic-community/mcp-gateway-registry/blob/<sha>/registry/utils/url_guard.py#L1151-L1155
```

Reference a path in the repo as a link. Reference a path outside the repo, such as a file in `.venv`, as plain `<code>` with no link, and say in the footer that it refers to an installed dependency rather than this repo. State in the footer which commit the line numbers belong to.

## Step 6: validate the HTML

Never hand over HTML you have not parsed. Check structure and self-containment:

```bash
python3 - <<'PY'
from html.parser import HTMLParser
VOID = {"meta","br","hr","img","input","link","source","area","base","col","embed","param","track","wbr",
        "path","rect","circle","line","polygon","polyline","ellipse","use","stop"}
class C(HTMLParser):
    def __init__(self): super().__init__(convert_charrefs=True); self.s=[]; self.e=[]
    def handle_startendtag(self,t,a): pass   # a self-closing tag is balanced by definition
    def handle_starttag(self,t,a):
        if t not in VOID: self.s.append((t,self.getpos()[0]))
    def handle_endtag(self,t):
        if t in VOID: return
        if not self.s: self.e.append(f"line {self.getpos()[0]}: stray </{t}>"); return
        top,ln=self.s.pop()
        if top!=t: self.e.append(f"line {self.getpos()[0]}: </{t}> closes <{top}> from line {ln}")
src=open("OUTDIR/explainer.html",encoding="utf-8").read()
c=C(); c.feed(src)
print("errors:", c.e or "none")
print("unclosed:", c.s or "none")
print("external refs:", [w for w in ("src=","@import","cdn.","fonts.googleapis","http-equiv") if w in src] or "none")
print("placeholders left:", [p for p in ("{{TITLE}}","{{BYLINE}}","{{BODY}}","{{FOOTER}}") if p in src] or "none")
print("em-dash:", "\u2014" in src)  # the banned long dash
PY
```

Fix anything it reports. `external refs`, `placeholders left`, and `em-dash` must all come back clean.

Then run the prose gate on both files. This one is required, not advisory: an explainer is generated prose, so it has no excuse for carrying a tell.

```bash
uv run python scripts/prose-scan.py --strict <outdir>/explainer.md <outdir>/explainer.html
```

`--strict` exits non-zero on any hit. Fix every hit and run it again until it reports `no prose tells found`. Rewrite the sentence rather than reaching for a synonym that slips past the regex; the regex is a net for the habit, not the rule itself.

Do not skip this because you already loaded the writing skill. Loading the skill and skipping its revision pass is exactly how a tell reaches the file: the HTML gate above gets run because it is a command, and the prose pass gets skipped because it is advice. That is why this one is a command too.

## Step 7: hand over, and offer the server as an option

Do not start a server. The HTML is self-contained, so most readers need nothing running, and starting a process the user did not ask for leaves something listening on their machine that they then have to find and stop.

Open the file in the editor so a preview is one click away:

```bash
code -r <outdir>/explainer.html
```

Then give the reader three ways in, cheapest first. The first two need no command at all:

1. The Live Preview button in the editor title bar for the focused HTML file.
2. Right-click the file and Download, then open the local copy. The file has no external references, so this works offline and always.
3. A local HTTP server, only if they want one. Offer the command; do not run it.

### The server command

Offer it as one paste-ready line, rooted at the document's own directory:

```bash
python3 -m http.server 8111 --bind 127.0.0.1 --directory /abs/path/to/.scratchpad/pr-NNNN
```

Then the URL is `http://127.0.0.1:8111/explainer.html`, or drop the filename for a directory listing.

Four things to get right:

- **Use the absolute path**, since the user may paste this from any working directory.
- **Keep `--bind 127.0.0.1`.** Never offer a command that binds `0.0.0.0`.
- **Point `--directory` at the single document's folder, never at `.scratchpad/` itself.** That folder holds credential files (`.hftoken`, `.oai`, `.bedrock`, `.gh-client-id-secret`), and `http.server` serves everything below its root. If several documents need to be reachable at once, copy the generated HTML into a folder of its own and serve that; the files are self-contained, so a copy works.
- **Have the user run it in a VS Code integrated terminal.** That is what makes VS Code forward the port so a browser on their laptop can reach it. A server started any other way listens only on the host and gives `ERR_CONNECTION_REFUSED`.

Say which port you picked and mention nothing is listening until they run it. If a port is already taken the command fails with "Address already in use", so suggest a fresh one rather than reusing a port from an earlier session.

### Formatting the handover

Put the URL on its own line as a bare autolink and each command in its own fenced block:

````markdown
The explainer is at `.scratchpad/pr-1833/explainer.html`. Open it with the Live Preview
button, or download it and open the copy.

If you want it over HTTP instead, paste this into a VS Code integrated terminal:

```bash
python3 -m http.server 8111 --bind 127.0.0.1 --directory /home/ubuntu/repos/mcp-gateway-registry/.scratchpad/pr-1833
```

Then open:

http://127.0.0.1:8111/explainer.html
````

Rules that keep those blocks usable:

- A bare URL on its own line renders as a clickable link in the terminal and in the chat. Do not wrap it in backticks, which kills the link, and do not bury it mid-sentence.
- One command per fenced block, and nothing else in that fence. The copy button takes the whole fence, so a comment line or a second command makes the paste fail or do something unintended.
- Never retype a path or port by hand. Use the real output directory.
- Do not present the URL as if something is already serving it. Say it works once they run the command.

One trap worth repeating if preview misbehaves: pasting a VS Code Live Preview URL into an external browser returns an empty 401 from port 3000, which renders as a blank white page showing the raw URL in the tab title instead of the document title. Live Preview's `openPreviewTarget` needs to be `Embedded Preview`. A blank page means the request reached a server and got an empty body; connection refused means it never arrived. Do not confuse the two.

## Handover

Report where both files landed and the commit you verified against. Lead your message with one line naming the audience you wrote for, so the user can correct it. Say plainly where the issue or PR was wrong or out of date, and list anything you could not verify.

Do not claim a server is running unless you started one because the user asked.
