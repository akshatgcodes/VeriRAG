# Contributing to VeriRAG

Thanks for taking the time to contribute! Bug reports, fixes, docs improvements and feature ideas are all welcome.

---

## Before you start

- **Found a bug?** Open a [bug report](../../issues/new?template=bug_report.yml). Search existing issues first.
- **Have an idea?** Open a [feature request](../../issues/new?template=feature_request.yml) before writing code, so we can agree on the approach.
- **Found a security issue?** Don't open a public issue. Follow [SECURITY.md](SECURITY.md).
- Small fixes (typos, docs, obvious bugs) can go straight to a pull request.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Local setup

Full instructions are in [docs/SETUP.md](docs/SETUP.md). Short version:

```bash
git clone https://github.com/<your-username>/VeriRAG   # your fork
cd VeriRAG
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/build_index.py
streamlit run app/streamlit_app.py
```

---

## Workflow

1. **Fork** the repo and create a branch from `main`:

   | Branch prefix | Use for |
   |---|---|
   | `feat/` | new feature, e.g. `feat/ragtruth-eval` |
   | `fix/` | bug fix, e.g. `fix/citation-parsing` |
   | `docs/` | documentation only |
   | `chore/` | tooling, config, dependencies |
   | `refactor/` | code change with no behaviour change |

2. **Make your change.** Keep it focused, one feature or fix per PR.
3. **Check it locally** before pushing:

   ```bash
   python -m compileall -q -x '(\.venv|venv|\.git)/' . && pytest -q
   ```

4. **Open a pull request** against `main`. The PR template will guide you.

---

## Commit messages

Commits follow an emoji-prefixed format:

```
<emoji> [#issue] [scope:] <message in present tense, lowercase>
```

| Emoji | Type | Example |
|---|---|---|
| ✨ | New feature | `✨ #12 add ragtruth evaluation` |
| 🐛 | Bug fix | `🐛 claims: fix citations after the full stop` |
| 🎊 | Improvement | `🎊 batch verifier calls across sources` |
| ♻️ | Refactor | `♻️ extract labelling into verify.py` |
| 📚 | Docs | `📚 add deploy guide` |
| 🧪 | Tests | `🧪 cover abstention handling` |
| 🔨 | Chore / config | `🔨 bump python to 3.12 in ci` |
| 👆 | Dependency update | `👆 upgrade sentence-transformers` |
| 🔒 | Security | `🔒 escape llm output before rendering` |
| 💀 | Code deletion | `💀 remove unused retriever option` |

Breaking changes add a line `🚨 BREAKING CHANGE: <what breaks>` in the commit body.

---

## Code style

- Follow the existing file structure and patterns. Each pipeline stage lives in its own module under `verirag/`.
- Python: `snake_case` for files, functions and variables. `PascalCase` for classes.
- Keep heavy imports (`torch`, `faiss`, `transformers`) inside the functions or classes that need them, so the unit tests run without model downloads.
- New settings go in `verirag/config.py` with an env-var override, and in `.env.example`.
- No secrets in code. Use env vars and add new ones to `.env.example`.
- Update docs when behaviour, setup or env vars change.

---

## Pull request checklist

- [ ] Branch is up to date with `main`
- [ ] Compile check and tests pass locally
- [ ] Commits follow the format above
- [ ] Docs / `.env.example` / `CHANGELOG.md` updated if needed
- [ ] PR description explains **what** and **why**

PRs are reviewed as soon as possible. Feedback is about the code, never the person.

---

<div align="center">

<h3>Akshat Gupta</h3>

<a href="https://akshatgcodes.in"><img src="https://img.shields.io/badge/Website-akshatgcodes.in-111111?style=flat-square&logo=googlechrome&logoColor=white" alt="Website"></a>
<a href="https://linkedin.com/in/akshatgcodes"><img src="https://img.shields.io/badge/LinkedIn-akshatgcodes-0A66C2?style=flat-square" alt="LinkedIn"></a>
<a href="https://github.com/akshatgcodes"><img src="https://img.shields.io/badge/GitHub-akshatgcodes-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>
<a href="mailto:akshatgcodes@gmail.com"><img src="https://img.shields.io/badge/Email-akshatgcodes%40gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email"></a>

</div>
