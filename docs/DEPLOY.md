# Deploying VeriRAG

> **Live:** not deployed yet · **Host:** Hugging Face Spaces (Docker, CPU) · **Auto-deploy:** on push to the Space repo

For local development, see [SETUP.md](SETUP.md).

---

## 1. Architecture in production

| Piece | Runs on | Notes |
|---|---|---|
| Streamlit app, retrieval, verifier | Hugging Face Space (Docker, free CPU) | One container, port 7860 |
| LLM | Groq API (free tier) | `LLM_PROVIDER=groq` is set in the Dockerfile |
| Index | Inside the container (`data/index/`) | Committed, or built on the first query |

---

## 2. Environment variables

Set these as Space **secrets/variables** (Settings → Variables and secrets). Never commit them. Names are in [`.env.example`](../.env.example).

| Variable | Required | Where it's set |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Space secret |
| `GROQ_MODEL` | ❌ | Space variable (default `llama-3.1-8b-instant`) |
| `VERIRAG_VERIFIER` | ❌ | Space variable (`minicheck` or `nli`) |

> After changing a variable, restart the Space. Running containers don't pick up new values.

---

## 3. First-time deploy

### Docker on Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) and choose the **Docker** SDK (blank template).
2. Spaces read their config from YAML front matter in the Space's `README.md`. Add this to the top of the README in the Space repo:

   ```yaml
   ---
   title: VeriRAG
   emoji: 🔎
   colorFrom: green
   colorTo: blue
   sdk: docker
   app_port: 7860
   ---
   ```

3. Add `GROQ_API_KEY` as a secret.
4. Push this repo to the Space:

   ```bash
   git remote add space https://huggingface.co/spaces/<hf-user>/VeriRAG
   git push space main
   ```

5. The Space builds the image and starts Streamlit. The first query downloads the models.

### Docker (self-host)

```bash
docker build -t verirag .
docker run -d -p 7860:7860 --env-file .env verirag
```

---

## 4. Index

There is no database and there are no migrations. Rebuild `data/index/` with `python scripts/build_index.py` whenever `data/raw/` changes, then commit it. If the index is missing, the app builds it on the first query.

---

## 5. Custom domain

Not configured. The toolkit's subdomain script supports Vercel and Render, not Hugging Face Spaces.

---

## 6. Verify a deploy

- [ ] The Space page loads with no errors in the Logs tab
- [ ] "When was Python 3.0 released?" returns a cited answer with green-underlined claims
- [ ] A question the sources don't cover (e.g. "What is the population of Paris?") shows the abstention notice
- [ ] The timings expander shows `groq/...` as the LLM

---

## 7. Rollback

| Host | How |
|---|---|
| Hugging Face Spaces | `git revert <bad-commit>` and push to the Space remote. The Space rebuilds from the new commit |

After rolling back, note it in a commit or PR so the next push doesn't silently re-ship the broken version.

---

## 8. Monitoring & logs

- **Logs:** Space → Logs (build and container logs)
- **Errors:** none yet
- **Uptime:** none yet
- **Dependencies:** Dependabot opens weekly update PRs and a maintenance report is emailed every week (see `.github/workflows/`)

---

## Known production limitations

- Free Spaces run on CPU, so verification takes several seconds per answer.
- Free Spaces go to sleep after a period of inactivity, and the first request afterwards waits for the container and models to load.
- The Groq free tier is rate-limited, so heavy use can return HTTP errors. These are shown in the UI.

---

<div align="center">

<h3>Akshat Gupta</h3>

<a href="https://akshatgcodes.in"><img src="https://img.shields.io/badge/Website-akshatgcodes.in-111111?style=flat-square&logo=googlechrome&logoColor=white" alt="Website"></a>
<a href="https://linkedin.com/in/akshatgcodes"><img src="https://img.shields.io/badge/LinkedIn-akshatgcodes-0A66C2?style=flat-square" alt="LinkedIn"></a>
<a href="https://github.com/akshatgcodes"><img src="https://img.shields.io/badge/GitHub-akshatgcodes-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>
<a href="mailto:akshatgcodes@gmail.com"><img src="https://img.shields.io/badge/Email-akshatgcodes%40gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email"></a>

</div>
