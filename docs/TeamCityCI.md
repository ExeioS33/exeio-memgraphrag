# TeamCity CI — project skeleton

The repository carries its TeamCity configuration as **versioned settings** in
Kotlin DSL: [`.teamcity/settings.kts`](../.teamcity/settings.kts). TeamCity reads
it from the repository, so the pipeline evolves through pull requests like the
code, and the GitHub Actions workflows under `.github/workflows/` stay the
reference for what each build does.

This is a skeleton. It builds, tests and packages; it does not deploy.

## What the DSL declares

| Build configuration | Runs | Gate |
|---------------------|------|------|
| **Lint** | `uv sync --frozen`, `ruff check .`, `ruff format --check --diff .` | VCS trigger |
| **Tests (Python 3.12)** / **(3.13)** | `./scripts/test.sh tests -q --junitxml=reports/junit.xml` | after Lint |
| **Coverage** | same suite with `--cov=memgraphrag --cov-report=xml` | after Tests 3.12 |
| **Docker image** | `docker build -t exeio-memgraphrag:<version>` (push commented out) | after both Tests |

Every step calls the same commands the `Makefile` and `scripts/test.sh` run
locally; there is no CI-only logic to drift. The offline suite needs no LLM
credential. Integration tests (`--run-integration`) are deliberately outside
the skeleton.

Placeholders, all marked `CHANGE-ME`:

- the uploaded SSH key that clones `git@github.com:ExeioS33/exeio-memgraphrag.git`;
- the Docker registry (`env.DOCKER_REGISTRY`) — the push lines are commented;
- the TeamCity server URL in `.teamcity/pom.xml` (IDE support only).

## Agent prerequisites

An agent that runs these builds needs:

```bash
# Linux agent, run once as the agent user
curl -LsSf https://astral.sh/uv/install.sh | sh          # uv (the DSL also installs it if missing)
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12 3.13                               # interpreters are managed by uv, no system Python needed
docker --version                                          # only for the "Docker image" configuration
```

TeamCity selects agents for **Docker image** on the `docker.server.version`
parameter, which the agent reports automatically when Docker is installed.

## Creating the project — commands

Everything below talks to the TeamCity REST API. Create an access token in
*Profile → Access Tokens* with the *Manage projects* permission, then:

```bash
export TC_URL="https://teamcity.example.internal"     # CHANGE-ME
export TC_TOKEN="…"                                   # never commit it
tc() { curl -fsS -H "Authorization: Bearer $TC_TOKEN" -H "Content-Type: application/json" -H "Accept: application/json" "$@"; }
```

### 1. Project

```bash
tc -X POST "$TC_URL/app/rest/projects" -d '{
  "name": "exeio-memgraphrag",
  "id": "ExeioMemgraphrag",
  "parentProject": { "locator": "_Root" }
}'
```

### 2. SSH key for the GitHub clone

Upload the deploy key through the UI (*Project → SSH Keys → Upload SSH key*),
with the name you put in `settings.kts` (`uploadedKey = "…"`). Register the
public half as a read-only deploy key on the GitHub repository.

### 3. VCS root that holds the settings

```bash
tc -X POST "$TC_URL/app/rest/vcs-roots" -d '{
  "name": "exeio-memgraphrag settings",
  "id": "ExeioMemgraphrag_Settings",
  "vcsName": "jetbrains.git",
  "project": { "id": "ExeioMemgraphrag" },
  "properties": { "property": [
    { "name": "url",            "value": "git@github.com:ExeioS33/exeio-memgraphrag.git" },
    { "name": "branch",         "value": "refs/heads/main" },
    { "name": "authMethod",     "value": "TEAMCITY_SSH_KEY" },
    { "name": "teamcitySshKey", "value": "CHANGE-ME-teamcity-memgraphrag-deploy-key" }
  ] }
}'
```

### 4. Point versioned settings at `.teamcity/`

```bash
tc -X PUT "$TC_URL/app/rest/projects/ExeioMemgraphrag/versionedSettings/config" -d '{
  "synchronizationMode": "enabled",
  "format": "kotlin",
  "allowUIEditing": false,
  "storeSecureValuesOutsideVcs": true,
  "buildSettingsMode": "useFromVCS",
  "showSettingsChanges": true,
  "vcsRoot": { "id": "ExeioMemgraphrag_Settings" }
}'
tc -X POST "$TC_URL/app/rest/projects/ExeioMemgraphrag/versionedSettings/loadingFromVCS"   # pull settings now
```

If your server predates the `versionedSettings` REST endpoints (2023.05), do the
same in *Project → Versioned Settings*: *Synchronization enabled*, format
*Kotlin*, *use settings from VCS*, and the VCS root from step 3.

### 5. Check what TeamCity generated

```bash
tc "$TC_URL/app/rest/buildTypes?locator=affectedProject:(id:ExeioMemgraphrag)" | python3 -m json.tool
```

Expected ids: `ExeioMemgraphrag_Lint`, `ExeioMemgraphrag_Tests312`,
`ExeioMemgraphrag_Tests313`, `ExeioMemgraphrag_Coverage`,
`ExeioMemgraphrag_DockerImage`.

### 6. Run a build and read the result

```bash
tc -X POST "$TC_URL/app/rest/buildQueue" -d '{ "buildType": { "id": "ExeioMemgraphrag_Tests312" } }'
tc "$TC_URL/app/rest/builds?locator=buildType:ExeioMemgraphrag_Tests312,count:1" | python3 -m json.tool
```

## GitHub status checks (next iteration)

Not in the skeleton. To report build status on pull requests, add a
`commitStatusPublisher` feature to each build type with a GitHub token stored as
a TeamCity secure parameter (`credentialsJSON:…`), and a `pullRequests` feature on
the VCS root so `refs/pull/*/head` branches are picked up. Both are one block each
in the DSL; they need a token, which is why they are left for the next pass.

## Keeping the two CIs in step

`.github/workflows/lint.yml` and `tests.yml` are the source of truth for the
commands. When a workflow step changes, mirror it in the matching `script { }`
block of `settings.kts` in the same pull request.
