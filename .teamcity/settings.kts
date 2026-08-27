import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.perfmon
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.vcs
import jetbrains.buildServer.configs.kotlin.vcs.GitVcsRoot

/*
 * TeamCity Kotlin DSL for exeio-memgraphrag — SKELETON.
 *
 * Mirrors .github/workflows/: lint (ruff), offline tests on 3.12 and 3.13,
 * coverage, then the Docker image. Every build runs the same commands a
 * developer runs locally (see Makefile / scripts/test.sh), through uv.
 *
 * Placeholders to replace before the first import (grep for CHANGE-ME):
 *   - the uploaded SSH key name used to clone the GitHub repository
 *   - the Docker registry in DockerImage
 * Setup commands: docs/TeamCityCI.md
 */

version = "2024.12"

project {
    description = "MemGraphRAG API server — lint, tests, coverage, image"

    vcsRoot(MemGraphRagGit)

    buildType(Lint)
    buildType(Tests312)
    buildType(Tests313)
    buildType(Coverage)
    buildType(DockerImage)

    buildTypesOrder = arrayListOf(Lint, Tests312, Tests313, Coverage, DockerImage)

    params {
        // Offline tests need no provider key; integration tests are gated behind
        // --run-integration and are not part of this skeleton.
        param("env.UV_LINK_MODE", "copy")
        param("env.UV_PYTHON_PREFERENCE", "only-managed")
    }
}

object MemGraphRagGit : GitVcsRoot({
    name = "exeio-memgraphrag (GitHub)"
    url = "git@github.com:ExeioS33/exeio-memgraphrag.git"
    branch = "refs/heads/main"
    branchSpec = """
        +:refs/heads/*
        +:refs/pull/*/head
    """.trimIndent()
    authMethod = uploadedKey {
        uploadedKey = "CHANGE-ME-teamcity-memgraphrag-deploy-key"
    }
})

/** Installs uv on the agent if absent and syncs the locked environment. */
fun BuildSteps.uvSync(python: String) {
    script {
        name = "uv sync ($python)"
        scriptContent = """
            set -euo pipefail
            if ! command -v uv >/dev/null 2>&1; then
              curl -LsSf https://astral.sh/uv/install.sh | sh
              export PATH="${'$'}HOME/.local/bin:${'$'}PATH"
            fi
            uv python install $python
            uv sync --frozen --python $python --extra api --extra pytest --extra client
        """.trimIndent()
    }
}

object Lint : BuildType({
    name = "Lint"
    description = "ruff check + ruff format --check"

    vcs { root(MemGraphRagGit) }

    steps {
        uvSync("3.12")
        script {
            name = "ruff"
            scriptContent = """
                set -euo pipefail
                export PATH="${'$'}HOME/.local/bin:${'$'}PATH"
                uv run --no-sync ruff check .
                uv run --no-sync ruff format --check --diff .
            """.trimIndent()
        }
    }

    triggers { vcs { } }
    features { perfmon { } }
})

/** The offline suite on one interpreter, as scripts/test.sh runs it. */
fun testsOn(python: String): BuildType.() -> Unit = {
    name = "Tests (Python $python)"

    vcs { root(MemGraphRagGit) }

    steps {
        uvSync(python)
        script {
            name = "pytest"
            scriptContent = """
                set -euo pipefail
                export PATH="${'$'}HOME/.local/bin:${'$'}PATH"
                uv run --no-sync ./scripts/test.sh tests -q --tb=short --junitxml=reports/junit.xml
            """.trimIndent()
        }
    }

    artifactRules = "reports/** => reports"

    features {
        perfmon { }
        // Publishes reports/junit.xml as TeamCity test results.
        feature {
            type = "xml-report-plugin"
            param("xmlReportParsing.reportType", "junit")
            param("xmlReportParsing.reportDirs", "reports/junit.xml")
        }
    }

    triggers { vcs { } }
    dependencies { snapshot(Lint) { onDependencyFailure = FailureAction.CANCEL } }
}

object Tests312 : BuildType(testsOn("3.12"))
object Tests313 : BuildType(testsOn("3.13"))

object Coverage : BuildType({
    name = "Coverage"

    vcs { root(MemGraphRagGit) }

    steps {
        uvSync("3.12")
        script {
            name = "pytest --cov"
            scriptContent = """
                set -euo pipefail
                export PATH="${'$'}HOME/.local/bin:${'$'}PATH"
                uv run --no-sync ./scripts/test.sh tests -q --cov=memgraphrag --cov-report=xml:reports/coverage.xml --cov-report=term
            """.trimIndent()
        }
    }

    artifactRules = "reports/** => reports"
    dependencies { snapshot(Tests312) { onDependencyFailure = FailureAction.CANCEL } }
})

object DockerImage : BuildType({
    name = "Docker image"
    description = "Builds exeio-memgraphrag:<version> exactly as docker-compose does"

    vcs { root(MemGraphRagGit) }

    params {
        param("env.MEMGRAPHRAG_VERSION", "0.1.0")
        param("env.DOCKER_REGISTRY", "CHANGE-ME-registry.internal/exeio")
    }

    steps {
        script {
            name = "docker build"
            scriptContent = """
                set -euo pipefail
                docker build -t exeio-memgraphrag:${'$'}MEMGRAPHRAG_VERSION -t exeio-memgraphrag:latest .
                # docker tag exeio-memgraphrag:${'$'}MEMGRAPHRAG_VERSION ${'$'}DOCKER_REGISTRY/exeio-memgraphrag:${'$'}MEMGRAPHRAG_VERSION
                # docker push ${'$'}DOCKER_REGISTRY/exeio-memgraphrag:${'$'}MEMGRAPHRAG_VERSION
            """.trimIndent()
        }
    }

    requirements { exists("docker.server.version") }
    dependencies {
        snapshot(Tests312) { onDependencyFailure = FailureAction.CANCEL }
        snapshot(Tests313) { onDependencyFailure = FailureAction.CANCEL }
    }
})
