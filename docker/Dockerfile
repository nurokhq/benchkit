# One shared environment for every condition of every benchmark.
#
# The image supplies tooling only — Ubuntu, Claude Code, the `nurok` command-line
# client, and utilities an agent may reasonably reach for. Which condition a run
# represents comes entirely from the arguments passed at `docker run`: the tool list,
# the system prompt, and for a corpus condition an `--mcp-config` pointing at a
# knowledge-base endpoint. The same image therefore backs all three conditions, which
# keeps their environments identical and leaves the corpus as the only difference.
#
# The container is also the isolation boundary: only the task's scratch directory is
# mounted, so the benchmark's questions, its answer key and its run manifest are simply
# not present in the filesystem the agent can see.
#
# Build:
#   docker build -t benchkit-agent -f docker/Dockerfile .
#
# The harness issues this once per task:
#   docker run --rm -i --network host \
#     -v <scratch>:/work -w /work \
#     --tmpfs /home/agent/.claude:rw,exec \
#     -e ANTHROPIC_API_KEY \
#     benchkit-agent claude --print --output-format json ...
#
# `ANTHROPIC_API_KEY` is named without a value on purpose: the key comes from the
# docker client's own environment, so it never appears in an argv another local user
# can read.
FROM ubuntu:noble

# Which apt mirror to build against. HTTP, not HTTPS: this layer runs before
# ca-certificates is installed, so TLS cannot be verified yet. Ubuntu 24.04 uses the
# deb822 format, so /etc/apt/sources.list.d/ubuntu.sources is replaced rather than
# /etc/apt/sources.list. One host serves every suite including noble-security, which
# keeps the build to a single connection pool.
#
# The default is the official archive. A closer mirror is worth setting on a slow
# link, e.g. --build-arg APT_MIRROR=http://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG APT_MIRROR=http://archive.ubuntu.com/ubuntu
RUN printf '%s\n' \
      'Types: deb' \
      'URIs: '"${APT_MIRROR}" \
      'Suites: noble noble-updates noble-backports noble-security' \
      'Components: main restricted universe multiverse' \
      'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
      > /etc/apt/sources.list.d/ubuntu.sources \
 && rm -f /etc/apt/sources.list.d/*.list /etc/apt/sources.list \
 && printf 'Acquire::Retries "5";\nAcquire::http::Timeout "20";\nAcquire::https::Timeout "20";\n' \
      > /etc/apt/apt.conf.d/99-network

# Needed at run time as well as build time: an agent may reasonably reach for git,
# rg, jq or python while working a question.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      jq \
      zstd \
      ripgrep \
      python3 \
      coreutils \
      findutils \
      less \
      procps \
 && rm -rf /var/lib/apt/lists/*

ENV HOME=/home/agent
ENV PATH=/home/agent/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
WORKDIR /home/agent

# Anthropic's own installer resolves the platform build and verifies it. It installs
# under $HOME/.local/bin, so the directory has to exist first. Pin a release with
# --build-arg CLAUDE_VERSION=2.1.268 when a reproducible image matters more than
# tracking the current one.
ARG CLAUDE_VERSION=stable
RUN mkdir -p /home/agent/.local/bin \
 && curl -fsSL https://claude.ai/install.sh | bash -s -- "${CLAUDE_VERSION}" \
 && claude --version

# The `nurok` CLI, for the condition that reads the corpus from disk rather than over
# MCP. Same pattern again: the vendor installer resolves the platform build and verifies
# a signed manifest. It installs under $NUROK_HOME/bin, which has to be on PATH for the
# agent to reach it.
#
# **The version is pinned, and has to be.** Two reasons, both of which produced a
# silently wrong image before this line existed:
#
#   * Docker caches a RUN layer by its command string. While this ARG was empty, the
#     command never changed, so a rebuild reused the cached layer and kept whatever CLI
#     it downloaded the first time -- across releases, with no warning. The ARG's value
#     is part of the layer hash, so a pin is also the cache key.
#   * The corpus conditions need a client that can reach the hosted API. Releases before
#     0.4.3 send no `User-Agent`, and the edge in front of `nurok.ai` refuses those
#     requests with 403 regardless of credentials. A `kb-cli` run then has no corpus to
#     read and no error to show for it, because the agent falls back to the open web.
#
# Override with --build-arg NUROK_VERSION=<x.y.z>; do not pass an empty one.
#
# This is the only benchmark-specific tool in the image. It is here because an image is
# the environment a benchmark runs in and building one per benchmark would make two
# conditions' environments differ in ways nobody would notice. A benchmark that needs a
# different client should add it the same way, or bring its own image with --image.
ARG NUROK_VERSION=0.4.3
ENV NUROK_HOME=/home/agent/.nurok
ENV PATH=/home/agent/.nurok/bin:/home/agent/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN export NUROK_VERSION \
 && curl -fsSL https://static.nurok.ai/cli/install.sh | bash \
 && nurok --version

# Scratch mount point, so `docker run` can mount the task directory here without the
# image knowing where it lives on the host. The Claude config directory is mounted as
# a tmpfs at run time, so every task starts with an empty config.
WORKDIR /work

ENTRYPOINT ["claude"]
CMD ["--help"]
