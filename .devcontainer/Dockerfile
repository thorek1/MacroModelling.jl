FROM julia:latest

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl sudo ca-certificates python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (Codespaces convention)
RUN groupadd -g 1000 vscode \
    && useradd -m -u 1000 -g vscode -s /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode

USER vscode
WORKDIR /home/vscode

# Pre-populate Julia depot with MacroModelling (main branch) deps
RUN git clone --depth 1 https://github.com/thorek1/MacroModelling.jl.git /tmp/MacroModelling \
    && cd /tmp/MacroModelling \
    && julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' \
    && rm -rf /tmp/MacroModelling

# Pre-compile LanguageServer for the Julia VS Code extension
RUN julia -e 'using Pkg; Pkg.add("LanguageServer"); using LanguageServer'
