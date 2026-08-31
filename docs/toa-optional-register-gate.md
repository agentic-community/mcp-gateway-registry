# Optional TOA verify for registry diligence

The MCP Gateway & Registry centralizes discovery, OAuth, and audit for MCP
servers and other AI assets. Registry diligence answers governed inventory. It
does not prove tool delivery quality from an outside probe.

[TOA](https://github.com/Carmel-Labs-Inc/toa) (`toa/0.1`) is adjacent signed
delivery evidence. Optional offline verify before approving a new registry
entry or enabling a route.

```yaml
      - name: Verify tool delivery attestation
        if: hashFiles('toa.json') != ''
        run: |
          pip install "git+https://github.com/Carmel-Labs-Inc/toa.git@345f24607919b5bdf143719b9ea062543cdfe88e#subdirectory=python"
          toa-verify toa.json --require-layer functional=pass
```

Example: [`examples/toa-after-register.yml`](../examples/toa-after-register.yml).

Not a wire protocol. Not per-call. No AgentStatus account required to verify.
