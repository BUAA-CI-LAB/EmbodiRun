# Security Policy

EmbodiRun is a deployment and execution runtime: a deployment holds
credentials, opens network services, and can move physical hardware. Security
reports are taken seriously, and this page explains how to send one.

## Reporting a vulnerability

**Do not open a public issue, pull request, or discussion for a security
problem.**

Report it privately through either channel:

1. GitHub private vulnerability reporting, if it is enabled for this
   repository: **Security** tab → **Report a vulnerability**.
2. Email **cclonelycc@outlook.com** with `[EmbodiRun security]` in the subject.

Include, as far as you can:

- the affected revision or release, and the component
  (`client`, `deployment`, `application`, `devices`, `model_services`,
  `robots`, `bindings`, `simulators`, or the HTTP/WirelessComm transport);
- a description of the impact and the conditions required to trigger it;
- a minimal reproduction, or the exact command, configuration shape, and
  observed result;
- whether the issue is already public anywhere.

Redact tokens, credentials, addresses, and personal data from anything you
send. Never test against systems, robots, or networks you do not own or have
explicit permission to test.

## What to expect

This is a research project maintained on a best-effort basis, so these are
targets rather than guarantees, measured from the first private report:

| Step | Target |
|---|---|
| Acknowledgement of the report | within 5 working days |
| Initial assessment and severity triage | within 10 working days |
| Fix or documented mitigation for confirmed issues | agreed with the reporter, based on severity |
| Public disclosure | coordinated with the reporter after a fix or mitigation is available |

We will credit reporters in the advisory unless you ask us not to.

## In scope

- Credential, token, or secret handling: leakage in logs, recordings, error
  responses, process arguments, or environment dumps; authentication or
  authorisation bypass in the HTTP or WirelessComm interfaces.
- Remote exposure: a service that binds beyond its documented interface, an
  unauthenticated endpoint that changes deployment or device state, or a
  TLS/transport configuration that silently weakens protection.
- Safety-relevant behaviour: a path that moves a device without going through
  bounded execution, a `stop` or `cancel` that reports success while motion
  continues, or action validation that can be bypassed.
- Code execution: injection through configuration, deserialisation, or a
  dependency that a deployment loads from an untrusted source.

## Out of scope

- The accuracy, safety, or licensing of model checkpoints, datasets, robot
  SDKs, and simulators. These are not distributed here; report them upstream.
- Vulnerabilities in third-party dependencies with no EmbodiRun-specific
  impact — report them to the upstream project, though we welcome a heads-up.
- Denial of service through resource exhaustion on a deployment you control,
  and reports produced only by a scanner without a demonstrated impact.
- Physical damage from deliberately driving hardware outside its documented
  limits. EmbodiRun's safety semantics are described in
  [`docs/en/safety.md`](docs/en/safety.md); behaviour that contradicts them is in
  scope.

## Supported versions

Security fixes are applied to the latest release on `main`. There are no
long-term support branches, so please confirm an issue reproduces on the
current `main` before reporting it.
