# Security policy

dpmread reads image files from the local disk and prints or exports the
decoded text. It has no network service, no plug-in loading and no update
mechanism. The relevant attack surface is untrusted input: image files and
the data encoded in a symbol.

## Reporting a vulnerability

Please report security issues privately through GitHub:
**Security → Report a vulnerability** on the repository page
(<https://github.com/aharjajarvi/dpmread/security/advisories/new>).
Do not open a public issue for them.

Include the version (`--version` output or the release name), the OS, and
an input file that reproduces the problem if you can share it.

Things worth reporting, for example:

- a crash, hang or unbounded memory use on a crafted image file;
- decoded text that can inject content into the CSV / JSON export or the
  terminal (control characters, formula prefixes) in a way the output does
  not escape;
- a problem in a bundled third-party library (zxing-cpp, image decoders)
  that dpmread exposes.

Reports are answered on a best-effort basis; this is a small project
without a security team. Fixes go into the next release, and the advisory
is published once a fixed release is available.

## Supported versions

Only the latest release receives fixes.
