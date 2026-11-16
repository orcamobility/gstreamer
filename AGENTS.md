# Agent instructions

## NVR TLS trust

Anything loading TLS CA certificates on an NVR, or configuring software that
does, must use the system default CA store, never bundled or compiled-in roots.
Customer TLS-gateway CAs are installed in `/usr/local/share/ca-certificates`
and merged by `update-ca-certificates` into `/etc/ssl/certs/ca-certificates.crt`
and `/etc/ssl/certs/`; containers must use the host-mounted store.
C/GLib: retain the default system GTlsDatabase/OpenSSL trust paths.
Do not replace them with shipped certificate files or compiled-in root lists.
