# Security policy

## Reporting a vulnerability

Do not open a public issue for suspected vulnerabilities or exposed credentials. Contact the repository owner privately and include reproduction steps, affected versions, and impact. Add a dedicated security contact before publishing the repository.

## Supported version

Security fixes are applied to the latest `3.x` release line.

## Deployment guidance

- Administrative routes are disabled unless `ADMIN_TOKEN` is configured.
- Use a randomly generated token of at least 24 characters in production and rotate it through a secrets manager.
- Restrict CORS and trusted hosts. Never use wildcard origins with credentialed requests.
- Treat uploaded ESG documents and generated reports as potentially confidential.
- Run containers as non-root, terminate TLS, constrain network access, and keep provider dependencies updated.
- Review logs and traces before enabling LangSmith; prompts and retrieved context can contain private document text.
