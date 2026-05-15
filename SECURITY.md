# Security Policy

## 🔒 Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## 🚨 Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability, please send an email to **visionforge02@gmail.com** with the following information:

### What to Include

- **Type of vulnerability** (e.g., XSS, SQL injection, remote code execution)
- **Full paths** of source file(s) related to the vulnerability
- **Location** of the affected source code (tag/branch/commit or direct URL)
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact** of the vulnerability and potential attack scenarios
- **Suggested fix** (if you have one)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Response**: Within 5 business days
- **Status Updates**: Every 7 days until resolved
- **Resolution Timeline**: Varies by severity
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 90 days

### Our Commitment

- We will confirm receipt of your vulnerability report
- We will send you regular updates about our progress
- We will credit you in the security advisory (unless you prefer to remain anonymous)
- We will notify you when the vulnerability is fixed

## 🛡️ Security Best Practices

### For Users

1. **Keep Dependencies Updated**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Use Environment Variables for Secrets**
   - Never hardcode API keys, passwords, or tokens
   - Use `.env` files (excluded from git)
   - Reference the `.env.example` template

3. **Container Security**
   ```bash
   # Run container as non-root user
   docker run --user mluser ...
   
   # Limit resources
   docker run --memory="4g" --cpus="2" ...
   ```

4. **Network Security**
   - Use HTTPS in production
   - Enable firewall rules
   - Limit exposed ports

5. **Input Validation**
   - Validate all uploaded files
   - Check file sizes and types
   - Sanitize file names

### For Developers

1. **Dependency Scanning**
   ```bash
   # Install safety
   pip install safety
   
   # Check for known vulnerabilities
   safety check
   ```

2. **Code Scanning**
   ```bash
   # Run bandit for security issues
   pip install bandit
   bandit -r core/ utils/
   ```

3. **Secret Scanning**
   - Use `git-secrets` or `truffleHog`
   - Enable GitHub secret scanning
   - Never commit `.env` files

4. **Container Scanning**
   ```bash
   # Scan Docker images with Trivy
   trivy image mlplatform:latest
   ```

5. **Secure Coding Guidelines**
   - Always validate and sanitize user input
   - Use parameterized queries (if using databases)
   - Implement proper authentication and authorization
   - Use secure random number generation
   - Keep dependencies minimal and updated

## 🔐 Known Security Considerations

### File Upload Security

The application accepts file uploads. Current security measures:

- **File type validation**: Only image formats allowed
- **File size limits**: Configurable max size
- **Filename sanitization**: Special characters removed
- **Isolated storage**: Uploads stored in dedicated directory
- **Non-executable permissions**: Files cannot be executed

**Recommendation**: Run the application in a sandboxed environment.

### Model Loading

Loading pre-trained models can be risky:

- **Trust issue**: Only load models from trusted sources
- **Pickle vulnerability**: PyTorch models use pickle (potential code execution)
- **Mitigation**: Running as non-root user limits impact

**Recommendation**: Validate model checksums before loading.

### API Exposure

If exposing the API publicly:

- **Rate limiting**: Implement to prevent abuse
- **Authentication**: Add API key or OAuth
- **CORS**: Configure appropriately
- **Input validation**: Strictly validate all inputs

### Dependencies

This project uses multiple ML frameworks:

- **Regular updates**: Check for security advisories
- **Minimal installation**: Only install required frameworks
- **Version pinning**: Use exact versions in production

## 🚀 Production Deployment Security

### Environment Configuration

```bash
# Production environment variables
ENVIRONMENT=production
DEBUG=false
SECURE_COOKIES=true
ENABLE_XSRF_PROTECTION=true
MAX_UPLOAD_SIZE=104857600  # 100MB
RATE_LIMIT_ENABLED=true
```

### Docker Security

```dockerfile
# Run as non-root user
USER mluser

# Read-only root filesystem (when possible)
docker run --read-only ...

# Drop capabilities
docker run --cap-drop=ALL ...

# Use security options
docker run --security-opt=no-new-privileges ...
```

### Network Security

```yaml
# docker-compose.yml
services:
  app:
    networks:
      - internal
    expose:
      - "8501"  # Don't publish directly
  
  nginx:
    networks:
      - internal
      - external
    ports:
      - "443:443"
```

### HTTPS Configuration

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
}
```

## 📋 Security Checklist

Before deploying to production:

- [ ] All dependencies updated and scanned
- [ ] Security headers configured (CSP, HSTS, etc.)
- [ ] HTTPS enabled with valid certificate
- [ ] Authentication and authorization implemented
- [ ] Rate limiting configured
- [ ] Input validation on all endpoints
- [ ] File upload restrictions enforced
- [ ] Logging and monitoring enabled
- [ ] Security scanning in CI/CD pipeline
- [ ] Secrets stored securely (not in code)
- [ ] Container running as non-root user
- [ ] Regular security audits scheduled

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)

## 🙏 Acknowledgments

We thank the following security researchers who have helped improve this project:

- (Your name could be here!)

---

**Last Updated**: May 13, 2026
