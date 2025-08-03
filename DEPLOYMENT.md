# Deployment Guide - ESG Analysis Platform

## 🌐 Overview

This guide covers deployment options for the ESG Analysis Platform, from local development to production environments.

## 🏠 Local Development

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Git

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd ESG-Analysis-Platform

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Start development environment
docker-compose up -d

# Verify deployment
curl http://localhost:8000/health
```

## 🚀 Production Deployment

### Option 1: Docker Compose (Small-Medium Scale)

#### Production docker-compose.yml
```yaml
version: '3.8'
services:
  esg-api:
    image: esg-analysis:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
    environment:
      - DEBUG=false
      - DATABASE_URL=postgresql://user:pass@postgres:5432/esg_prod
    networks:
      - esg-prod
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - esg-api
```

#### Nginx Configuration
```nginx
upstream esg_api {
    server esg-api:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    location / {
        proxy_pass http://esg_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Option 2: Kubernetes (Enterprise Scale)

#### Namespace
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: esg-analysis
```

#### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: esg-config
  namespace: esg-analysis
data:
  DEBUG: "false"
  VECTOR_DB_PROVIDER: "pinecone"
  CHUNK_SIZE: "1000"
  CHUNK_OVERLAP: "200"
```

#### Secret
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: esg-secrets
  namespace: esg-analysis
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-key>
  PINECONE_API_KEY: <base64-encoded-key>
  DATABASE_URL: <base64-encoded-url>
```

#### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: esg-api
  namespace: esg-analysis
spec:
  replicas: 5
  selector:
    matchLabels:
      app: esg-api
  template:
    metadata:
      labels:
        app: esg-api
    spec:
      containers:
      - name: esg-api
        image: esg-analysis:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: esg-config
        - secretRef:
            name: esg-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

#### Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: esg-api-service
  namespace: esg-analysis
spec:
  selector:
    app: esg-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### Ingress
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: esg-api-ingress
  namespace: esg-analysis
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.esg-platform.com
    secretName: esg-api-tls
  rules:
  - host: api.esg-platform.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: esg-api-service
            port:
              number: 80
```

### Option 3: Cloud Provider Deployment

#### AWS ECS with Fargate
```yaml
# task-definition.json
{
  "family": "esg-analysis",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "esg-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/esg-analysis:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DEBUG",
          "value": "false"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:esg/openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/esg-analysis",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Run
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: esg-analysis
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 80
      containers:
      - image: gcr.io/project/esg-analysis:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: 2000m
            memory: 4Gi
        env:
        - name: DEBUG
          value: "false"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: esg-secrets
              key: openai-key
```

## 🗄️ Database Setup

### PostgreSQL (Production)
```sql
-- Create database
CREATE DATABASE esg_analysis;
CREATE USER esg_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE esg_analysis TO esg_user;

-- Create tables (handled by application migration)
-- Vector extension for PostgreSQL
CREATE EXTENSION IF NOT EXISTS vector;
```

### Database Migration
```bash
# Using Alembic
alembic upgrade head

# Or using application startup
python -m src.core.database migrate
```

## 🔧 Configuration Management

### Environment Variables by Environment

#### Development
```bash
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=sqlite:///./data/dev.db
VECTOR_DB_PROVIDER=chroma
RATE_LIMIT_REQUESTS_PER_MINUTE=1000
```

#### Staging
```bash
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:pass@staging-db:5432/esg_staging
VECTOR_DB_PROVIDER=qdrant
QDRANT_URL=http://qdrant-staging:6333
RATE_LIMIT_REQUESTS_PER_MINUTE=500
```

#### Production
```bash
DEBUG=false
LOG_LEVEL=WARNING
DATABASE_URL=postgresql://user:pass@prod-db:5432/esg_production
VECTOR_DB_PROVIDER=pinecone
PINECONE_API_KEY=prod-key
RATE_LIMIT_REQUESTS_PER_MINUTE=100
ENABLE_RESPONSE_CACHING=true
CACHE_TTL_SECONDS=3600
```

## 📊 Monitoring Setup

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'esg-api'
    static_configs:
      - targets: ['esg-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboard Import
```bash
# Import dashboard
curl -X POST \
  http://admin:admin@grafana:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana/esg-dashboard.json
```

### Log Aggregation (ELK Stack)
```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    
  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
    depends_on:
      - elasticsearch
    
  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
    depends_on:
      - elasticsearch
```

## 🔒 Security Hardening

### SSL/TLS Configuration
```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/private.key \
  -out ssl/certificate.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=your-domain.com"
```

### Security Headers
```nginx
# Nginx security headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header X-Content-Type-Options "nosniff" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
```

### API Rate Limiting
```python
# Application-level rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/query")
@limiter.limit("10/minute")
async def query_rag(request: Request, query: RAGQueryRequest):
    # Implementation
    pass
```

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy ESG Analysis Platform

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio
    - name: Run tests
      run: pytest tests/
    
  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Build Docker image
      run: |
        docker build -t esg-analysis:${{ github.sha }} .
        docker tag esg-analysis:${{ github.sha }} esg-analysis:latest
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push esg-analysis:${{ github.sha }}
        docker push esg-analysis:latest
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to production
      run: |
        # Deployment commands (kubectl, helm, etc.)
        kubectl set image deployment/esg-api esg-api=esg-analysis:${{ github.sha }}
```

## 📈 Performance Optimization

### Database Optimization
```sql
-- Index optimization for PostgreSQL
CREATE INDEX CONCURRENTLY idx_documents_framework ON documents(esg_framework);
CREATE INDEX CONCURRENTLY idx_documents_category ON documents(esg_category);
CREATE INDEX CONCURRENTLY idx_documents_company ON documents(company_id);

-- Vector index for similarity search
CREATE INDEX CONCURRENTLY idx_embeddings_vector ON embeddings USING ivfflat (vector vector_cosine_ops);
```

### Caching Strategy
```python
# Redis caching configuration
import redis
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379, db=0)

def cache_response(ttl=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached = redis_client.get(cache_key)
            
            if cached:
                return json.loads(cached)
            
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator
```

### Load Testing
```bash
# Using Apache Bench
ab -n 1000 -c 10 -H "Content-Type: application/json" \
   -p query.json http://localhost:8000/api/v1/query

# Using Locust
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## 🚨 Backup and Recovery

### Database Backup
```bash
# PostgreSQL backup
pg_dump -h localhost -U esg_user -d esg_analysis > backup_$(date +%Y%m%d_%H%M%S).sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > $BACKUP_DIR/esg_backup_$DATE.sql
gzip $BACKUP_DIR/esg_backup_$DATE.sql

# Cleanup old backups (keep 7 days)
find $BACKUP_DIR -name "esg_backup_*.sql.gz" -mtime +7 -delete
```

### Vector Database Backup
```bash
# ChromaDB backup
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz data/vector_db/

# Pinecone backup (export to JSON)
python scripts/backup_pinecone.py --index esg-analysis --output pinecone_backup.json
```

### Disaster Recovery Plan
1. **RTO (Recovery Time Objective)**: 4 hours
2. **RPO (Recovery Point Objective)**: 1 hour
3. **Backup Frequency**: Daily full, hourly incremental
4. **Recovery Testing**: Monthly
5. **Geographic Distribution**: Multi-region deployment

## 📞 Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
docker stats

# Optimize chunk size and batch processing
export CHUNK_SIZE=500
export BATCH_SIZE=10
```

#### Slow Query Performance
```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM documents WHERE esg_framework = 'CSRD';

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats WHERE tablename = 'documents';
```

#### Vector Search Issues
```python
# Debug vector search
import logging
logging.getLogger("chromadb").setLevel(logging.DEBUG)

# Check embedding dimensions
print(f"Expected: {settings.embedding_dimension}")
print(f"Actual: {len(embedding_vector)}")
```

### Health Check Commands
```bash
# API health
curl -f http://localhost:8000/health || echo "API down"

# Database connectivity
pg_isready -h localhost -p 5432

# Vector database
curl -f http://localhost:6333/health || echo "Qdrant down"

# Redis cache
redis-cli ping
```

## 📋 Maintenance

### Regular Tasks
- **Daily**: Log rotation, backup verification
- **Weekly**: Security updates, performance review
- **Monthly**: Dependency updates, capacity planning
- **Quarterly**: DR testing, security audit

### Update Procedures
```bash
# Rolling update with zero downtime
kubectl set image deployment/esg-api esg-api=esg-analysis:new-version
kubectl rollout status deployment/esg-api

# Rollback if needed
kubectl rollout undo deployment/esg-api
```

---

This deployment guide provides comprehensive coverage for deploying the ESG Analysis Platform across different environments and scales. Choose the appropriate deployment option based on your requirements and infrastructure capabilities.