# LAFA Frontend Multi-Stage Docker Build
# Stage 1: Generate JSON data from TSV files
# Stage 2: Build React frontend
# Stage 3: Serve with Nginx

# ============================================
# Stage 1: Data Generation
# ============================================
FROM python:3.11-slim AS data

WORKDIR /app

# Install minimal Python dependencies
RUN pip install --no-cache-dir pandas

# Copy application code and data
COPY ../app/ app/
COPY ../data/ data/
COPY scripts/ frontend/scripts/

# Create output directory and generate JSON data
RUN mkdir -p frontend/public/data && \
    python frontend/scripts/generate_data.py

# ============================================
# Stage 2: Build Frontend
# ============================================
FROM node:20-slim AS build

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# Copy frontend source
COPY . ./

# Copy generated data from previous stage
COPY --from=data /app/frontend/public/data ./public/data

# Build the application
RUN npm run build

# ============================================
# Stage 3: Serve with Nginx
# ============================================
FROM nginx:alpine

# Copy built frontend
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget -q --spider http://localhost:8501/_health || exit 1

# Run nginx
CMD ["nginx", "-g", "daemon off;"]
