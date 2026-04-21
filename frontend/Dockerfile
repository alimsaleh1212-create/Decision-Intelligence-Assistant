# Stage 1: build the React app
# Node Alpine keeps the builder layer small.
FROM node:20-alpine AS builder

WORKDIR /build

# Copy manifests first — Docker caches the npm install layer separately
# from source code so rebuilds are fast when only source changes.
COPY package.json package-lock.json* ./

RUN npm ci --silent

# Copy source and build
COPY . .

# VITE_API_BASE_URL is injected at build time via docker-compose env
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build

# Stage 2: serve with nginx
# nginx:alpine is ~25 MB — much smaller than a full Node image.
FROM nginx:1.27-alpine AS runtime

# Replace default nginx config with our SPA config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from builder
COPY --from=builder /build/dist /usr/share/nginx/html

# nginx runs as the default nginx user — no privilege escalation needed
EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
